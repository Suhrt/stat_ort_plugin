#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "stat_ort_plugin.h"
#include <onnxruntime_c_api.h>

const OrtApi* g_ort = NULL;

struct VaaniPipeline {
    OrtEnv*          env;
    OrtSession*      encoder;
    OrtSession*      decoder;
    OrtMemoryInfo*   memory_info;
    OrtRunOptions*   dec_run_opts;
    MelProcessor*    processor;
    char**           vocab;
    int              vocab_size;
};

#define ORT_RETURN_NULL_ON_ERR(expr) \
    do { \
        OrtStatus* _s = (expr); \
        if (_s != NULL) { \
            LOGE("ORT error: %s", g_ort->GetErrorMessage(_s)); \
            g_ort->ReleaseStatus(_s); \
            return NULL; \
        } \
    } while (0)

#define ORT_GOTO_CLEANUP_ON_ERR(expr) \
    do { \
        OrtStatus* _s = (expr); \
        if (_s != NULL) { \
            LOGE("ORT error: %s", g_ort->GetErrorMessage(_s)); \
            g_ort->ReleaseStatus(_s); \
            goto cleanup; \
        } \
    } while (0)

/* ─────────────────────────────────────────────
   Init
   ───────────────────────────────────────────── */
FFI_EXPORT VaaniPipeline* vaani_pipeline_init(
        const char* enc_path, const char* dec_path, const char* vocab_path,
        int encoder_threads
) {
    if (!g_ort) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!g_ort) return NULL;
    }

    VaaniPipeline* p = calloc(1, sizeof(VaaniPipeline));
    if (!p) return NULL;

    ORT_RETURN_NULL_ON_ERR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vaani", &p->env));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &p->memory_info));

    /* ── Encoder session ── */
    OrtSessionOptions* enc_opts;
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSessionOptions(&enc_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetSessionGraphOptimizationLevel(enc_opts, ORT_ENABLE_ALL));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetIntraOpNumThreads(enc_opts, encoder_threads));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetInterOpNumThreads(enc_opts, encoder_threads));

    ORT_RETURN_NULL_ON_ERR(g_ort->DisableMemPattern(enc_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSession(p->env, enc_path, enc_opts, &p->encoder));
    g_ort->ReleaseSessionOptions(enc_opts);

    /* ── Decoder session ── */
    OrtSessionOptions* dec_opts;
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSessionOptions(&dec_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetSessionGraphOptimizationLevel(dec_opts, ORT_ENABLE_ALL));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetIntraOpNumThreads(dec_opts, 1));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetInterOpNumThreads(dec_opts, 1));

    ORT_RETURN_NULL_ON_ERR(g_ort->EnableMemPattern(dec_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSession(p->env, dec_path, dec_opts, &p->decoder));
    g_ort->ReleaseSessionOptions(dec_opts);

    /* ── Decoder run options (created once, reused every iteration) ── */
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateRunOptions(&p->dec_run_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->RunOptionsSetRunLogSeverityLevel(p->dec_run_opts, 3));

    /* ── Vocab + mel processor ── */
    p->vocab = load_vocab(vocab_path, &p->vocab_size);
    if (!p->vocab) {
        vaani_pipeline_free(p);
        return NULL;
    }

    p->processor = mel_processor_new(16000, 512, 160, 400, 128);
    if (!p->processor) {
        vaani_pipeline_free(p);
        return NULL;
    }

    return p;
}

/* ─────────────────────────────────────────────
   Free
   ───────────────────────────────────────────── */
FFI_EXPORT void vaani_pipeline_free(VaaniPipeline* p) {
    if (!p) return;
    if (p->vocab) {
        for (int i = 0; i < p->vocab_size; i++) free(p->vocab[i]);
        free(p->vocab);
    }
    if (p->processor)    mel_processor_free(p->processor);
    if (p->dec_run_opts) g_ort->ReleaseRunOptions(p->dec_run_opts);
    if (p->memory_info)  g_ort->ReleaseMemoryInfo(p->memory_info);
    if (p->encoder)      g_ort->ReleaseSession(p->encoder);
    if (p->decoder)      g_ort->ReleaseSession(p->decoder);
    if (p->env)          g_ort->ReleaseEnv(p->env);
    free(p);
}

/* ─────────────────────────────────────────────
   Helpers
   ───────────────────────────────────────────── */
FFI_EXPORT void vaani_string_free(char* str) {
    free(str);
}

static char* create_empty_string(void) {
    char* s = malloc(1);
    if (s) s[0] = '\0';
    return s;
}

/* ─────────────────────────────────────────────
   Transcribe
   ───────────────────────────────────────────── */
FFI_EXPORT char* vaani_pipeline_transcribe(VaaniPipeline* p, const char* wav_path) {
    if (!p || !wav_path) return create_empty_string();

    /* declare all owned resources up front so cleanup: can safely release them */
    int16_t*  pcm_data        = NULL;
    float*    samples         = NULL;
    float*    mel             = NULL;
    float*    mel_t           = NULL;
    float*    transposed_enc  = NULL;
    float*    state_block     = NULL;
    char*     result_str      = NULL;
    OrtValue* enc_inputs[2]   = {NULL, NULL};
    OrtValue* enc_out         = NULL;
    OrtValue* t_last_token    = NULL;
    OrtValue* t_target_len    = NULL;
    OrtValue* t_h1_A          = NULL;
    OrtValue* t_h2_A          = NULL;
    OrtValue* t_h1_B          = NULL;
    OrtValue* t_h2_B          = NULL;

    /* ── 1. WAV read ── */
    FILE* file = fopen(wav_path, "rb");
    if (!file) return create_empty_string();

    unsigned char chunk_header[8];
    int data_size = 0;
    fseek(file, 12, SEEK_SET);
    while (fread(chunk_header, 1, 8, file) == 8) {
        int chunk_size = (int)(chunk_header[4]         |
                               (chunk_header[5] <<  8) |
                               (chunk_header[6] << 16) |
                               (chunk_header[7] << 24));
        if (memcmp(chunk_header, "data", 4) == 0) { data_size = chunk_size; break; }
        fseek(file, chunk_size, SEEK_CUR);
    }
    if (data_size <= 0) { fclose(file); return create_empty_string(); }

    int num_samples = data_size / 2;
    pcm_data = malloc(data_size);
    if (!pcm_data) { fclose(file); return create_empty_string(); }
    fread(pcm_data, 1, data_size, file);
    fclose(file);

    samples = malloc((size_t)num_samples * sizeof(float));
    if (!samples) goto cleanup;
    for (int i = 0; i < num_samples; i++)
        samples[i] = pcm_data[i] * (1.0f / 32768.0f);
    free(pcm_data); pcm_data = NULL;

    /* ── 2. Mel extraction ── */
    int seq_len = 0;
    mel = mel_processor_extract(p->processor, samples, num_samples, &seq_len);
    free(samples); samples = NULL;
    if (!mel || seq_len <= 0) goto cleanup;

    const int feat_dim = 128;
    mel_t = malloc((size_t)feat_dim * seq_len * sizeof(float));
    if (!mel_t) goto cleanup;
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < feat_dim; j++)
            mel_t[j * seq_len + i] = mel[i * feat_dim + j];
    free(mel);
    mel = NULL;

    /* ── 3. Encoder (runs once) ── */
    int64_t enc_in_dims[]  = {1, feat_dim, seq_len};
    int64_t enc_len_dims[] = {1};
    int64_t len_val        = seq_len;

    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, mel_t, (size_t)feat_dim * seq_len * sizeof(float),
            enc_in_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_inputs[0]));
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, &len_val, sizeof(int64_t),
            enc_len_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &enc_inputs[1]));

    const char* enc_in_names[]  = {"audio_signal", "length"};
    const char* enc_out_names[] = {"outputs"};
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->Run(p->encoder, NULL,
                                       enc_in_names, (const OrtValue* const*)enc_inputs, 2,
                                       enc_out_names, 1, &enc_out));

    float* enc_out_data = NULL;
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->GetTensorMutableData(enc_out, (void**)&enc_out_data));
    OrtTensorTypeAndShapeInfo* enc_out_info = NULL;
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->GetTensorTypeAndShape(enc_out, &enc_out_info));
    int64_t enc_out_shape[3];
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->GetDimensions(enc_out_info, enc_out_shape, 3));
    g_ort->ReleaseTensorTypeAndShapeInfo(enc_out_info);

    const int hidden_dim = (int)enc_out_shape[1];
    const int time_steps = (int)enc_out_shape[2];

    transposed_enc = malloc((size_t)time_steps * hidden_dim * sizeof(float));
    if (!transposed_enc) goto cleanup;
    for (int t2 = 0; t2 < time_steps; t2++)
        for (int d = 0; d < hidden_dim; d++)
            transposed_enc[t2 * hidden_dim + d] = enc_out_data[d * time_steps + t2];

    /* encoder resources no longer needed — free before the loop */
    if (enc_inputs[0]) { g_ort->ReleaseValue(enc_inputs[0]); enc_inputs[0] = NULL; }
    if (enc_inputs[1]) { g_ort->ReleaseValue(enc_inputs[1]); enc_inputs[1] = NULL; }
    if (enc_out)       { g_ort->ReleaseValue(enc_out);       enc_out       = NULL; }
    free(mel_t); mel_t = NULL;

    /* ── 4. Decoder loop ── */
    const int blank_id    = 3000;
    const int state_size  = 640;
    const int durations[] = {0, 1, 2, 3, 4};

    int out_buf_cap = time_steps * 8 + 64;
    result_str = malloc(out_buf_cap);
    if (!result_str) goto cleanup;
    result_str[0] = '\0';
    int out_buf_len = 0;

    state_block = calloc((size_t)4 * state_size, sizeof(float));
    if (!state_block) goto cleanup;
    float* h1_A = state_block;
    float* h2_A = state_block +     state_size;
    float* h1_B = state_block + 2 * state_size;
    float* h2_B = state_block + 3 * state_size;

    int32_t last_token = 0;
    int32_t target_len = 1;

    int64_t frame_dims[]  = {1, hidden_dim, 1};
    int64_t scalar_dims[] = {1, 1};
    int64_t len_dims[]    = {1};
    int64_t state_dims[]  = {1, 1, state_size};

    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, &last_token, sizeof(int32_t),
            scalar_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &t_last_token));
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, &target_len, sizeof(int32_t),
            len_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &t_target_len));

    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, h1_A, (size_t)state_size * sizeof(float),
            state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h1_A));
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, h2_A, (size_t)state_size * sizeof(float),
            state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h2_A));
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, h1_B, (size_t)state_size * sizeof(float),
            state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h1_B));
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, h2_B, (size_t)state_size * sizeof(float),
            state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h2_B));

    OrtValue* cur_h1_in  = t_h1_A;
    OrtValue* cur_h2_in  = t_h2_A;
    OrtValue* cur_h1_out = t_h1_B;
    OrtValue* cur_h2_out = t_h2_B;

    const char* dec_in_names[]  = {"encoder_outputs", "targets", "target_length",
                                   "input_states_1",  "input_states_2"};
    const char* dec_out_names[] = {"outputs", "prednet_lengths",
                                   "output_states_1", "output_states_2"};

    int t = 0, same_frame_count = 0;

    while (t < time_steps) {
        /*
         * Point directly into transposed_enc at the current frame offset —
         * avoids a memcpy per iteration. OrtValue wrapping is cheap (no data
         * copy); we release it at the end of each step.
         */
        OrtValue* t_frame = NULL;
        ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
                p->memory_info,
                &transposed_enc[t * hidden_dim],
                (size_t)hidden_dim * sizeof(float),
                frame_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_frame));

        OrtValue* dec_inputs[5]  = {t_frame, t_last_token, t_target_len,
                                    cur_h1_in, cur_h2_in};
        OrtValue* dec_outputs[4] = {NULL, NULL, cur_h1_out, cur_h2_out};

        OrtStatus* run_status = g_ort->Run(p->decoder, p->dec_run_opts,
                                           dec_in_names,  (const OrtValue* const*)dec_inputs,  5,
                                           dec_out_names, 4, dec_outputs);

        if (run_status != NULL) {
            LOGE("ORT decoder error: %s", g_ort->GetErrorMessage(run_status));
            g_ort->ReleaseStatus(run_status);
            g_ort->ReleaseValue(t_frame);
            goto cleanup;
        }

        float* logits = NULL;
        OrtStatus* get_status = g_ort->GetTensorMutableData(dec_outputs[0], (void**)&logits);
        if (get_status != NULL) {
            LOGE("ORT error: %s", g_ort->GetErrorMessage(get_status));
            g_ort->ReleaseStatus(get_status);
            g_ort->ReleaseValue(t_frame);
            g_ort->ReleaseValue(dec_outputs[0]);
            if (dec_outputs[1]) g_ort->ReleaseValue(dec_outputs[1]);
            goto cleanup;
        }

        float max_val = -INFINITY;
        int token_id = 0;
        for (int i = 0; i <= blank_id; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; token_id = i; }
        }

        float max_dur_val = -INFINITY;
        int dur_idx = 0;
        for (int i = 0; i < 5; i++) {
            if (logits[blank_id + 1 + i] > max_dur_val) {
                max_dur_val = logits[blank_id + 1 + i];
                dur_idx = i;
            }
        }

        g_ort->ReleaseValue(dec_outputs[0]);
        if (dec_outputs[1]) g_ort->ReleaseValue(dec_outputs[1]);
        g_ort->ReleaseValue(t_frame);

        int duration = durations[dur_idx];
        if (duration == 0) {
            if (++same_frame_count > 10) { duration = 1; same_frame_count = 0; }
        } else {
            same_frame_count = 0;
        }

        if (token_id == blank_id) {
            t += (duration > 1) ? duration : 1;
        } else {
            if (token_id < p->vocab_size) {
                const char* token = p->vocab[token_id];
                if ((unsigned char)token[0] == 0xE2 &&
                    (unsigned char)token[1] == 0x96 &&
                    (unsigned char)token[2] == 0x81 &&
                    token[3] == '\0')
                    token = " ";

                int len = (int)strlen(token);
                if (out_buf_len + len >= out_buf_cap) {
                    out_buf_cap = (out_buf_len + len + 64) * 2;
                    char* tmp = realloc(result_str, out_buf_cap);
                    if (!tmp) goto cleanup;
                    result_str = tmp;
                }
                memcpy(result_str + out_buf_len, token, len);
                out_buf_len += len;
                result_str[out_buf_len] = '\0';
            }
            last_token = token_id;

            OrtValue* tmp;
            tmp = cur_h1_in; cur_h1_in = cur_h1_out; cur_h1_out = tmp;
            tmp = cur_h2_in; cur_h2_in = cur_h2_out; cur_h2_out = tmp;

            t += duration;
        }
    }

    {
        char* trimmed = realloc(result_str, out_buf_len + 1);
        if (trimmed) result_str = trimmed;
        normalize_whitespace(result_str);
    }

    cleanup:
    free(pcm_data);
    free(samples);
    if (mel)  free(mel);
    free(mel_t);
    free(transposed_enc);
    free(state_block);

    if (enc_inputs[0]) g_ort->ReleaseValue(enc_inputs[0]);
    if (enc_inputs[1]) g_ort->ReleaseValue(enc_inputs[1]);
    if (enc_out)       g_ort->ReleaseValue(enc_out);
    if (t_last_token)  g_ort->ReleaseValue(t_last_token);
    if (t_target_len)  g_ort->ReleaseValue(t_target_len);
    if (t_h1_A)        g_ort->ReleaseValue(t_h1_A);
    if (t_h2_A)        g_ort->ReleaseValue(t_h2_A);
    if (t_h1_B)        g_ort->ReleaseValue(t_h1_B);
    if (t_h2_B)        g_ort->ReleaseValue(t_h2_B);

    return result_str ? result_str : create_empty_string();
}