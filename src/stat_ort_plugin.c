#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "stat_ort_plugin.h"
#include <onnxruntime_c_api.h>

const OrtApi* g_ort = NULL;

struct VaaniPipeline {
    OrtEnv* env;
    OrtSession* encoder;
    OrtSession* decoder;
    MelProcessor* processor;
    char** vocab;
    int vocab_size;
    OrtMemoryInfo* memory_info;
};

/* Removed LOGE from the error handling macro */
#define ORT_RETURN_NULL_ON_ERR(expr) \
    do { \
        OrtStatus* status = (expr); \
        if (status != NULL) { \
            g_ort->ReleaseStatus(status); \
            return NULL; \
        } \
    } while (0)

FFI_EXPORT VaaniPipeline* vaani_pipeline_init(
        const char* enc_path, const char* dec_path, const char* vocab_path,
        int encoder_threads, int decoder_threads
) {
    if (!g_ort) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!g_ort) {
            return NULL;
        }
    }

    VaaniPipeline* p = calloc(1, sizeof(VaaniPipeline));

    ORT_RETURN_NULL_ON_ERR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vaani", &p->env));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &p->memory_info));

    OrtSessionOptions* enc_opts;
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSessionOptions(&enc_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetSessionGraphOptimizationLevel(enc_opts, ORT_ENABLE_ALL));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetIntraOpNumThreads(enc_opts, encoder_threads));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSession(p->env, enc_path, enc_opts, &p->encoder));
    g_ort->ReleaseSessionOptions(enc_opts);

    OrtSessionOptions* dec_opts;
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSessionOptions(&dec_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetSessionGraphOptimizationLevel(dec_opts, ORT_ENABLE_ALL));
    ORT_RETURN_NULL_ON_ERR(g_ort->SetIntraOpNumThreads(dec_opts, decoder_threads));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateSession(p->env, dec_path, dec_opts, &p->decoder));
    g_ort->ReleaseSessionOptions(dec_opts);

    p->vocab = load_vocab(vocab_path, &p->vocab_size);
    if (!p->vocab) {
        vaani_pipeline_free(p);
        return NULL;
    }

    p->processor = mel_processor_new(16000, 512, 160, 400, 128);

    return p;
}

FFI_EXPORT void vaani_pipeline_free(VaaniPipeline* p) {
    if (!p) return;
    for (int i = 0; i < p->vocab_size; i++) free(p->vocab[i]);
    free(p->vocab);
    mel_processor_free(p->processor);
    g_ort->ReleaseMemoryInfo(p->memory_info);
    g_ort->ReleaseSession(p->encoder);
    g_ort->ReleaseSession(p->decoder);
    g_ort->ReleaseEnv(p->env);
    free(p);
}

FFI_EXPORT void vaani_string_free(char* str) {
    if (str) free(str);
}

static char* create_empty_string() {
    char* empty = malloc(1);
    empty[0] = '\0';
    return empty;
}

FFI_EXPORT char* vaani_pipeline_transcribe(VaaniPipeline* p, const char* wav_path) {
    if (!p) {
        return create_empty_string();
    }
    if (!wav_path) {
        return create_empty_string();
    }

    FILE* file = fopen(wav_path, "rb");
    if (!file) {
        return create_empty_string();
    }

    unsigned char chunk_header[8];
    int data_size = 0;

    fseek(file, 12, SEEK_SET);
    while (fread(chunk_header, 1, 8, file) == 8) {
        int chunk_size = chunk_header[4] | (chunk_header[5] << 8) | (chunk_header[6] << 16) | (chunk_header[7] << 24);

        if (strncmp((char*)chunk_header, "data", 4) == 0) {
            data_size = chunk_size;
            break;
        }
        fseek(file, chunk_size, SEEK_CUR);
    }

    if (data_size <= 0) {
        fclose(file);
        return create_empty_string();
    }

    int num_samples = data_size / 2;

    int16_t* pcm_data = malloc(data_size);
    fread(pcm_data, 1, data_size, file);
    fclose(file);

    float* samples = malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; i++) {
        samples[i] = pcm_data[i] / 32768.0f;
    }
    free(pcm_data);

    int seq_len = 0;
    float* mel = mel_processor_extract(p->processor, samples, num_samples, &seq_len);
    free(samples);

    if (!mel || seq_len <= 0) {
        return create_empty_string();
    }

    int feat_dim = 128;

    float* mel_t = malloc(feat_dim * seq_len * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < feat_dim; j++) {
            mel_t[j * seq_len + i] = mel[i * feat_dim + j];
        }
    }
    free(mel);

    int64_t enc_in_dims[] = {1, feat_dim, seq_len};
    int64_t enc_len_dims[] = {1};
    int64_t len_val = seq_len;

    OrtValue* enc_inputs[2] = {NULL, NULL};
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, mel_t, feat_dim * seq_len * sizeof(float), enc_in_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_inputs[0]));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, &len_val, sizeof(int64_t), enc_len_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &enc_inputs[1]));

    const char* enc_in_names[] = {"audio_signal", "length"};
    const char* enc_out_names[] = {"outputs"};
    OrtValue* enc_out = NULL;

    ORT_RETURN_NULL_ON_ERR(g_ort->Run(p->encoder, NULL, enc_in_names, (const OrtValue* const*)enc_inputs, 2, enc_out_names, 1, &enc_out));

    float* enc_out_data = NULL;
    ORT_RETURN_NULL_ON_ERR(g_ort->GetTensorMutableData(enc_out, (void**)&enc_out_data));
    OrtTensorTypeAndShapeInfo* enc_out_info = NULL;
    ORT_RETURN_NULL_ON_ERR(g_ort->GetTensorTypeAndShape(enc_out, &enc_out_info));
    int64_t enc_out_shape[3];
    ORT_RETURN_NULL_ON_ERR(g_ort->GetDimensions(enc_out_info, enc_out_shape, 3));
    g_ort->ReleaseTensorTypeAndShapeInfo(enc_out_info);

    int hidden_dim = (int)enc_out_shape[1];
    int time_steps = (int)enc_out_shape[2];


    int blank_id = 3000;
    int durations[] = {0, 1, 2, 3, 4};
    int state_size = 640;

    int out_buf_cap = time_steps * 32 + 1;
    char* result_str = malloc(out_buf_cap);
    result_str[0] = '\0';
    int out_buf_len = 0;

    float* frame = malloc(hidden_dim * sizeof(float));
    float* h1_A = calloc(state_size, sizeof(float));
    float* h2_A = calloc(state_size, sizeof(float));
    float* h1_B = calloc(state_size, sizeof(float));
    float* h2_B = calloc(state_size, sizeof(float));

    int32_t last_token = 0;
    int32_t target_len = 1;

    int64_t frame_dims[] = {1, hidden_dim, 1};
    int64_t scalar_dims[] = {1, 1};
    int64_t len_dims[] = {1};
    int64_t state_dims[] = {1, 1, state_size};

    OrtValue *t_frame, *t_last_token, *t_target_len;
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, frame, hidden_dim * sizeof(float), frame_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_frame));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, &last_token, sizeof(int32_t), scalar_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &t_last_token));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, &target_len, sizeof(int32_t), len_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &t_target_len));

    OrtValue *t_h1_A, *t_h2_A, *t_h1_B, *t_h2_B;
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, h1_A, state_size * sizeof(float), state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h1_A));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, h2_A, state_size * sizeof(float), state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h2_A));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, h1_B, state_size * sizeof(float), state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h1_B));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(p->memory_info, h2_B, state_size * sizeof(float), state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h2_B));

    OrtValue* current_h1_in = t_h1_A;
    OrtValue* current_h2_in = t_h2_A;
    OrtValue* current_h1_out = t_h1_B;
    OrtValue* current_h2_out = t_h2_B;

    const char* dec_in_names[] = {"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"};
    const char* dec_out_names[] = {"outputs", "prednet_lengths", "output_states_1", "output_states_2"};

    int t = 0;
    int same_frame_count = 0;

    OrtValue* t_logits = NULL;
    OrtValue* t_pred_len = NULL;

    while (t < time_steps) {
        for (int d = 0; d < hidden_dim; d++) {
            frame[d] = enc_out_data[d * time_steps + t];
        }

        OrtValue* dec_inputs[5] = {t_frame, t_last_token, t_target_len, current_h1_in, current_h2_in};
        OrtValue* dec_outputs[4] = {t_logits, t_pred_len, current_h1_out, current_h2_out};

        ORT_RETURN_NULL_ON_ERR(g_ort->Run(p->decoder, NULL, dec_in_names, (const OrtValue* const*)dec_inputs, 5, dec_out_names, 4, dec_outputs));

        float* logits = NULL;
        ORT_RETURN_NULL_ON_ERR(g_ort->GetTensorMutableData(dec_outputs[0], (void**)&logits));

        float max_val = -INFINITY;
        int token_id = 0;
        for (int i = 0; i <= blank_id; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                token_id = i;
            }
        }

        float max_dur_val = -INFINITY;
        int dur_idx = 0;
        for (int i = 0; i < 5; i++) {
            if (logits[blank_id + 1 + i] > max_dur_val) {
                max_dur_val = logits[blank_id + 1 + i];
                dur_idx = i;
            }
        }

        int duration = durations[dur_idx];

        if (duration == 0) {
            same_frame_count++;
            if (same_frame_count > 10) {
                duration = 1;
                same_frame_count = 0;
            }
        } else {
            same_frame_count = 0;
        }

        if (token_id == blank_id) {
            t += (duration > 1) ? duration : 1;
        } else {
            if (token_id < p->vocab_size) {
                const char* token = p->vocab[token_id];
                if (strcmp(token, "▁") == 0) token = " ";

                int len = strlen(token);

                if (out_buf_len + len >= out_buf_cap) {
                    out_buf_cap = (out_buf_len + len) * 2;
                    result_str = realloc(result_str, out_buf_cap);
                }

                strcpy(result_str + out_buf_len, token);
                out_buf_len += len;
            }
            last_token = token_id;

            OrtValue* temp_h1 = current_h1_in;
            current_h1_in = current_h1_out;
            current_h1_out = temp_h1;

            OrtValue* temp_h2 = current_h2_in;
            current_h2_in = current_h2_out;
            current_h2_out = temp_h2;

            t += duration;
        }

        g_ort->ReleaseValue(dec_outputs[0]);
        if (dec_outputs[1]) g_ort->ReleaseValue(dec_outputs[1]);
    }

    result_str = realloc(result_str, out_buf_len + 1);
    normalize_whitespace(result_str);

    g_ort->ReleaseValue(t_frame);
    g_ort->ReleaseValue(t_last_token);
    g_ort->ReleaseValue(t_target_len);
    g_ort->ReleaseValue(t_h1_A);
    g_ort->ReleaseValue(t_h2_A);
    g_ort->ReleaseValue(t_h1_B);
    g_ort->ReleaseValue(t_h2_B);

    free(frame);
    free(h1_A);
    free(h2_A);
    free(h1_B);
    free(h2_B);

    free(mel_t);
    g_ort->ReleaseValue(enc_inputs[0]);
    g_ort->ReleaseValue(enc_inputs[1]);
    g_ort->ReleaseValue(enc_out);

    return result_str;
}