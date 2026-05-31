#include "vaani_internal.h"
#include "stat_ort_plugin.h"
#include <math.h>

const OrtApi* g_ort = NULL;

FFI_EXPORT VaaniPipeline* vaani_pipeline_init(
        const char* enc_path, const char* dec_path, const char* vocab_path,
        const char* vad_path, const char* speaker_path, int encoder_threads
) {
    if (!g_ort) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!g_ort) return NULL;
    }

    VaaniPipeline* p = calloc(1, sizeof(VaaniPipeline));
    if (!p) return NULL;

    p->processor = mel_processor_new(SAMPLE_RATE, 512, 160, 400, 128);
    if (!p->processor) {
        LOGE("vaani_pipeline_init: Failed to create ASR mel processor");
        vaani_pipeline_free(p);
        return NULL;
    }

    if (speaker_path) {
        p->speaker_processor = mel_processor_new(SAMPLE_RATE, 512, 160, 400, 80);
        if (!p->speaker_processor) {
            LOGE("vaani_pipeline_init: Failed to create speaker mel processor");
            vaani_pipeline_free(p);
            return NULL;
        }
    }

    ORT_RETURN_NULL_ON_ERR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vaani", &p->env));
    ORT_RETURN_NULL_ON_ERR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &p->memory_info));

    {
        OrtSessionOptions* enc_opts = NULL;
        OrtStatus* s = g_ort->CreateSessionOptions(&enc_opts);
        if (s) { g_ort->ReleaseStatus(s); vaani_pipeline_free(p); return NULL; }
        g_ort->SetSessionGraphOptimizationLevel(enc_opts, ORT_ENABLE_ALL);
        g_ort->SetIntraOpNumThreads(enc_opts, encoder_threads);
        g_ort->SetInterOpNumThreads(enc_opts, encoder_threads);
        g_ort->DisableMemPattern(enc_opts);
        s = g_ort->CreateSession(p->env, enc_path, enc_opts, &p->encoder);
        g_ort->ReleaseSessionOptions(enc_opts);
        if (s) {
            LOGE("vaani_pipeline_init: Failed to create encoder session: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
            vaani_pipeline_free(p);
            return NULL;
        }
        LOGD("vaani: [INIT] Encoder session created: %s", enc_path);
    }

    {
        OrtSessionOptions* dec_opts = NULL;
        OrtStatus* s = g_ort->CreateSessionOptions(&dec_opts);
        if (s) { g_ort->ReleaseStatus(s); vaani_pipeline_free(p); return NULL; }
        g_ort->SetSessionGraphOptimizationLevel(dec_opts, ORT_ENABLE_ALL);
        g_ort->SetIntraOpNumThreads(dec_opts, 1);
        g_ort->SetInterOpNumThreads(dec_opts, 1);
        g_ort->EnableMemPattern(dec_opts);
        s = g_ort->CreateSession(p->env, dec_path, dec_opts, &p->decoder);
        g_ort->ReleaseSessionOptions(dec_opts);
        if (s) {
            LOGE("vaani_pipeline_init: Failed to create decoder session: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
            vaani_pipeline_free(p);
            return NULL;
        }
        LOGD("vaani: [INIT] Decoder session created: %s", dec_path);
    }

    if (vad_path) {
        OrtSessionOptions* vad_opts = NULL;
        OrtStatus* s = g_ort->CreateSessionOptions(&vad_opts);
        if (s) { g_ort->ReleaseStatus(s); vaani_pipeline_free(p); return NULL; }
        g_ort->SetSessionGraphOptimizationLevel(vad_opts, ORT_ENABLE_ALL);
        g_ort->SetIntraOpNumThreads(vad_opts, 1);
        g_ort->SetInterOpNumThreads(vad_opts, 1);
        s = g_ort->CreateSession(p->env, vad_path, vad_opts, &p->vad);
        g_ort->ReleaseSessionOptions(vad_opts);
        if (s) {
            LOGE("vaani_pipeline_init: Failed to create VAD session: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
            vaani_pipeline_free(p);
            return NULL;
        }
        LOGD("vaani: [INIT] VAD session created: %s", vad_path);
    } else {
        LOGD("vaani: [INIT] No VAD path provided — VAD disabled");
    }

    if (speaker_path) {
        OrtSessionOptions* spk_opts = NULL;
        OrtStatus* s = g_ort->CreateSessionOptions(&spk_opts);
        if (s) { g_ort->ReleaseStatus(s); vaani_pipeline_free(p); return NULL; }
        g_ort->SetSessionGraphOptimizationLevel(spk_opts, ORT_ENABLE_ALL);
        g_ort->SetIntraOpNumThreads(spk_opts, 1);
        g_ort->SetInterOpNumThreads(spk_opts, 1);
        s = g_ort->CreateSession(p->env, speaker_path, spk_opts, &p->speaker);
        g_ort->ReleaseSessionOptions(spk_opts);
        if (s) {
            LOGE("vaani_pipeline_init: Failed to create speaker session: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
            vaani_pipeline_free(p);
            return NULL;
        }
        LOGD("vaani: [INIT] Speaker session created: %s", speaker_path);
    } else {
        LOGD("vaani: [INIT] No speaker path provided — diarization disabled");
    }

    ORT_RETURN_NULL_ON_ERR(g_ort->CreateRunOptions(&p->dec_run_opts));
    ORT_RETURN_NULL_ON_ERR(g_ort->RunOptionsSetRunLogSeverityLevel(p->dec_run_opts, 3));

    p->vocab = load_vocab(vocab_path, &p->vocab_size);
    if (!p->vocab) {
        LOGE("vaani_pipeline_init: Failed to load vocab from %s", vocab_path);
        vaani_pipeline_free(p);
        return NULL;
    }

    LOGD("vaani: [INIT] Pipeline ready. encoder_threads=%d vocab_size=%d vad=%s speaker=%s",
         encoder_threads, p->vocab_size,
         p->vad ? "YES" : "NO",
         p->speaker ? "YES" : "NO");
    return p;
}

FFI_EXPORT void vaani_pipeline_free(VaaniPipeline* p) {
    if (!p) return;
    LOGD("vaani: [FREE] Releasing pipeline");
    if (p->vocab) {
        for (int i = 0; i < p->vocab_size; i++) free(p->vocab[i]);
        free(p->vocab);
    }
    if (p->processor)         mel_processor_free(p->processor);
    if (p->speaker_processor) mel_processor_free(p->speaker_processor);
    if (p->dec_run_opts)      g_ort->ReleaseRunOptions(p->dec_run_opts);
    if (p->memory_info)       g_ort->ReleaseMemoryInfo(p->memory_info);
    if (p->encoder)           g_ort->ReleaseSession(p->encoder);
    if (p->decoder)           g_ort->ReleaseSession(p->decoder);
    if (p->vad)               g_ort->ReleaseSession(p->vad);
    if (p->speaker)           g_ort->ReleaseSession(p->speaker);
    if (p->env)               g_ort->ReleaseEnv(p->env);
    free(p);
    LOGD("vaani: [FREE] Pipeline released");
}

char* decode_audio_buffer(VaaniPipeline* p, float* samples, int num_samples) {
    if (!p || !p->processor || !samples || num_samples <= 0) {
        LOGE("vaani: [DECODE] Invalid input: p=%p processor=%p samples=%p num_samples=%d",
             p, p ? p->processor : NULL, samples, num_samples);
        return create_empty_string();
    }

    LOGD("vaani: [DECODE] Starting decode: num_samples=%d (%.2fs)",
         num_samples, (float)num_samples / SAMPLE_RATE);

    float* mel             = NULL;
    float* mel_t           = NULL;
    float* transposed_enc  = NULL;
    float* state_block     = NULL;
    char*  result_str      = NULL;
    OrtValue* enc_inputs[2] = {NULL, NULL};
    OrtValue* enc_out      = NULL;
    OrtValue* t_last_token = NULL;
    OrtValue* t_target_len = NULL;
    OrtValue* t_h1_A       = NULL;
    OrtValue* t_h2_A       = NULL;
    OrtValue* t_h1_B       = NULL;
    OrtValue* t_h2_B       = NULL;

    int seq_len = 0;
    mel = mel_processor_extract(p->processor, samples, num_samples, &seq_len);
    if (!mel || seq_len <= 0) {
        LOGE("vaani: [DECODE] mel extraction failed: mel=%p seq_len=%d", mel, seq_len);
        goto cleanup;
    }
    LOGD("vaani: [DECODE] Mel extracted: seq_len=%d frames", seq_len);

    const int feat_dim = 128;
    mel_t = malloc((size_t)feat_dim * seq_len * sizeof(float));
    if (!mel_t) { LOGE("vaani: [DECODE] malloc mel_t failed"); goto cleanup; }
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < feat_dim; j++)
            mel_t[j * seq_len + i] = mel[i * feat_dim + j];
    free(mel);
    mel = NULL;

    int64_t enc_in_dims[]  = {1, feat_dim, seq_len};
    int64_t enc_len_dims[] = {1};
    int64_t len_val        = seq_len;

    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, mel_t, (size_t)feat_dim * seq_len * sizeof(float),
            enc_in_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_inputs[0]));
    ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, &len_val, sizeof(int64_t),
            enc_len_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &enc_inputs[1]));

    LOGD("vaani: [DECODE] Running encoder: input shape [1, %d, %d]", feat_dim, seq_len);
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
    LOGD("vaani: [DECODE] Encoder output: shape [%lld, %d, %d] hidden_dim=%d time_steps=%d",
         enc_out_shape[0], hidden_dim, time_steps, hidden_dim, time_steps);

    transposed_enc = malloc((size_t)time_steps * hidden_dim * sizeof(float));
    if (!transposed_enc) { LOGE("vaani: [DECODE] malloc transposed_enc failed"); goto cleanup; }
    for (int t2 = 0; t2 < time_steps; t2++)
        for (int d = 0; d < hidden_dim; d++)
            transposed_enc[t2 * hidden_dim + d] = enc_out_data[d * time_steps + t2];

    if (enc_inputs[0]) { g_ort->ReleaseValue(enc_inputs[0]); enc_inputs[0] = NULL; }
    if (enc_inputs[1]) { g_ort->ReleaseValue(enc_inputs[1]); enc_inputs[1] = NULL; }
    if (enc_out)       { g_ort->ReleaseValue(enc_out);       enc_out       = NULL; }
    free(mel_t); mel_t = NULL;

    const int blank_id    = 3000;
    const int state_size  = 640;
    const int durations[] = {0, 1, 2, 3, 4};

    int out_buf_cap = time_steps * 8 + 64;
    result_str = malloc(out_buf_cap);
    if (!result_str) { LOGE("vaani: [DECODE] malloc result_str failed"); goto cleanup; }
    result_str[0] = '\0';
    int out_buf_len = 0;

    state_block = calloc((size_t)4 * state_size, sizeof(float));
    if (!state_block) { LOGE("vaani: [DECODE] malloc state_block failed"); goto cleanup; }
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

    LOGD("vaani: [DECODE] Starting decoder loop: time_steps=%d", time_steps);

    OrtValue* cur_h1_in  = t_h1_A;
    OrtValue* cur_h2_in  = t_h2_A;
    OrtValue* cur_h1_out = t_h1_B;
    OrtValue* cur_h2_out = t_h2_B;

    const char* dec_in_names[]  = {"encoder_outputs", "targets", "target_length",
                                   "input_states_1",  "input_states_2"};
    const char* dec_out_names[] = {"outputs", "prednet_lengths",
                                   "output_states_1", "output_states_2"};

    int t = 0, same_frame_count = 0;
    int total_tokens_emitted = 0;
    int total_blanks = 0;

    while (t < time_steps) {
        OrtValue* t_frame = NULL;
        ORT_GOTO_CLEANUP_ON_ERR(g_ort->CreateTensorWithDataAsOrtValue(
                p->memory_info,
                &transposed_enc[t * hidden_dim],
                (size_t)hidden_dim * sizeof(float),
                frame_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_frame));

        OrtValue* dec_inputs[5]  = {t_frame, t_last_token, t_target_len, cur_h1_in, cur_h2_in};
        OrtValue* dec_outputs[4] = {NULL, NULL, cur_h1_out, cur_h2_out};

        OrtStatus* run_status = g_ort->Run(p->decoder, p->dec_run_opts,
                                           dec_in_names,  (const OrtValue* const*)dec_inputs,  5,
                                           dec_out_names, 4, dec_outputs);
        if (run_status != NULL) {
            LOGE("vaani: [DECODE] Decoder run failed at t=%d: %s",
                 t, g_ort->GetErrorMessage(run_status));
            g_ort->ReleaseStatus(run_status);
            g_ort->ReleaseValue(t_frame);
            goto cleanup;
        }

        float* logits = NULL;
        OrtStatus* get_status = g_ort->GetTensorMutableData(dec_outputs[0], (void**)&logits);
        if (get_status != NULL) {
            LOGE("vaani: [DECODE] GetTensorMutableData failed at t=%d", t);
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

        if (token_id == blank_id) {
            total_blanks++;
            t += (duration > 1) ? duration : 1;
            same_frame_count = 0;
        } else {
            if (duration == 0) {
                if (++same_frame_count > 10) {
                    LOGD("vaani: [DECODE] same_frame_count > 10 at t=%d token=%d, forcing advance",
                         t, token_id);
                    duration = 1;
                    same_frame_count = 0;
                }
            } else {
                same_frame_count = 0;
            }

            if (token_id < p->vocab_size) {
                const char* token = p->vocab[token_id];
                if ((unsigned char)token[0] == 0xE2 &&
                    (unsigned char)token[1] == 0x96 &&
                    (unsigned char)token[2] == 0x81 &&
                    token[3] == '\0')
                    token = " ";

                int tok_len = (int)strlen(token);
                if (out_buf_len + tok_len >= out_buf_cap) {
                    out_buf_cap = (out_buf_len + tok_len + 64) * 2;
                    char* tmp = realloc(result_str, out_buf_cap);
                    if (!tmp) { LOGE("vaani: [DECODE] realloc result_str failed"); goto cleanup; }
                    result_str = tmp;
                }
                memcpy(result_str + out_buf_len, token, tok_len);
                out_buf_len += tok_len;
                result_str[out_buf_len] = '\0';
                total_tokens_emitted++;
            } else {
                LOGE("vaani: [DECODE] token_id %d out of vocab range %d", token_id, p->vocab_size);
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
    if (mel)           free(mel);
    if (mel_t)         free(mel_t);
    if (transposed_enc) free(transposed_enc);
    if (state_block)   free(state_block);

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

// Run Silero VAD v5 on one 16 kHz audio frame.
//
// Silero v5 requires VAD_CONTEXT_SAMPLES (64) samples carried over from
// the previous call to be prepended to the 512-sample audio frame, so the
// input tensor is actually VAD_INPUT_SAMPLES (576) long. Without that
// prepend the model still runs but its STFT windows misalign and output
// probabilities collapse to near-zero on real speech. The Python
// reference (silero_vad/utils_vad.py OnnxWrapper) does the same thing
// via torch.cat([self._context, x], dim=1).
//
// Returns a probability in [0, 1] on success, or -1.0 on any failure
// (bad inputs, ORT errors). Callers should treat -1.0 as "no opinion this
// frame" — not as silence and not as speech.
float run_silero_vad(struct VaaniPipeline* p, const float* audio, int len) {
    if (!p->vad || !audio || len != VAD_AUDIO_SAMPLES) {
        LOGE("vaani: [VAD] invalid input (vad=%p, audio=%p, len=%d, expected=%d)",
             p->vad, audio, len, VAD_AUDIO_SAMPLES);
        return -1.0f;
    }

    if (!p->vad_state_initialized) {
        memset(p->vad_state,   0, sizeof(p->vad_state));
        memset(p->vad_context, 0, sizeof(p->vad_context));
        p->vad_state_initialized = true;
    }

    // Build the 576-sample input by prepending the last 64 samples from
    // the previous call.
    float vad_input[VAD_INPUT_SAMPLES];
    memcpy(vad_input,                       p->vad_context, VAD_CONTEXT_SAMPLES * sizeof(float));
    memcpy(vad_input + VAD_CONTEXT_SAMPLES, audio,          VAD_AUDIO_SAMPLES   * sizeof(float));

    int64_t   input_shape[] = {1, VAD_INPUT_SAMPLES};
    OrtValue* input_tensor  = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, (void*)vad_input, VAD_INPUT_SAMPLES * sizeof(float),
            input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status) {
        LOGE("vaani: [VAD] input tensor failed: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1.0f;
    }

    int64_t   state_shape[] = {2, 1, 128};
    OrtValue* state_tensor  = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, p->vad_state, sizeof(p->vad_state),
            state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor);
    if (status) {
        LOGE("vaani: [VAD] state tensor failed: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        return -1.0f;
    }

    int64_t   sr_val     = SAMPLE_RATE;
    OrtValue* sr_tensor  = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, &sr_val, sizeof(int64_t),
            NULL, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor);
    if (status) {
        LOGE("vaani: [VAD] sr tensor failed: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(state_tensor);
        return -1.0f;
    }

    const char* input_names[]   = {"input", "state", "sr"};
    OrtValue*   input_values[]  = {input_tensor, state_tensor, sr_tensor};
    const char* output_names[]  = {"output", "stateN"};
    OrtValue*   output_values[] = {NULL, NULL};

    status = g_ort->Run(p->vad, NULL,
                        input_names, (const OrtValue* const*)input_values, 3,
                        output_names, 2, output_values);

    float probability = -1.0f;

    if (status == NULL && output_values[0] != NULL && output_values[1] != NULL) {
        float*     out_data = NULL;
        OrtStatus* s        = g_ort->GetTensorMutableData(output_values[0], (void**)&out_data);
        if (s) {
            LOGE("vaani: [VAD] get output failed: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
        } else {
            probability = out_data[0];
        }

        float* new_state = NULL;
        s = g_ort->GetTensorMutableData(output_values[1], (void**)&new_state);
        if (s) {
            LOGE("vaani: [VAD] get stateN failed: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
        } else {
            memcpy(p->vad_state, new_state, sizeof(p->vad_state));
        }
    } else if (status) {
        LOGE("vaani: [VAD] inference failed: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
    } else {
        LOGE("vaani: [VAD] output tensor NULL");
    }

    // Save the last VAD_CONTEXT_SAMPLES of this chunk for the next call.
    memcpy(p->vad_context,
           audio + (VAD_AUDIO_SAMPLES - VAD_CONTEXT_SAMPLES),
           VAD_CONTEXT_SAMPLES * sizeof(float));

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(state_tensor);
    g_ort->ReleaseValue(sr_tensor);
    if (output_values[0]) g_ort->ReleaseValue(output_values[0]);
    if (output_values[1]) g_ort->ReleaseValue(output_values[1]);

    return probability;
}

void run_speaker_embedding(struct VaaniPipeline* p, const float* audio, int len, float* out_emb) {
    if (!p->speaker || !p->speaker_processor || !audio || len == 0) {
        LOGE("vaani: [SPEAKER] Invalid input: speaker=%p processor=%p audio=%p len=%d",
             p->speaker, p->speaker_processor, audio, len);
        memset(out_emb, 0, EMBEDDING_DIM * sizeof(float));
        return;
    }

    LOGD("vaani: [SPEAKER] Extracting embedding: len=%d samples (%.2fs)",
         len, (float)len / SAMPLE_RATE);

    int seq_len = 0;
    float* fbanks = mel_processor_extract(p->speaker_processor, audio, len, &seq_len);
    if (!fbanks || seq_len <= 0) {
        LOGE("vaani: [SPEAKER] fbank extraction failed: fbanks=%p seq_len=%d", fbanks, seq_len);
        memset(out_emb, 0, EMBEDDING_DIM * sizeof(float));
        if (fbanks) free(fbanks);
        return;
    }
    LOGD("vaani: [SPEAKER] Fbanks extracted: seq_len=%d, input shape [1, %d, 80]", seq_len, seq_len);

    int64_t input_shape[] = {1, seq_len, 80};
    OrtValue* input_tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, (void*)fbanks, (size_t)seq_len * 80 * sizeof(float),
            input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    if (status) {
        LOGE("run_speaker_embedding: input tensor failed: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        free(fbanks);
        memset(out_emb, 0, EMBEDDING_DIM * sizeof(float));
        return;
    }

    const char* input_names[]  = {"feats"};
    const char* output_names[] = {"embs"};
    OrtValue* output_tensor = NULL;

    status = g_ort->Run(p->speaker, NULL,
                        input_names, (const OrtValue* const*)&input_tensor, 1,
                        output_names, 1, &output_tensor);

    if (status == NULL && output_tensor != NULL) {
        float* out_data = NULL;
        OrtStatus* s = g_ort->GetTensorMutableData(output_tensor, (void**)&out_data);
        if (s) {
            LOGE("run_speaker_embedding: GetTensorMutableData failed: %s", g_ort->GetErrorMessage(s));
            g_ort->ReleaseStatus(s);
            memset(out_emb, 0, EMBEDDING_DIM * sizeof(float));
        } else {
            memcpy(out_emb, out_data, EMBEDDING_DIM * sizeof(float));

            float norm = 0.0f;
            for (int i = 0; i < EMBEDDING_DIM; i++) norm += out_emb[i] * out_emb[i];
            norm = sqrtf(norm);
            if (norm > 1e-6f) {
                for (int i = 0; i < EMBEDDING_DIM; i++) out_emb[i] /= norm;
                LOGD("vaani: [SPEAKER] Embedding extracted: norm=%.6f (normalized)", norm);
            } else {
                LOGE("vaani: [SPEAKER] Near-zero embedding norm (%.6f), zeroing", norm);
                memset(out_emb, 0, EMBEDDING_DIM * sizeof(float));
            }
        }
    } else {
        if (status) {
            LOGE("run_speaker_embedding: inference failed: %s", g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
        } else {
            LOGE("run_speaker_embedding: output tensor NULL");
        }
        memset(out_emb, 0, EMBEDDING_DIM * sizeof(float));
    }

    g_ort->ReleaseValue(input_tensor);
    if (output_tensor) g_ort->ReleaseValue(output_tensor);
    free(fbanks);
    LOGD("vaani: [SPEAKER] Embedding done");
}