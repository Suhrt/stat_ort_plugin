#include "vaani_internal.h"
#include "stat_ort_plugin.h"
#include <math.h>

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

    const int feat_dim = ASR_FEAT_DIM;
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

    const int blank_id    = ASR_BLANK_ID;
    const int state_size  = ASR_STATE_SIZE;
    const int durations[ASR_NUM_DURATIONS] = {0, 1, 2, 3, 4};

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

    // Decoder logits layout: [0..blank_id] token scores, then 5 duration
    // scores at [blank_id+1 .. blank_id+5]. Validated against the real tensor
    // size on the first iteration so the argmax loops below can't read OOB if
    // the model's vocabulary differs from blank_id.
    const int required_logits = blank_id + 1 + ASR_NUM_DURATIONS;
    int64_t logits_count = -1;

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

        if (logits_count < 0) {
            OrtTensorTypeAndShapeInfo* li = NULL;
            if (g_ort->GetTensorTypeAndShape(dec_outputs[0], &li) == NULL) {
                size_t cnt = 0;
                if (g_ort->GetTensorShapeElementCount(li, &cnt) == NULL)
                    logits_count = (int64_t)cnt;
                g_ort->ReleaseTensorTypeAndShapeInfo(li);
            }
            if (logits_count < required_logits) {
                LOGE("vaani: [DECODE] logits tensor too small (%lld < %d); aborting",
                     (long long)logits_count, required_logits);
                g_ort->ReleaseValue(dec_outputs[0]);
                if (dec_outputs[1]) g_ort->ReleaseValue(dec_outputs[1]);
                g_ort->ReleaseValue(t_frame);
                goto cleanup;
            }
        }

        float max_val = -INFINITY;
        int token_id = 0;
        for (int i = 0; i <= blank_id; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; token_id = i; }
        }

        float max_dur_val = -INFINITY;
        int dur_idx = 0;
        for (int i = 0; i < ASR_NUM_DURATIONS; i++) {
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
                if ((unsigned char)token[0] == SP_MARKER_BYTE0 &&
                    (unsigned char)token[1] == SP_MARKER_BYTE1 &&
                    (unsigned char)token[2] == SP_MARKER_BYTE2 &&
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
