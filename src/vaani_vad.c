#include "vaani_internal.h"
#include "stat_ort_plugin.h"

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
//
// vad_state (VAD_STATE_SIZE floats) and vad_context (VAD_CONTEXT_SAMPLES
// floats) are owned by the caller's stream and updated in place each call.
// They must be zeroed before the first frame of a session.
float run_silero_vad(struct VaaniPipeline* p, float* vad_state,
                     float* vad_context, const float* audio, int len) {
    if (!p->vad || !vad_state || !vad_context || !audio || len != VAD_AUDIO_SAMPLES) {
        LOGE("vaani: [VAD] invalid input (vad=%p, state=%p, ctx=%p, audio=%p, len=%d, expected=%d)",
             p->vad, (void*)vad_state, (void*)vad_context, audio, len, VAD_AUDIO_SAMPLES);
        return -1.0f;
    }

    // Build the 576-sample input by prepending the last 64 samples from
    // the previous call.
    float vad_input[VAD_INPUT_SAMPLES];
    memcpy(vad_input,                       vad_context, VAD_CONTEXT_SAMPLES * sizeof(float));
    memcpy(vad_input + VAD_CONTEXT_SAMPLES, audio,       VAD_AUDIO_SAMPLES   * sizeof(float));

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
            p->memory_info, vad_state, VAD_STATE_SIZE * sizeof(float),
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
            memcpy(vad_state, new_state, VAD_STATE_SIZE * sizeof(float));
        }
    } else if (status) {
        LOGE("vaani: [VAD] inference failed: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
    } else {
        LOGE("vaani: [VAD] output tensor NULL");
    }

    // Save the last VAD_CONTEXT_SAMPLES of this chunk for the next call.
    memcpy(vad_context,
           audio + (VAD_AUDIO_SAMPLES - VAD_CONTEXT_SAMPLES),
           VAD_CONTEXT_SAMPLES * sizeof(float));

    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(state_tensor);
    g_ort->ReleaseValue(sr_tensor);
    if (output_values[0]) g_ort->ReleaseValue(output_values[0]);
    if (output_values[1]) g_ort->ReleaseValue(output_values[1]);

    return probability;
}
