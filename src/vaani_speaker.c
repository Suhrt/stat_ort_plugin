#include "vaani_internal.h"
#include "stat_ort_plugin.h"
#include <math.h>

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
    LOGD("vaani: [SPEAKER] Fbanks extracted: seq_len=%d, input shape [1, %d, %d]",
         seq_len, seq_len, SPEAKER_FEAT_DIM);

    int64_t input_shape[] = {1, seq_len, SPEAKER_FEAT_DIM};
    OrtValue* input_tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
            p->memory_info, (void*)fbanks, (size_t)seq_len * SPEAKER_FEAT_DIM * sizeof(float),
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