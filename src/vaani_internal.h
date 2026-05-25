#ifndef VAANI_INTERNAL_H
#define VAANI_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <onnxruntime_c_api.h>

// Include Android log only on Android
#ifdef __ANDROID__
#include <android/log.h>
#endif

// Include stat_ort_plugin.h for LOGD/LOGE macros (must come after android/log.h)
#include "stat_ort_plugin.h"

#define MAX_SPEAKERS 10
#define EMBEDDING_DIM 256
#define VAD_THRESHOLD 0.5f
#define SIMILARITY_THRESHOLD 0.5f
#define SAMPLE_RATE 16000
#define PREROLL_MILLISECONDS 500
#define PREROLL_MAX_SAMPLES ((PREROLL_MILLISECONDS * SAMPLE_RATE) / 1000)


extern const OrtApi* g_ort;

// ── 1. Math & Feature Structs ──
typedef struct {
    float re;
    float im;
} complex_t;

typedef struct {
    int sample_rate;
    int n_fft;
    int hop_length;
    int win_length;
    int n_mels;
    int n_freqs;
    float* hann_window;
    float* filterbank;
} MelProcessor;

// ── 2. Core Pipeline Structs ──
struct VaaniPipeline {
    OrtEnv* env;
    OrtSession* encoder;
    OrtSession* decoder;
    OrtSession* vad;
    OrtSession* speaker;
    OrtMemoryInfo* memory_info;
    OrtRunOptions* dec_run_opts;
    MelProcessor* processor;        // ASR mel processor (128-dim)
    MelProcessor* speaker_processor;  // Speaker embedding processor (80-dim)
    char** vocab;
    int vocab_size;
    float vad_state[2 * 1 * 128];
    bool  vad_state_initialized;
};

typedef struct {
    int id;
    float embedding[EMBEDDING_DIM];
} SpeakerProfile;

struct VaaniStreamState {
    struct VaaniPipeline* pipeline;
    float* chunk_buffer;
    int chunk_len;
    int chunk_cap;
    SpeakerProfile speakers[MAX_SPEAKERS];
    int num_speakers;
    FILE* wav_out_file;
    uint32_t total_samples_written;
    uint64_t global_sample_offset;
    uint64_t segment_start_sample;
    float preroll_buffer[PREROLL_MAX_SAMPLES];
    int   preroll_len;
    int   silence_samples;
};

// ── 3. External Model Stubs ──
extern float run_silero_vad(struct VaaniPipeline* p, const float* audio, int len);
extern void run_speaker_embedding(struct VaaniPipeline* p, const float* audio, int len, float* out_emb);
extern char** load_vocab(const char* path, int* out_size);
extern void normalize_whitespace(char* str);

// ── 4. Mel Processor External Functions ──
extern MelProcessor* mel_processor_new(int sample_rate, int n_fft, int hop_length, int win_length, int n_mels);
extern float* mel_processor_extract(MelProcessor *mp, const float * restrict samples, int num_samples, int *out_num_frames);
extern void mel_processor_free(MelProcessor *mp);
extern void compute_fft(complex_t* buffer, int n_fft);

// ── 5. Macros & Internal Functions ──
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

FFI_EXPORT char* create_empty_string(void);
extern float cosine_similarity(const float* a, const float* b, int dim);
extern char* decode_audio_buffer(struct VaaniPipeline* p, float* samples, int num_samples);

#endif // VAANI_INTERNAL_H