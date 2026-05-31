#ifndef VAANI_INTERNAL_H
#define VAANI_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <onnxruntime_c_api.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

// Include stat_ort_plugin.h for LOGD/LOGE macros (must come after android/log.h)
#include "stat_ort_plugin.h"

// ── Pipeline-wide ─────────────────────────────────────────────────────────
#define SAMPLE_RATE          16000
#define MAX_SPEAKERS         10
#define EMBEDDING_DIM        256
#define SIMILARITY_THRESHOLD 0.45f

// ── VAD (Silero v5 at 16 kHz) ─────────────────────────────────────────────
//
// Silero VAD v5 at 16 kHz takes 512 audio samples per call AND requires the
// last 64 samples of the previous call to be prepended as "context". Total
// tensor length per inference is therefore 576, not 512. Without the
// context prepend the model still runs, but its STFT windows misalign,
// the conv filters lose waveform continuity, and probabilities cluster
// near zero on real speech.
// See snakers4/silero-vad utils_vad.py:OnnxWrapper.__call__.
//
// HYSTERESIS — we use two thresholds rather than one, because the cost of
// false onset is different from the cost of false EoU:
//
//   VAD_ONSET_THRESHOLD = 0.5  — high bar to OPEN a segment. Onset is
//     irreversible (we'll decode whatever comes next), so we want high
//     confidence that real speech has started.
//
//   VAD_SUSTAIN_THRESHOLD = 0.15 — low bar to STAY IN a segment. Once
//     we're decoding a sentence, mid-sentence pauses, soft consonants,
//     "uh", and breath sounds commonly produce VAD probs in 0.05-0.20.
//     Treating those as silence cuts utterances mid-thought. The low
//     sustain threshold means only real silence closes the segment.
//
// VAD_ONSET_FRAMES is how many consecutive ≥onset frames must pass before
// we commit to opening a segment. At 32 ms/frame, 2 frames = 64 ms.
#define VAD_AUDIO_SAMPLES      512
#define VAD_CONTEXT_SAMPLES    64
#define VAD_INPUT_SAMPLES      (VAD_CONTEXT_SAMPLES + VAD_AUDIO_SAMPLES)
#define VAD_ONSET_THRESHOLD    0.5f
#define VAD_SUSTAIN_THRESHOLD  0.15f
#define VAD_ONSET_FRAMES       2

// ── Streaming segmentation ────────────────────────────────────────────────
#define PREROLL_MILLISECONDS 500
#define PREROLL_MAX_SAMPLES  ((PREROLL_MILLISECONDS * SAMPLE_RATE) / 1000)


extern const OrtApi* g_ort;

// ── Math / feature structs ────────────────────────────────────────────────
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

// ── Core pipeline ─────────────────────────────────────────────────────────
struct VaaniPipeline {
    OrtEnv*        env;
    OrtSession*    encoder;
    OrtSession*    decoder;
    OrtSession*    vad;
    OrtSession*    speaker;
    OrtMemoryInfo* memory_info;
    OrtRunOptions* dec_run_opts;
    MelProcessor*  processor;          // ASR mel processor (128-dim)
    MelProcessor*  speaker_processor;  // Speaker embedding processor (80-dim)
    char**         vocab;
    int            vocab_size;
    // Silero VAD recurrent state and context buffer. Both are zeroed by
    // vaani_stream_init on each new stream session.
    float          vad_state[2 * 1 * 128];
    float          vad_context[VAD_CONTEXT_SAMPLES];
    bool           vad_state_initialized;
};

typedef struct {
    int   id;
    float embedding[EMBEDDING_DIM];
} SpeakerProfile;

struct VaaniStreamState {
    struct VaaniPipeline* pipeline;
    float*         chunk_buffer;
    int            chunk_len;
    int            chunk_cap;
    SpeakerProfile speakers[MAX_SPEAKERS];
    int            num_speakers;
    FILE*          wav_out_file;
    uint32_t       total_samples_written;
    uint64_t       global_sample_offset;
    uint64_t       segment_start_sample;
    float          preroll_buffer[PREROLL_MAX_SAMPLES];
    int            preroll_len;
    int            silence_samples;
    int            vad_onset_count;
};

// ── Model stubs ───────────────────────────────────────────────────────────
extern float run_silero_vad(struct VaaniPipeline* p, const float* audio, int len);
extern void  run_speaker_embedding(struct VaaniPipeline* p, const float* audio, int len, float* out_emb);
extern char** load_vocab(const char* path, int* out_size);
extern void  normalize_whitespace(char* str);

// ── Mel processor ─────────────────────────────────────────────────────────
extern MelProcessor* mel_processor_new(int sample_rate, int n_fft, int hop_length, int win_length, int n_mels);
extern float* mel_processor_extract(MelProcessor *mp, const float * restrict samples, int num_samples, int *out_num_frames);
extern void   mel_processor_free(MelProcessor *mp);
extern void   compute_fft(complex_t* buffer, int n_fft);

// ── ORT error-handling macros ─────────────────────────────────────────────
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

// ── Internal utilities ────────────────────────────────────────────────────
FFI_EXPORT char* create_empty_string(void);
extern float cosine_similarity(const float* a, const float* b, int dim);
extern char* decode_audio_buffer(struct VaaniPipeline* p, float* samples, int num_samples);

#endif // VAANI_INTERNAL_H