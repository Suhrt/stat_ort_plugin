#ifndef STAT_ORT_PLUGIN_H
#define STAT_ORT_PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

// Logging — defined here so all .c files get it before any macro that uses it
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "VaaniCPlugin"
#define LOGD(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...) printf("VAANI_DEBUG: " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "VAANI_ERROR: " __VA_ARGS__)
#endif


#if _WIN32
#define FFI_EXPORT __declspec(dllexport)
#else
#define FFI_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

// --- Public FFI API ---
typedef struct MelProcessor MelProcessor;
typedef struct VaaniPipeline VaaniPipeline;

FFI_EXPORT MelProcessor* mel_processor_new(int sample_rate, int n_fft, int hop_length, int win_length, int n_mels);
FFI_EXPORT void mel_processor_free(MelProcessor *mp);
FFI_EXPORT float* mel_processor_extract(MelProcessor *mp, const float *samples, int num_samples, int *out_num_frames);
FFI_EXPORT void mel_data_free(float *data);

FFI_EXPORT VaaniPipeline* vaani_pipeline_init(const char* enc_path, const char* dec_path, const char* vocab_path, int encoder_threads, int decoder_threads);
FFI_EXPORT void vaani_pipeline_free(VaaniPipeline* p);
FFI_EXPORT char* vaani_pipeline_transcribe(VaaniPipeline* p, const char* wav_path);
FFI_EXPORT void vaani_string_free(char* str);

// --- Internal C API (Ignored by Dart/FFI) ---
typedef struct {
    float re;
    float im;
} complex_t;

void compute_fft(complex_t* x, int n);
void normalize_whitespace(char* str);

#ifdef __cplusplus
}
#endif

#endif