#ifndef STAT_ORT_PLUGIN_H
#define STAT_ORT_PLUGIN_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

// Opaque pointers for FFI
typedef struct VaaniPipeline VaaniPipeline;
typedef struct VaaniStreamState VaaniStreamState;

// String memory management
FFI_EXPORT void vaani_string_free(char* str);

// Pipeline Init / Free
FFI_EXPORT VaaniPipeline* vaani_pipeline_init(
        const char* enc_path, const char* dec_path, const char* vocab_path,
        const char* vad_path, const char* speaker_path, int encoder_threads);
FFI_EXPORT void vaani_pipeline_free(VaaniPipeline* p);

// File Processing
FFI_EXPORT char* vaani_pipeline_transcribe(VaaniPipeline* p, const char* wav_path);

// Streaming API
FFI_EXPORT VaaniStreamState* vaani_stream_init(VaaniPipeline* pipeline, const char* out_wav_path);
FFI_EXPORT char* vaani_stream_push_chunk(VaaniStreamState* state, const int16_t* pcm_data, int num_samples);
FFI_EXPORT void vaani_stream_close(VaaniStreamState* state);


#ifdef __cplusplus
}
#endif

#endif