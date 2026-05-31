#ifndef STAT_ORT_PLUGIN_H
#define STAT_ORT_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Log macros ─────────────────────────────────────────────────────────
// LOGD is verbose debug/progress logging; LOGE is error-level (real failures
// only). LOGD is compiled out unless VAANI_DEBUG is defined, so release builds
// carry no per-frame/per-segment logging overhead. LOGE is always active.
// Enable verbose logging by building with -DVAANI_DEBUG (CMake: -DVAANI_DEBUG=ON).
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "VaaniCPlugin"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#ifdef VAANI_DEBUG
#define LOGD(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...) ((void)0)
#endif
#else
#include <stdio.h>
#define LOGE(...) fprintf(stderr, "VAANI_ERROR: " __VA_ARGS__)
#ifdef VAANI_DEBUG
#define LOGD(...) printf("VAANI_DEBUG: " __VA_ARGS__)
#else
#define LOGD(...) ((void)0)
#endif
#endif

// ── FFI export decoration ─────────────────────────────────────────────
#if _WIN32
#define FFI_EXPORT __declspec(dllexport)
#else
#define FFI_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

// ── Opaque types ───────────────────────────────────────────────────────
typedef struct VaaniPipeline    VaaniPipeline;
typedef struct VaaniStreamState VaaniStreamState;

// ── String memory management ──────────────────────────────────────────
// Every char* returned by a vaani_* function must be freed via this.
FFI_EXPORT void vaani_string_free(char* str);

// ── Pipeline lifecycle ────────────────────────────────────────────────
// vad_path and speaker_path may be NULL to disable VAD or diarization.
FFI_EXPORT VaaniPipeline* vaani_pipeline_init(
        const char* enc_path,
        const char* dec_path,
        const char* vocab_path,
        const char* vad_path,
        const char* speaker_path,
        int         encoder_threads);

FFI_EXPORT void vaani_pipeline_free(VaaniPipeline* p);

// ── File transcription ────────────────────────────────────────────────
// Transcribe a 16 kHz mono int16 WAV file. Returns full transcript.
FFI_EXPORT char* vaani_pipeline_transcribe(VaaniPipeline* p, const char* wav_path);

// ── Streaming API ─────────────────────────────────────────────────────
// Lifecycle:
//   1. vaani_stream_init(pipeline, opt_debug_wav_path) -> state
//   2. vaani_stream_push_chunk(state, pcm, 512) repeatedly (one call per
//      512-sample frame). Returns NULL until a segment closes; then
//      returns the formatted transcript line (caller frees).
//   3. vaani_stream_flush(state) -> trailing transcript or NULL.
//      MUST be called before close() to retrieve the final segment.
//   4. vaani_stream_close(state) — tears down.
//
// Calling close() without first calling flush() drops any in-progress
// segment.

FFI_EXPORT VaaniStreamState* vaani_stream_init(
        VaaniPipeline* pipeline,
        const char*    out_wav_path);

// Push exactly 512 samples of int16 PCM (one Silero VAD frame at 16 kHz).
// Returns:
//   * NULL if no segment closed during this push
//   * malloc'd string with one or more "[mm:ss.SSS - mm:ss.SSS] [Speaker
//     N]: ..." lines (newline-terminated) if a segment closed
FFI_EXPORT char* vaani_stream_push_chunk(
        VaaniStreamState* state,
        const int16_t*    pcm_data,
        int               num_samples);

// Emit any in-progress segment as final text. Returns NULL if there is
// nothing buffered, the buffer is too short to be a real segment, or
// decoding failed. Safe to call multiple times — subsequent calls
// return NULL.
FFI_EXPORT char* vaani_stream_flush(VaaniStreamState* state);

// Tear down the stream. Call vaani_stream_flush() first if you want the
// trailing segment.
FFI_EXPORT void vaani_stream_close(VaaniStreamState* state);

#ifdef __cplusplus
}
#endif

#endif // STAT_ORT_PLUGIN_H