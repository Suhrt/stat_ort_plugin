#include "vaani_internal.h"
#include "stat_ort_plugin.h"
#include <math.h>
#include <stdbool.h>

#ifndef _WIN32
#include <pthread.h>
#endif

const OrtApi* g_ort = NULL;

// Resolve the ORT API table exactly once, even if vaani_pipeline_init is
// called concurrently from multiple threads. POSIX platforms (Android, iOS,
// macOS, Linux) use pthread_once; the _WIN32 fallback is best-effort since
// Windows is not a supported target.
static void init_ort_api(void) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

static void ensure_ort_api(void) {
#ifndef _WIN32
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init_ort_api);
#else
    if (!g_ort) init_ort_api();
#endif
}

// Create an ORT session for `path` with the standard option set (graph
// optimisation, thread counts, memory-pattern toggle). On success writes the
// session to *out and returns NULL; on failure returns the OrtStatus, which
// the caller logs and releases.
static OrtStatus* create_ort_session(VaaniPipeline* p, const char* path,
                                     int intra_threads, int inter_threads,
                                     bool enable_mem_pattern, OrtSession** out) {
    OrtSessionOptions* opts = NULL;
    OrtStatus* s = g_ort->CreateSessionOptions(&opts);
    if (s) return s;
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    g_ort->SetIntraOpNumThreads(opts, intra_threads);
    g_ort->SetInterOpNumThreads(opts, inter_threads);
    if (enable_mem_pattern) g_ort->EnableMemPattern(opts);
    else                    g_ort->DisableMemPattern(opts);
    s = g_ort->CreateSession(p->env, path, opts, out);
    g_ort->ReleaseSessionOptions(opts);
    return s;
}

// ORT error helpers that free the half-built pipeline before bailing. Used
// only inside vaani_pipeline_init, where a `fail:` label and `p` are in scope.
#define ORT_GOTO_FAIL_ON_ERR(expr) \
    do { \
        OrtStatus* _s = (expr); \
        if (_s != NULL) { \
            LOGE("ORT error: %s", g_ort->GetErrorMessage(_s)); \
            g_ort->ReleaseStatus(_s); \
            goto fail; \
        } \
    } while (0)

#define ORT_SESSION_OR_FAIL(name, expr) \
    do { \
        OrtStatus* _s = (expr); \
        if (_s != NULL) { \
            LOGE("vaani_pipeline_init: Failed to create " name " session: %s", \
                 g_ort->GetErrorMessage(_s)); \
            g_ort->ReleaseStatus(_s); \
            goto fail; \
        } \
    } while (0)

FFI_EXPORT VaaniPipeline* vaani_pipeline_init(
        const char* enc_path, const char* dec_path, const char* vocab_path,
        const char* vad_path, const char* speaker_path, int encoder_threads
) {
    ensure_ort_api();
    if (!g_ort) return NULL;

    VaaniPipeline* p = calloc(1, sizeof(VaaniPipeline));
    if (!p) return NULL;

    p->processor = mel_processor_new(SAMPLE_RATE, MEL_N_FFT, MEL_HOP_LENGTH, MEL_WIN_LENGTH, ASR_FEAT_DIM);
    if (!p->processor) {
        LOGE("vaani_pipeline_init: Failed to create ASR mel processor");
        vaani_pipeline_free(p);
        return NULL;
    }

    if (speaker_path) {
        p->speaker_processor = mel_processor_new(SAMPLE_RATE, MEL_N_FFT, MEL_HOP_LENGTH, MEL_WIN_LENGTH, SPEAKER_FEAT_DIM);
        if (!p->speaker_processor) {
            LOGE("vaani_pipeline_init: Failed to create speaker mel processor");
            vaani_pipeline_free(p);
            return NULL;
        }
    }

    ORT_GOTO_FAIL_ON_ERR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vaani", &p->env));
    ORT_GOTO_FAIL_ON_ERR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &p->memory_info));

    // Encoder runs multi-threaded; mem pattern disabled (variable-length
    // input). Decoder/VAD/speaker run single-threaded with mem pattern on.
    ORT_SESSION_OR_FAIL("encoder",
        create_ort_session(p, enc_path, encoder_threads, encoder_threads, false, &p->encoder));
    LOGD("vaani: [INIT] Encoder session created: %s", enc_path);

    ORT_SESSION_OR_FAIL("decoder",
        create_ort_session(p, dec_path, 1, 1, true, &p->decoder));
    LOGD("vaani: [INIT] Decoder session created: %s", dec_path);

    if (vad_path) {
        ORT_SESSION_OR_FAIL("VAD",
            create_ort_session(p, vad_path, 1, 1, true, &p->vad));
        LOGD("vaani: [INIT] VAD session created: %s", vad_path);
    } else {
        LOGD("vaani: [INIT] No VAD path provided — VAD disabled");
    }

    if (speaker_path) {
        ORT_SESSION_OR_FAIL("speaker",
            create_ort_session(p, speaker_path, 1, 1, true, &p->speaker));
        LOGD("vaani: [INIT] Speaker session created: %s", speaker_path);
    } else {
        LOGD("vaani: [INIT] No speaker path provided — diarization disabled");
    }

    ORT_GOTO_FAIL_ON_ERR(g_ort->CreateRunOptions(&p->dec_run_opts));
    ORT_GOTO_FAIL_ON_ERR(g_ort->RunOptionsSetRunLogSeverityLevel(p->dec_run_opts, 3));

    p->vocab = load_vocab(vocab_path, &p->vocab_size);
    if (!p->vocab) {
        LOGE("vaani_pipeline_init: Failed to load vocab from %s", vocab_path);
        goto fail;
    }

    LOGD("vaani: [INIT] Pipeline ready. encoder_threads=%d vocab_size=%d vad=%s speaker=%s",
         encoder_threads, p->vocab_size,
         p->vad ? "YES" : "NO",
         p->speaker ? "YES" : "NO");
    return p;

fail:
    vaani_pipeline_free(p);
    return NULL;
}

#undef ORT_GOTO_FAIL_ON_ERR
#undef ORT_SESSION_OR_FAIL

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

