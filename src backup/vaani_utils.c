#include "vaani_internal.h"
#include "stat_ort_plugin.h"

char* create_empty_string(void) {
    char* s = malloc(1);
    if (s) s[0] = '\0';
    return s;
}

FFI_EXPORT void vaani_string_free(char* str) {
    free(str);
}

float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}