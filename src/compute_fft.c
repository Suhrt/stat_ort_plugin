#include <math.h>
#include "vaani_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Removed struct definition and 'static' keyword (now in header)
//
// Radix-2 Cooley-Tukey FFT, in place. n MUST be a power of two; the
// bit-reversal and butterfly stages produce garbage otherwise, so guard
// against it rather than silently corrupting the spectrum.
void compute_fft(complex_t* x, int n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        LOGE("vaani: [FFT] n=%d is not a positive power of two; skipping", n);
        return;
    }
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            complex_t temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        float wlen_re = cosf(angle);
        float wlen_im = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f;
            float w_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                complex_t u = x[i + j];
                complex_t v;
                v.re = x[i + j + len / 2].re * w_re - x[i + j + len / 2].im * w_im;
                v.im = x[i + j + len / 2].re * w_im + x[i + j + len / 2].im * w_re;
                
                x[i + j].re = u.re + v.re;
                x[i + j].im = u.im + v.im;
                x[i + j + len / 2].re = u.re - v.re;
                x[i + j + len / 2].im = u.im - v.im;
                
                float next_w_re = w_re * wlen_re - w_im * wlen_im;
                float next_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
    }
}