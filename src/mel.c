#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "vaani_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Removed compute_fft function and complex_t struct (now imported via header)

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// The MelProcessor struct is defined in vaani_internal.h (shared with the
// rest of the pipeline). This file only implements its operations.

MelProcessor* mel_processor_new(int sample_rate, int n_fft, int hop_length, int win_length, int n_mels) {
    // calloc so every pointer field (including the scratch buffers) starts
    // NULL — the cleanup path and mel_processor_free rely on that.
    MelProcessor *mp = (MelProcessor*)calloc(1, sizeof(MelProcessor));
    if (!mp) return NULL;

    mp->sample_rate = sample_rate;
    mp->n_fft = n_fft;
    mp->hop_length = hop_length;
    mp->win_length = win_length;
    mp->n_freqs = n_fft / 2 + 1;
    mp->n_mels = n_mels;

    float *mel_pts = NULL, *hz_pts = NULL, *bin_freqs = NULL;

    mp->hann_window = (float*)malloc((size_t)win_length * sizeof(float));
    if (!mp->hann_window) goto fail;
    for (int i = 0; i < win_length; i++) {
        mp->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i / (float)win_length));
    }

    mp->filterbank = (float*)calloc((size_t)n_mels * mp->n_freqs, sizeof(float));
    if (!mp->filterbank) goto fail;

    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel((float)sample_rate / 2.0f);

    mel_pts = (float*)malloc((size_t)(n_mels + 2) * sizeof(float));
    hz_pts  = (float*)malloc((size_t)(n_mels + 2) * sizeof(float));
    if (!mel_pts || !hz_pts) goto fail;
    for (int i = 0; i < n_mels + 2; i++) {
        mel_pts[i] = mel_min + (float)i * (mel_max - mel_min) / (float)(n_mels + 1);
        hz_pts[i] = mel_to_hz(mel_pts[i]);
    }

    bin_freqs = (float*)malloc((size_t)mp->n_freqs * sizeof(float));
    if (!bin_freqs) goto fail;
    for (int i = 0; i < mp->n_freqs; i++) {
        bin_freqs[i] = (float)i * (float)sample_rate / (float)n_fft;
    }

    for (int m = 0; m < n_mels; m++) {
        float l = hz_pts[m];
        float c = hz_pts[m + 1];
        float u = hz_pts[m + 2];
        float scale = 2.0f / (u - l);

        for (int f = 0; f < mp->n_freqs; f++) {
            float freq = bin_freqs[f];
            if (freq > l && freq < u) {
                float val = (freq <= c) ? (freq - l) / (c - l) : (u - freq) / (u - c);
                mp->filterbank[m * mp->n_freqs + f] = scale * val;
            }
        }
    }

    free(mel_pts);
    free(hz_pts);
    free(bin_freqs);

    return mp;

fail:
    LOGE("vaani: [MEL] mel_processor_new: allocation failed");
    free(mel_pts);
    free(hz_pts);
    free(bin_freqs);
    mel_processor_free(mp);
    return NULL;
}

void mel_processor_free(MelProcessor *mp) {
    if (!mp) return;
    free(mp->filterbank);
    free(mp->hann_window);
    free(mp);
}

float* mel_processor_extract(MelProcessor *mp, const float * restrict samples, int num_samples, int *out_num_frames) {
    *out_num_frames = 0;
    if (!mp || num_samples <= 0) {
        return NULL;
    }

    float * restrict padded     = NULL;
    float * restrict power_spec = NULL;
    complex_t * restrict fft_buffer = NULL;
    float * restrict mel        = NULL;
    float * restrict means      = NULL;
    float * restrict vars       = NULL;

    int pad = mp->win_length / 2;
    // size_t throughout so large inputs can't overflow an int mid-calculation.
    size_t padded_len = (size_t)num_samples + 2 * (size_t)pad;

    padded = (float*)malloc(padded_len * sizeof(float));
    if (!padded) {
        LOGE("vaani: [MEL] padded buffer alloc failed (%zu floats)", padded_len);
        goto fail;
    }

    memcpy(padded + pad, samples, (size_t)num_samples * sizeof(float));

    for (int i = 0; i < pad; i++) {
        int left_src = (i + 1) < (num_samples - 1) ? (i + 1) : (num_samples - 1);
        padded[pad - 1 - i] = samples[left_src];

        int right_src = num_samples - 2 - i;
        right_src = right_src > 0 ? right_src : 0;
        padded[num_samples + pad + i] = samples[right_src];
    }

    int num_frames = ((int)padded_len - mp->win_length) / mp->hop_length + 1;
    if (num_frames <= 0) {
        goto fail;
    }

    size_t power_count = (size_t)num_frames * (size_t)mp->n_freqs;
    power_spec = (float*)malloc(power_count * sizeof(float));
    fft_buffer = (complex_t*)malloc((size_t)mp->n_fft * sizeof(complex_t));
    if (!power_spec || !fft_buffer) {
        LOGE("vaani: [MEL] power_spec/fft buffer alloc failed");
        goto fail;
    }

    for (int i = 0; i < num_frames; i++) {
        int start = i * mp->hop_length;

        for (int j = 0; j < mp->n_fft; j++) {
            if (j < mp->win_length) {
                fft_buffer[j].re = padded[start + j] * mp->hann_window[j];
            } else {
                fft_buffer[j].re = 0.0f;
            }
            fft_buffer[j].im = 0.0f;
        }

        compute_fft(fft_buffer, mp->n_fft);

        for (int j = 0; j < mp->n_freqs; j++) {
            float re = fft_buffer[j].re;
            float im = fft_buffer[j].im;
            power_spec[i * mp->n_freqs + j] = re * re + im * im;
        }
    }

    // The mel matrix is the function's return value, owned and freed by the
    // caller, so it's a fresh allocation each call (which also keeps this
    // function reentrant — important since one MelProcessor is shared across
    // streams on a pipeline).
    mel = (float*)malloc((size_t)num_frames * (size_t)mp->n_mels * sizeof(float));
    if (!mel) {
        LOGE("vaani: [MEL] output buffer alloc failed");
        goto fail;
    }

    /* Replace this nested loop with cblas_sgemm from OpenBLAS/Accelerate if a BLAS library is available */
    for (int i = 0; i < num_frames; i++) {
        for (int m = 0; m < mp->n_mels; m++) {
            float sum = 0.0f;
            for (int f = 0; f < mp->n_freqs; f++) {
                sum += power_spec[i * mp->n_freqs + f] * mp->filterbank[m * mp->n_freqs + f];
            }
            mel[i * mp->n_mels + m] = logf(sum + 1e-10f);
        }
    }

    /* Changed normalization to compute sequentially across frames to prevent cache thrashing */
    means = (float*)calloc(mp->n_mels, sizeof(float));
    vars  = (float*)calloc(mp->n_mels, sizeof(float));
    if (!means || !vars) {
        LOGE("vaani: [MEL] mean/var buffer alloc failed");
        goto fail;
    }
    float inv_num_frames = 1.0f / (float)num_frames;

    for (int i = 0; i < num_frames; i++) {
        for (int m = 0; m < mp->n_mels; m++) {
            means[m] += mel[i * mp->n_mels + m];
        }
    }

    for (int m = 0; m < mp->n_mels; m++) {
        means[m] *= inv_num_frames; /* Multiplication is faster than division */
    }

    for (int i = 0; i < num_frames; i++) {
        for (int m = 0; m < mp->n_mels; m++) {
            float diff = mel[i * mp->n_mels + m] - means[m];
            vars[m] += diff * diff;
        }
    }

    float denom = (num_frames - 1.0f) > 1.0f ? (num_frames - 1.0f) : 1.0f;
    float inv_denom = 1.0f / denom;

    for (int m = 0; m < mp->n_mels; m++) {
        vars[m] = 1.0f / (sqrtf(vars[m] * inv_denom) + 1e-5f); /* Store scaling factor to avoid re-calculation */
    }

    for (int i = 0; i < num_frames; i++) {
        for (int m = 0; m < mp->n_mels; m++) {
            mel[i * mp->n_mels + m] = (mel[i * mp->n_mels + m] - means[m]) * vars[m];
        }
    }

    free(padded);
    free(power_spec);
    free(fft_buffer);
    free(means);
    free(vars);

    *out_num_frames = num_frames;
    return mel;

fail:
    free(padded);
    free(power_spec);
    free(fft_buffer);
    free(mel);
    free(means);
    free(vars);
    return NULL;
}