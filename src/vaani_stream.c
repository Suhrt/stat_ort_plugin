#include "vaani_internal.h"
#include "stat_ort_plugin.h"

static char* process_speech_segment(VaaniStreamState* state) {
    if (state->chunk_len == 0) return create_empty_string();

    int best_speaker_id = 0;

    if (state->pipeline->speaker) {
        float current_embedding[EMBEDDING_DIM];
        run_speaker_embedding(state->pipeline, state->chunk_buffer, state->chunk_len, current_embedding);

        float best_score = -1.0f;
        int best_idx = -1;
        best_speaker_id = -1;

        for (int i = 0; i < state->num_speakers; i++) {
            float score = cosine_similarity(current_embedding, state->speakers[i].embedding, EMBEDDING_DIM);
            if (score > best_score) {
                best_score = score;
                best_idx = i;
                best_speaker_id = state->speakers[i].id;
            }
        }

        if (best_score >= SIMILARITY_THRESHOLD && best_idx >= 0) {
            float alpha = 0.05f;
            float* stored = state->speakers[best_idx].embedding;
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                stored[i] = (1.0f - alpha) * stored[i] + alpha * current_embedding[i];
            }
        } else if (best_score < SIMILARITY_THRESHOLD) {
            if (state->num_speakers < MAX_SPEAKERS) {
                best_speaker_id = state->num_speakers;
                state->speakers[state->num_speakers].id = best_speaker_id;
                memcpy(state->speakers[state->num_speakers].embedding, current_embedding, sizeof(float) * EMBEDDING_DIM);
                state->num_speakers++;
            } else {
                LOGE("vaani: MAX_SPEAKERS reached, assigning to closest speaker %d (score=%.3f)", best_speaker_id, best_score);
            }
        }
    }

    char* result_str = decode_audio_buffer(state->pipeline, state->chunk_buffer, state->chunk_len);
    if (!result_str) {
        state->chunk_len = 0;
        // NEW: Reset preroll after segment completion to avoid leaking old audio
        state->preroll_len = 0;
        return create_empty_string();
    }

    int start_total_ms = (int)(((float)state->segment_start_sample / (float)SAMPLE_RATE) * 1000.0f + 0.5f);
    int end_total_ms   = (int)(((float)(state->segment_start_sample + state->chunk_len) / (float)SAMPLE_RATE) * 1000.0f + 0.5f);

    int s_min = start_total_ms / 60000;
    int s_sec = (start_total_ms % 60000) / 1000;
    int s_ms  = start_total_ms % 1000;

    int e_min = end_total_ms / 60000;
    int e_sec = (end_total_ms % 60000) / 1000;
    int e_ms  = end_total_ms % 1000;

    size_t out_len = strlen(result_str) + 64;
    char* final_output = malloc(out_len);
    if (!final_output) {
        free(result_str);
        state->chunk_len = 0;
        // NEW: Reset preroll
        state->preroll_len = 0;
        return create_empty_string();
    }

    snprintf(final_output, out_len, "[%02d:%02d.%03d - %02d:%02d.%03d] [Speaker %d]: %s\n",
             s_min, s_sec, s_ms,
             e_min, e_sec, e_ms,
             best_speaker_id, result_str);

    free(result_str);
    state->chunk_len = 0;
    // NEW: Reset preroll
    state->preroll_len = 0;
    return final_output;
}


FFI_EXPORT char* vaani_stream_push_chunk(VaaniStreamState* state, const int16_t* pcm_data, int num_samples) {
    if (!state || num_samples < 0) return NULL;

    if (num_samples > state->chunk_cap) {
        LOGE("vaani_stream_push_chunk: num_samples (%d) exceeds chunk_cap (%d); dropping chunk.",
             num_samples, state->chunk_cap);
        state->global_sample_offset += num_samples;
        return NULL;
    }

    if (num_samples == 0) {
        if (state->chunk_len > 0) {
            state->silence_samples = 0;
            return process_speech_segment(state);
        }
        return NULL;
    }

    if (!pcm_data) return NULL;

    if (state->wav_out_file) {
        fwrite(pcm_data, sizeof(int16_t), num_samples, state->wav_out_file);
        state->total_samples_written += num_samples;
    }

    char* result = NULL;

    if (state->chunk_len + num_samples > state->chunk_cap) {
        if (state->chunk_len > SAMPLE_RATE) {
            result = process_speech_segment(state);
        } else {
            state->chunk_len = 0;
            state->preroll_len = 0;
        }
        state->silence_samples = 0;

        if (state->chunk_len == 0) {
            state->segment_start_sample = state->global_sample_offset;
        }

        if (state->chunk_len + num_samples > state->chunk_cap) {
            state->global_sample_offset += num_samples;
            return result;
        }
    }

    float* write_ptr = state->chunk_buffer + state->chunk_len;
    for (int i = 0; i < num_samples; i++) {
        write_ptr[i] = pcm_data[i] * (1.0f / 32768.0f);
    }

    float vad_prob = 1.0f;
    if (state->pipeline->vad) {
        vad_prob = run_silero_vad(state->pipeline, write_ptr, num_samples);
    }

    if (vad_prob >= VAD_THRESHOLD) {
        state->silence_samples = 0;
    } else {
        state->silence_samples += num_samples;
    }

    if (state->chunk_len == 0) {
        if (vad_prob < VAD_THRESHOLD) {
            // CHANGED: Replaced hardcoded 4800 with PREROLL_MAX_SAMPLES macro
            int keep = num_samples > PREROLL_MAX_SAMPLES ? PREROLL_MAX_SAMPLES : num_samples;
            int shift = state->preroll_len + keep - PREROLL_MAX_SAMPLES;

            if (shift > 0) {
                memmove(state->preroll_buffer, state->preroll_buffer + shift, (state->preroll_len - shift) * sizeof(float));
                state->preroll_len -= shift;
            }
            memcpy(state->preroll_buffer + state->preroll_len, write_ptr + (num_samples - keep), keep * sizeof(float));
            state->preroll_len += keep;

            state->global_sample_offset += num_samples;
            return result;
        } else {
            if (state->preroll_len > 0) {
                memmove(state->chunk_buffer + state->preroll_len, write_ptr, num_samples * sizeof(float));
                memcpy(state->chunk_buffer, state->preroll_buffer, state->preroll_len * sizeof(float));

                state->chunk_len = state->preroll_len;
                state->segment_start_sample = (state->global_sample_offset >= (uint64_t)state->preroll_len)
                                              ? (state->global_sample_offset - state->preroll_len) : 0;
                state->preroll_len = 0;
            } else {
                state->segment_start_sample = state->global_sample_offset;
            }
        }
    }

    state->chunk_len += num_samples;
    state->global_sample_offset += num_samples;

    if (!result) {
        if (state->chunk_len >= state->chunk_cap) {
            result = process_speech_segment(state);
            state->silence_samples = 0;
        }
        else if (state->silence_samples >= 6400) {
            // CHANGED: Replaced 4800 with PREROLL_MAX_SAMPLES
            if (state->chunk_len > (PREROLL_MAX_SAMPLES + 2400)) {
                result = process_speech_segment(state);
            } else {
                state->chunk_len = 0;
                state->preroll_len = 0;
                state->segment_start_sample = state->global_sample_offset;
            }
            state->silence_samples = 0;
        }
    }

    return result;
}

FFI_EXPORT VaaniStreamState* vaani_stream_init(VaaniPipeline* pipeline, const char* out_wav_path) {
    if (!pipeline) return NULL;

    VaaniStreamState* s = calloc(1, sizeof(VaaniStreamState));
    if (!s) return NULL;

    s->pipeline = pipeline;
    s->chunk_cap = SAMPLE_RATE * 30;
    s->chunk_buffer = malloc(s->chunk_cap * sizeof(float));
    if (!s->chunk_buffer) {
        free(s);
        return NULL;
    }

    if (out_wav_path) {
        s->wav_out_file = fopen(out_wav_path, "wb");
        if (s->wav_out_file) {
            fseek(s->wav_out_file, 44, SEEK_SET);
        }
    }
    return s;
}

// CORRECTION: Re-added missing cleanup function
FFI_EXPORT void vaani_stream_close(VaaniStreamState* state) {
    if (!state) return;

    if (state->chunk_len > SAMPLE_RATE) {
        char* trailing = process_speech_segment(state);
        if (trailing) {
            LOGE("Trailing audio processed but discarded - %zu chars", strlen(trailing));
            free(trailing);
        }
    }

    if (state->wav_out_file) {
        uint32_t file_size = state->total_samples_written * 2 + 36;
        uint32_t data_size = state->total_samples_written * 2;

        fseek(state->wav_out_file, 0, SEEK_SET);
        fwrite("RIFF", 1, 4, state->wav_out_file);
        fwrite(&file_size, 4, 1, state->wav_out_file);
        fwrite("WAVE", 1, 4, state->wav_out_file);
        fwrite("fmt ", 1, 4, state->wav_out_file);

        uint32_t fmt_size = 16;
        uint16_t audio_format = 1;
        uint16_t num_channels = 1;
        uint32_t sample_rate = SAMPLE_RATE;
        uint32_t byte_rate = SAMPLE_RATE * 2;
        uint16_t block_align = 2;
        uint16_t bits_per_sample = 16;

        fwrite(&fmt_size, 4, 1, state->wav_out_file);
        fwrite(&audio_format, 2, 1, state->wav_out_file);
        fwrite(&num_channels, 2, 1, state->wav_out_file);
        fwrite(&sample_rate, 4, 1, state->wav_out_file);
        fwrite(&byte_rate, 4, 1, state->wav_out_file);
        fwrite(&block_align, 2, 1, state->wav_out_file);
        fwrite(&bits_per_sample, 2, 1, state->wav_out_file);
        fwrite("data", 1, 4, state->wav_out_file);
        fwrite(&data_size, 4, 1, state->wav_out_file);

        fclose(state->wav_out_file);
    }

    free(state->chunk_buffer);
    free(state);
}