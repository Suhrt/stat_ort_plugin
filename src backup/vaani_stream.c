#include "vaani_internal.h"
#include "stat_ort_plugin.h"

// ── Segmentation tuning ────────────────────────────────────────────────────
//
// END_OF_UTTERANCE_SILENCE_SAMPLES: how much continuous silence closes a
//   segment. Set well above natural mid-sentence pause length (commas,
//   "uh"s typically 300–800 ms) so we cut only on real utterance
//   boundaries. 1500 ms works for general dictation. For domains with
//   long thinking pauses (medical prescription dictation, legal drafting)
//   consider bumping to 2000–2500 ms. The VAD sustain threshold (see
//   VAD_SUSTAIN_THRESHOLD in vaani_internal.h) controls what *counts* as
//   silence during this window — soft/breath/quiet-word frames stay
//   inside the segment thanks to the low sustain bar.
//
// MIN_SEGMENT_SAMPLES: segments shorter than this are discarded as likely
//   false starts (cough, mic bump, single short syllable).
#define END_OF_UTTERANCE_SILENCE_SAMPLES   24000          // 1500 ms at 16 kHz
#define MIN_SEGMENT_SAMPLES                SAMPLE_RATE    // 1 s

// Reset segment-tracking state after a segment closes or is discarded.
// Used by all the "the buffered audio is no longer part of an in-progress
// segment" paths so they behave identically.
static void reset_segment_state(VaaniStreamState* state) {
    state->chunk_len       = 0;
    state->preroll_len     = 0;
    state->vad_onset_count = 0;
    state->silence_samples = 0;
}

// Run speaker embedding + ASR decode on the currently-buffered audio,
// formatted timestamp + speaker id + transcript. Caller frees the returned
// malloc'd string. May return an empty string on decode failure.
static char* process_speech_segment(VaaniStreamState* state) {
    if (state->chunk_len == 0) return create_empty_string();

    LOGD("vaani: [SEGMENT] start chunk_len=%d (%.2fs) seg_start=%llu speakers=%d",
         state->chunk_len,
         (float)state->chunk_len / (float)SAMPLE_RATE,
         (unsigned long long)state->segment_start_sample,
         state->num_speakers);

    int best_speaker_id = 0;

    if (state->pipeline->speaker) {
        float current_embedding[EMBEDDING_DIM];

        // Speaker embedding works best on ~3 s of audio. For longer segments
        // take a centered 3 s window — that's typically the most stable part
        // of the utterance.
        const int SPEAKER_MAX_SAMPLES = SAMPLE_RATE * 3;
        int    spk_len   = state->chunk_len;
        float* spk_audio = state->chunk_buffer;
        if (spk_len > SPEAKER_MAX_SAMPLES) {
            int offset = (spk_len - SPEAKER_MAX_SAMPLES) / 2;
            spk_audio  = state->chunk_buffer + offset;
            spk_len    = SPEAKER_MAX_SAMPLES;
        }

        run_speaker_embedding(state->pipeline, spk_audio, spk_len, current_embedding);

        float best_score = -1.0f;
        int   best_idx   = -1;
        best_speaker_id  = -1;
        for (int i = 0; i < state->num_speakers; i++) {
            float score = cosine_similarity(current_embedding,
                                            state->speakers[i].embedding,
                                            EMBEDDING_DIM);
            if (score > best_score) {
                best_score      = score;
                best_idx        = i;
                best_speaker_id = state->speakers[i].id;
            }
        }

        if (best_score >= SIMILARITY_THRESHOLD && best_idx >= 0) {
            // Match: nudge the stored centroid slightly toward the new sample
            // to track gradual voice changes (volume, distance, mood).
            LOGD("vaani: [SPEAKER] matched id=%d score=%.4f",
                 best_speaker_id, best_score);
            const float alpha = 0.05f;
            float* stored = state->speakers[best_idx].embedding;
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                stored[i] = (1.0f - alpha) * stored[i] + alpha * current_embedding[i];
            }
        } else if (state->num_speakers < MAX_SPEAKERS) {
            // No good match and room for a new speaker: enroll.
            best_speaker_id = state->num_speakers;
            LOGD("vaani: [SPEAKER] new id=%d (best_score=%.4f)",
                 best_speaker_id, best_score);
            state->speakers[state->num_speakers].id = best_speaker_id;
            memcpy(state->speakers[state->num_speakers].embedding,
                   current_embedding, sizeof(float) * EMBEDDING_DIM);
            state->num_speakers++;
        } else {
            // Capacity full: assign to closest existing speaker.
            LOGD("vaani: [SPEAKER] MAX_SPEAKERS reached, assigning closest id=%d (score=%.3f)",
                 best_speaker_id, best_score);
        }
    }

    char* result_str = decode_audio_buffer(state->pipeline,
                                           state->chunk_buffer,
                                           state->chunk_len);
    if (!result_str) {
        LOGE("vaani: [DECODE] returned NULL");
        reset_segment_state(state);
        return create_empty_string();
    }

    int start_ms = (int)(((float)state->segment_start_sample / (float)SAMPLE_RATE) * 1000.0f + 0.5f);
    int end_ms   = (int)(((float)(state->segment_start_sample + state->chunk_len) / (float)SAMPLE_RATE) * 1000.0f + 0.5f);

    int s_min = start_ms / 60000, s_sec = (start_ms % 60000) / 1000, s_ms = start_ms % 1000;
    int e_min = end_ms   / 60000, e_sec = (end_ms   % 60000) / 1000, e_ms = end_ms   % 1000;

    size_t out_len = strlen(result_str) + 64;
    char* final_output = malloc(out_len);
    if (!final_output) {
        LOGE("vaani: [SEGMENT] malloc failed");
        free(result_str);
        reset_segment_state(state);
        return create_empty_string();
    }

    snprintf(final_output, out_len, "[%02d:%02d.%03d - %02d:%02d.%03d] [Speaker %d]: %s\n",
             s_min, s_sec, s_ms, e_min, e_sec, e_ms, best_speaker_id, result_str);
    free(result_str);

    LOGD("vaani: [SEGMENT] done: %s", final_output);

    reset_segment_state(state);
    return final_output;
}

FFI_EXPORT char* vaani_stream_push_chunk(VaaniStreamState* state,
                                         const int16_t* pcm_data,
                                         int num_samples) {
    if (!state || num_samples < 0) return NULL;

    if (num_samples > state->chunk_cap) {
        LOGE("vaani: [PUSH] num_samples=%d > chunk_cap=%d; dropping",
             num_samples, state->chunk_cap);
        state->global_sample_offset += num_samples;
        return NULL;
    }

    // Explicit flush from Dart side (zero-length chunk). Emits the
    // in-progress segment, if any.
    if (num_samples == 0) {
        LOGD("vaani: [PUSH] flush signal, chunk_len=%d", state->chunk_len);
        if (state->chunk_len > 0) {
            state->silence_samples = 0;
            return process_speech_segment(state);
        }
        return NULL;
    }

    if (!pcm_data) return NULL;

    // Persist raw PCM to the debug WAV before any further processing.
    if (state->wav_out_file) {
        fwrite(pcm_data, sizeof(int16_t), num_samples, state->wav_out_file);
        state->total_samples_written += num_samples;
    }

    char* result = NULL;

    // 30 s hard cap: if the incoming chunk would overflow the in-segment
    // buffer, force a segment close and start fresh with this chunk.
    if (state->chunk_len + num_samples > state->chunk_cap) {
        LOGD("vaani: [PUSH] chunk_cap overflow: chunk_len=%d + %d > cap=%d",
             state->chunk_len, num_samples, state->chunk_cap);
        if (state->chunk_len > MIN_SEGMENT_SAMPLES) {
            result = process_speech_segment(state);
        } else {
            reset_segment_state(state);
        }
        // segment_start_sample for the next segment is where THIS chunk
        // begins. global_sample_offset hasn't been advanced yet for the
        // current chunk, so it's exactly that timestamp.
        state->segment_start_sample = state->global_sample_offset;
    }

    // Append the new chunk to chunk_buffer as float32 in [-1, 1].
    float* write_ptr = state->chunk_buffer + state->chunk_len;
    for (int i = 0; i < num_samples; i++) {
        write_ptr[i] = pcm_data[i] * (1.0f / 32768.0f);
    }

    // VAD on the new chunk. Silero only accepts exactly 512 samples per
    // call. Three outcomes:
    //   1. VAD disabled: behave as if everything is speech, but don't
    //      drive vad_onset_count — onset would never become "confirmed".
    //   2. Wrong chunk size: same as (1). External callers must feed 512.
    //   3. VAD ran: vad_prob in [0,1] for a real opinion, -1.0 if the call
    //      itself failed (treat as UNKNOWN — don't trust the answer).
    float vad_prob = 1.0f;
    int   vad_valid = 0;
    if (!state->pipeline->vad) {
        // No VAD model loaded — leave defaults; chunks always count as
        // "stay in segment" but never as a confirmed onset.
    } else if (num_samples == VAD_AUDIO_SAMPLES) {
        vad_prob  = run_silero_vad(state->pipeline, write_ptr, num_samples);
        vad_valid = (vad_prob >= 0.0f);
        if (!vad_valid) {
            LOGE("vaani: [VAD] inference failed for this chunk; treating as UNKNOWN");
            vad_prob = 1.0f;
        }
    } else {
        LOGE("vaani: [PUSH] chunk size %d != %d (VAD frame size); skipping VAD",
             num_samples, VAD_AUDIO_SAMPLES);
    }

    if (vad_valid) {
        // Hysteresis: which threshold applies depends on the current state.
        //   - WAITING (chunk_len == 0): use the high onset threshold. Onset
        //     should require confident speech, not noise blips.
        //   - IN-SEG (chunk_len > 0): use the low sustain threshold. Once
        //     decoding a sentence, mid-utterance breaths and soft sounds
        //     should not be treated as silence and cut the segment.
        float threshold = (state->chunk_len == 0)
                          ? VAD_ONSET_THRESHOLD
                          : VAD_SUSTAIN_THRESHOLD;
        if (vad_prob >= threshold) {
            state->silence_samples = 0;
            state->vad_onset_count++;
        } else {
            state->silence_samples += num_samples;
            state->vad_onset_count  = 0;
        }
    }
    // else: leave both counters as-is. We have no reliable opinion this
    // frame, so neither open a segment nor close one.

    // ── State machine ────────────────────────────────────────────────
    // Two states: WAITING (no segment open) vs IN-SEG (chunk_len > 0).
    if (state->chunk_len == 0) {
        // WAITING — accumulate into the preroll ring buffer. Wait for
        // VAD_ONSET_FRAMES consecutive speech frames before opening
        // a segment, so VAD blips on noise don't open spurious segments.
        if (state->vad_onset_count < VAD_ONSET_FRAMES) {
            int keep  = num_samples > PREROLL_MAX_SAMPLES ? PREROLL_MAX_SAMPLES : num_samples;
            int shift = state->preroll_len + keep - PREROLL_MAX_SAMPLES;
            if (shift > 0) {
                memmove(state->preroll_buffer,
                        state->preroll_buffer + shift,
                        (state->preroll_len - shift) * sizeof(float));
                state->preroll_len -= shift;
            }
            memcpy(state->preroll_buffer + state->preroll_len,
                   write_ptr + (num_samples - keep),
                   keep * sizeof(float));
            state->preroll_len += keep;

            state->global_sample_offset += num_samples;
            return result;
        }

        // Onset confirmed. Open a segment by prepending the preroll buffer
        // so word-initial consonants aren't clipped by VAD detection lag.
        //
        // Defensive: with current constants preroll + chunk is far smaller
        // than chunk_cap (8000 + 512 << 480000) but if those constants ever
        // change, drop preroll rather than overflow.
        if (state->preroll_len + num_samples > state->chunk_cap) {
            LOGE("vaani: [ONSET] preroll(%d)+chunk(%d) > cap(%d); dropping preroll",
                 state->preroll_len, num_samples, state->chunk_cap);
            state->preroll_len = 0;
        }

        LOGD("vaani: [ONSET] confirmed (preroll=%d samples)", state->preroll_len);
        if (state->preroll_len > 0) {
            // Shift the just-written chunk up to make room for preroll,
            // then prepend preroll. memmove handles the (overlapping) shift.
            memmove(state->chunk_buffer + state->preroll_len, write_ptr,
                    num_samples * sizeof(float));
            memcpy(state->chunk_buffer, state->preroll_buffer,
                   state->preroll_len * sizeof(float));
            state->chunk_len            = state->preroll_len;
            state->segment_start_sample = (state->global_sample_offset >= (uint64_t)state->preroll_len)
                                          ? (state->global_sample_offset - state->preroll_len) : 0;
            state->preroll_len          = 0;
        } else {
            state->segment_start_sample = state->global_sample_offset;
        }

        // Reset silence counter — it was accumulating during WAITING and
        // could otherwise immediately trigger end-of-utterance on the
        // first silent frame inside the freshly-opened segment.
        state->silence_samples = 0;
    }

    // IN-SEG — grow the segment with the current chunk.
    state->chunk_len            += num_samples;
    state->global_sample_offset += num_samples;

    // Overflow above may have produced a segment to return. Don't also
    // check end-of-utterance — this chunk is the start of the next segment.
    if (result) return result;

    // End-of-utterance: enough continuous silence closes the segment,
    // provided the segment is long enough to be worth decoding.
    if (state->silence_samples >= END_OF_UTTERANCE_SILENCE_SAMPLES) {
        if (state->chunk_len > MIN_SEGMENT_SAMPLES) {
            LOGD("vaani: [TRIGGER] end-of-utterance silence (chunk_len=%d)",
                 state->chunk_len);
            result = process_speech_segment(state);
        } else {
            LOGD("vaani: [TRIGGER] segment too short (%d samples), discarding",
                 state->chunk_len);
            reset_segment_state(state);
            state->segment_start_sample = state->global_sample_offset;
        }
    }

    return result;
}

FFI_EXPORT VaaniStreamState* vaani_stream_init(VaaniPipeline* pipeline,
const char* out_wav_path) {
if (!pipeline) return NULL;

VaaniStreamState* s = calloc(1, sizeof(VaaniStreamState));
if (!s) return NULL;

s->pipeline    = pipeline;
s->chunk_cap   = SAMPLE_RATE * 30;
s->chunk_buffer = malloc(s->chunk_cap * sizeof(float));
if (!s->chunk_buffer) {
free(s);
return NULL;
}

// VAD recurrent state lives on the pipeline, not the stream. Reset it
// on every new stream session so probabilities at session start aren't
// conditioned on the previous session's tail audio. Same for the
// 64-sample Silero context buffer.
if (pipeline->vad) {
memset(pipeline->vad_state,   0, sizeof(pipeline->vad_state));
memset(pipeline->vad_context, 0, sizeof(pipeline->vad_context));
pipeline->vad_state_initialized = true;
}

LOGD("vaani: [INIT] chunk_cap=%d (%.0fs) preroll_max=%d "
"eou_silence=%d (%.2fs) min_segment=%d (%.2fs) "
"vad_onset=%.2f vad_sustain=%.2f onset_frames=%d",
s->chunk_cap, (float)s->chunk_cap / SAMPLE_RATE,
PREROLL_MAX_SAMPLES,
END_OF_UTTERANCE_SILENCE_SAMPLES,
(float)END_OF_UTTERANCE_SILENCE_SAMPLES / SAMPLE_RATE,
MIN_SEGMENT_SAMPLES,
(float)MIN_SEGMENT_SAMPLES / SAMPLE_RATE,
VAD_ONSET_THRESHOLD,
VAD_SUSTAIN_THRESHOLD,
VAD_ONSET_FRAMES);

if (out_wav_path) {
s->wav_out_file = fopen(out_wav_path, "wb");
if (s->wav_out_file) {
LOGD("vaani: [INIT] WAV debug: %s", out_wav_path);
// Leave room for the 44-byte WAV header; write data first, fill
// the header in close() once we know the final sample count.
fseek(s->wav_out_file, 44, SEEK_SET);
} else {
LOGE("vaani: [INIT] couldn't open WAV: %s", out_wav_path);
}
}
return s;
}

// Emit any in-progress segment as final text. Callers should invoke this
// once when the audio stream ends, BEFORE vaani_stream_close(), to retrieve
// the trailing utterance. Returns NULL if there is nothing to emit (no
// audio buffered, audio too short to be a real segment, or decode failure).
// Safe to call multiple times; subsequent calls return NULL.
FFI_EXPORT char* vaani_stream_flush(VaaniStreamState* state) {
    if (!state) return NULL;
    if (state->chunk_len == 0) return NULL;

    // Honour the MIN_SEGMENT_SAMPLES gate on flush too — a fraction of a
    // second of stray buffered audio isn't worth decoding.
    if (state->chunk_len <= MIN_SEGMENT_SAMPLES) {
        LOGD("vaani: [FLUSH] buffered audio too short (%d samples), discarding",
             state->chunk_len);
        reset_segment_state(state);
        return NULL;
    }

    LOGD("vaani: [FLUSH] emitting trailing segment (chunk_len=%d)",
         state->chunk_len);
    state->silence_samples = 0;
    char* trailing = process_speech_segment(state);
    // process_speech_segment returns "" on decode failure; collapse to NULL
    // so the caller can treat NULL as "nothing emitted".
    if (trailing && trailing[0] == '\0') {
        free(trailing);
        return NULL;
    }
    return trailing;
}

FFI_EXPORT void vaani_stream_close(VaaniStreamState* state) {
    if (!state) return;

    LOGD("vaani: [CLOSE] chunk_len=%d global_offset=%llu total_written=%u",
         state->chunk_len,
         (unsigned long long)state->global_sample_offset,
         state->total_samples_written);

    // Safety net: if a caller forgot to flush, we still try to emit the
    // trailing segment — but with no return path, we can only log it.
    // The proper fix is on the caller side (call vaani_stream_flush()).
    if (state->chunk_len > MIN_SEGMENT_SAMPLES) {
        char* trailing = process_speech_segment(state);
        if (trailing) {
            LOGE("vaani: [CLOSE] trailing segment dropped — caller should "
                 "have called vaani_stream_flush() before close. Lost text: %s",
                 trailing);
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

        uint32_t fmt_size        = 16;
        uint16_t audio_format    = 1;
        uint16_t num_channels    = 1;
        uint32_t sample_rate     = SAMPLE_RATE;
        uint32_t byte_rate       = SAMPLE_RATE * 2;
        uint16_t block_align     = 2;
        uint16_t bits_per_sample = 16;

        fwrite(&fmt_size,        4, 1, state->wav_out_file);
        fwrite(&audio_format,    2, 1, state->wav_out_file);
        fwrite(&num_channels,    2, 1, state->wav_out_file);
        fwrite(&sample_rate,     4, 1, state->wav_out_file);
        fwrite(&byte_rate,       4, 1, state->wav_out_file);
        fwrite(&block_align,     2, 1, state->wav_out_file);
        fwrite(&bits_per_sample, 2, 1, state->wav_out_file);
        fwrite("data",           1, 4, state->wav_out_file);
        fwrite(&data_size,       4, 1, state->wav_out_file);

        fclose(state->wav_out_file);
        LOGD("vaani: [CLOSE] WAV closed (%u samples)", state->total_samples_written);
    }

    free(state->chunk_buffer);
    free(state);
}