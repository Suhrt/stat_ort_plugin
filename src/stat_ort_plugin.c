#include "vaani_internal.h"
#include "stat_ort_plugin.h"

FFI_EXPORT char* vaani_pipeline_transcribe(VaaniPipeline* p, const char* wav_path) {
    LOGD("vaani_pipeline_transcribe: Starting for %s", wav_path);

    if (!p || !wav_path) {
        LOGE("vaani_pipeline_transcribe: Null pipeline or wav_path");
        return create_empty_string();
    }

    FILE* file = fopen(wav_path, "rb");
    if (!file) {
        LOGE("vaani_pipeline_transcribe: Failed to open %s", wav_path);
        return create_empty_string();
    }

    char riff_header[4];
    char wave_header[4];
    if (fread(riff_header, 1, 4, file) != 4 || memcmp(riff_header, "RIFF", 4) != 0) {
        LOGE("vaani_pipeline_transcribe: Missing RIFF header");
        fclose(file);
        return create_empty_string();
    }
    fseek(file, 4, SEEK_CUR);
    if (fread(wave_header, 1, 4, file) != 4 || memcmp(wave_header, "WAVE", 4) != 0) {
        LOGE("vaani_pipeline_transcribe: Missing WAVE header");
        fclose(file);
        return create_empty_string();
    }

    unsigned char chunk_header[8];
    int data_size      = 0;
    int audio_format   = 0;
    int channels       = 0;
    int sample_rate    = 0;
    int bits_per_sample = 0;

    fseek(file, 12, SEEK_SET);
    while (fread(chunk_header, 1, 8, file) == 8) {
        // WAV is little-endian. Assemble multi-byte fields from individual
        // bytes (rather than casting/reading raw) so parsing is correct
        // regardless of host byte order.
        int chunk_size = (int)(chunk_header[4]         |
                               (chunk_header[5] <<  8) |
                               (chunk_header[6] << 16) |
                               (chunk_header[7] << 24));

        LOGD("WAV chunk: '%.4s', size: %d bytes", chunk_header, chunk_size);

        if (memcmp(chunk_header, "fmt ", 4) == 0) {
            unsigned char fmt_data[16];
            if (fread(fmt_data, 1, 16, file) != 16) {
                LOGE("vaani_pipeline_transcribe: Failed to read fmt chunk");
                fclose(file);
                return create_empty_string();
            }
            audio_format    = fmt_data[0]  | (fmt_data[1]  << 8);
            channels        = fmt_data[2]  | (fmt_data[3]  << 8);
            sample_rate     = fmt_data[4]  | (fmt_data[5]  << 8) | (fmt_data[6]  << 16) | (fmt_data[7]  << 24);
            bits_per_sample = fmt_data[14] | (fmt_data[15] << 8);

            LOGD("WAV format -> Format: %d (1=PCM, 3=Float), Channels: %d, SampleRate: %d, Bits: %d",
                 audio_format, channels, sample_rate, bits_per_sample);

            if (audio_format != 1) {
                LOGE("vaani_pipeline_transcribe: Unsupported audio format %d (only PCM int16 supported)", audio_format);
                fclose(file);
                return create_empty_string();
            }
            if (channels != 1) {
                LOGE("vaani_pipeline_transcribe: Unsupported channel count %d (only mono supported)", channels);
                fclose(file);
                return create_empty_string();
            }
            if (sample_rate != 16000) {
                LOGE("vaani_pipeline_transcribe: Unsupported sample rate %d Hz (expected 16000)", sample_rate);
                fclose(file);
                return create_empty_string();
            }
            if (bits_per_sample != 16) {
                LOGE("vaani_pipeline_transcribe: Unsupported bit depth %d (expected 16)", bits_per_sample);
                fclose(file);
                return create_empty_string();
            }

            if (chunk_size > 16) fseek(file, chunk_size - 16, SEEK_CUR);
        }
        else if (memcmp(chunk_header, "data", 4) == 0) {
            data_size = chunk_size;
            break;
        }
        else {
            LOGD("Skipping unknown WAV chunk '%.4s'", chunk_header);
            fseek(file, chunk_size, SEEK_CUR);
        }
    }

    if (audio_format == 0) {
        LOGE("vaani_pipeline_transcribe: fmt chunk not found before data chunk");
        fclose(file);
        return create_empty_string();
    }
    if (data_size <= 0) {
        LOGE("vaani_pipeline_transcribe: data chunk not found or empty");
        fclose(file);
        return create_empty_string();
    }

    int total_samples = data_size / sizeof(int16_t);
    LOGD("Audio data: %d bytes, %d samples, %.2f seconds",
         data_size, total_samples, (float)total_samples / 16000.0f);

    if (total_samples < 512) {
        LOGE("vaani_pipeline_transcribe: Audio too short (%d samples)", total_samples);
        fclose(file);
        return create_empty_string();
    }

    VaaniStreamState* state = vaani_stream_init(p, NULL);
    if (!state) {
        LOGE("vaani_pipeline_transcribe: vaani_stream_init failed");
        fclose(file);
        return create_empty_string();
    }

    size_t full_cap = 4096;
    char* full_transcript = malloc(full_cap);
    if (!full_transcript) {
        LOGE("vaani_pipeline_transcribe: Failed to allocate transcript buffer");
        vaani_stream_close(state);
        fclose(file);
        return create_empty_string();
    }
    full_transcript[0] = '\0';
    size_t full_len = 0;

    int16_t buffer[512];
    int total_data_bytes = 0;
    int chunk_counter    = 0;
    int segments_found   = 0;
    int max_amp_global   = 0;

    while (total_data_bytes < data_size) {
        int remaining_samples = (data_size - total_data_bytes) / (int)sizeof(int16_t);
        if (remaining_samples <= 0) break;

        size_t samples_to_read = (remaining_samples < 512) ? remaining_samples : 512;
        size_t samples_read = fread(buffer, sizeof(int16_t), samples_to_read, file);
        if (samples_read == 0) break;

        // Pad final undersized chunk to exactly 512 so Silero VAD gets a valid input
        if (samples_read < 512) {
            LOGD("vaani_pipeline_transcribe: Padding final chunk from %zu to 512 samples", samples_read);
            memset(buffer + samples_read, 0, (512 - samples_read) * sizeof(int16_t));
            samples_read = 512;
        }

        // Amplitude tracking
        int max_amp = 0;
        for (size_t i = 0; i < samples_read; i++) {
            int abs_val = buffer[i] < 0 ? -buffer[i] : buffer[i];
            if (abs_val > max_amp) max_amp = abs_val;
        }
        if (max_amp > max_amp_global) max_amp_global = max_amp;

        if (chunk_counter == 0) {
            LOGD("First chunk max amplitude: %d (expected >500 for real speech, <100 suggests corrupt/silent file)", max_amp);
            if (max_amp < 100) {
                LOGE("vaani_pipeline_transcribe: Suspiciously low amplitude on first chunk (%d). "
                     "File may be silent, corrupt, wrong format, or wrong bit depth.", max_amp);
            }
        }

        total_data_bytes += (int)(samples_read * sizeof(int16_t));
        chunk_counter++;

        char* segment_text = vaani_stream_push_chunk(state, buffer, (int)samples_read);
        if (segment_text) {
            size_t seg_len = strlen(segment_text);
            if (seg_len > 0) {
                segments_found++;
                LOGD("Chunk %d segment %d: '%s'", chunk_counter, segments_found, segment_text);

                if (full_len + seg_len + 1 > full_cap) {
                    full_cap = (full_len + seg_len + 1) * 2;
                    char* tmp = realloc(full_transcript, full_cap);
                    if (!tmp) {
                        LOGE("vaani_pipeline_transcribe: realloc failed at segment %d", segments_found);
                        free(segment_text);
                        break;
                    }
                    full_transcript = tmp;
                }
                memcpy(full_transcript + full_len, segment_text, seg_len + 1);
                full_len += seg_len;
            }
            free(segment_text);
        }
    }

    LOGD("File read complete: %d chunks, %d segments, peak amplitude: %d, bytes read: %d",
         chunk_counter, segments_found, max_amp_global, total_data_bytes);

    if (max_amp_global < 100) {
        LOGE("vaani_pipeline_transcribe: Peak amplitude across entire file was %d. "
             "Transcription result will be unreliable.", max_amp_global);
    }

    fclose(file);

    // Flush any remaining buffered audio
    char* trailing = vaani_stream_push_chunk(state, NULL, 0);
    if (trailing) {
        size_t seg_len = strlen(trailing);
        if (seg_len > 0) {
            LOGD("Trailing segment: '%s'", trailing);
            if (full_len + seg_len + 1 > full_cap) {
                full_cap = (full_len + seg_len + 1) * 2;
                char* tmp = realloc(full_transcript, full_cap);
                if (tmp) {
                    full_transcript = tmp;
                    memcpy(full_transcript + full_len, trailing, seg_len + 1);
                    full_len += seg_len;
                }
            } else {
                memcpy(full_transcript + full_len, trailing, seg_len + 1);
                full_len += seg_len;
            }
        }
        free(trailing);
    }

    LOGD("Transcription complete: %zu chars, %d segments", full_len, segments_found);
    if (full_len == 0) {
        LOGD("vaani_pipeline_transcribe: Empty transcript. Check amplitude, VAD threshold, and chunk sizes.");
    }

    vaani_stream_close(state);
    return full_transcript;
}