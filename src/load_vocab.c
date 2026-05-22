#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stat_ort_plugin.h"


char** load_vocab(const char* vocab_path, int* out_size) {
    LOGD("Attempting to load vocab from: %s", vocab_path);
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        LOGE("Failed to open vocab file at %s", vocab_path);
        return NULL;
    }

    char** vocab = NULL;
    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;

        // STRIP ID: Find the last space and terminate the string there
        char* last_space = strrchr(line, ' ');
        if (last_space != NULL) {
            *last_space = '\0';
        }

        vocab = realloc(vocab, (count + 1) * sizeof(char*));
        vocab[count] = strdup(line);
        count++;
    }
    fclose(f);
    *out_size = count;
    LOGD("Successfully loaded %d vocab tokens", count);
    return vocab;
}