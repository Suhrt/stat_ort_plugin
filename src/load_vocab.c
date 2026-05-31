#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vaani_internal.h"

char** load_vocab(const char* vocab_path, int* out_size) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        return NULL;
    }

    char** vocab = NULL;
    int count = 0;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;
        char* last_space = strrchr(line, ' ');
        if (last_space != NULL) {
            *last_space = '\0';
        }
        char** tmp = realloc(vocab, (count + 1) * sizeof(char*));
        if (!tmp) {
            for (int i = 0; i < count; i++) free(vocab[i]);
            free(vocab);
            fclose(f);
            return NULL;
        }
        vocab = tmp;
        vocab[count] = strdup(line);
        if (!vocab[count]) {
            // Out of memory: free everything already loaded so callers never
            // see a vocab array with a NULL hole in it (which would crash the
            // decode loop and vaani_pipeline_free).
            for (int i = 0; i < count; i++) free(vocab[i]);
            free(vocab);
            fclose(f);
            return NULL;
        }
        count++;
    }

    fclose(f);
    *out_size = count;
    return vocab;
}