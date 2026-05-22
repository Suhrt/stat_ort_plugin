#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stat_ort_plugin.h"


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
        vocab = realloc(vocab, (count + 1) * sizeof(char*));
        vocab[count] = strdup(line);
        count++;
    }

    fclose(f);
    *out_size = count;
    return vocab;
}