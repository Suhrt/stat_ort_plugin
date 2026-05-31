#include <ctype.h>
#include <string.h>
#include "vaani_internal.h"

void normalize_whitespace(char* str) {
    if (!str) return;
    char* read = str;
    char* write = str;
    int in_space = 1;

    while (*read) {
        if (strncmp(read, "▁", strlen("▁")) == 0) {
            if (!in_space) {
                *write++ = ' ';
                in_space = 1;
            }
            read += strlen("▁");
        }
        else if (isspace((unsigned char)*read)) {
            if (!in_space) {
                *write++ = ' ';
                in_space = 1;
            }
            read++;
        } else {
            *write++ = *read++;
            in_space = 0;
        }
    }

    if (write > str && *(write - 1) == ' ') {
        write--;
    }
    *write = '\0';
}