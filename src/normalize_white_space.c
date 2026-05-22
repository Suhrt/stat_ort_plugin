#include <ctype.h>
#include "stat_ort_plugin.h"

void normalize_whitespace(char* str) {
    if (!str) return;
    char* read = str;
    char* write = str;
    int in_space = 1;

    while (*read) {
        if (isspace((unsigned char)*read)) {
            if (!in_space) {
                *write++ = ' ';
                in_space = 1;
            }
        } else {
            *write++ = *read;
            in_space = 0;
        }
        read++;
    }

    if (write > str && *(write - 1) == ' ') {
        write--;
    }
    *write = '\0';
}