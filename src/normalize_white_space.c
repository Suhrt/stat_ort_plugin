#include <ctype.h>
#include <string.h>

void normalize_whitespace(char* str) {
    if (!str) return;
    char* read = str;
    char* write = str;
    int in_space = 1;

    while (*read) {
        // CHANGED: Identify and replace the "▁" token with a space, advancing read pointer by token byte length
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