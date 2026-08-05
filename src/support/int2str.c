// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <stdlib.h>
#include "dtypes.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

char *uint2str(char *dest, size_t len, uint64_t num, uint32_t base)
{
    int i = len-1;
    uint64_t b = (uint64_t)base;
    char ch;
    dest[i--] = '\0';
    while (i >= 0) {
        ch = (char)(num % b);
        if (ch < 10)
            ch += '0';
        else
            ch = ch-10+'a';
        dest[i--] = ch;
        num /= b;
        if (num == 0)
            break;
    }
    return &dest[i+1];
}

int isdigit_base(char c, int base)
{
    if (base < 11)
        return (c >= '0' && c < '0'+base);
    return ((c >= '0' && c <= '9') ||
            (c >= 'a' && c < 'a'+base-10) ||
            (c >= 'A' && c < 'A'+base-10));
}

#ifdef __cplusplus
}
#endif
