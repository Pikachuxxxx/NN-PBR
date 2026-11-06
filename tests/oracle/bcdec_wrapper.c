#define BCDEC_IMPLEMENTATION
#include "bcdec.h"
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
EXPORT void decode_block(const void* input, float* output, int is_signed) {
    bcdec_bc6h_float(input, output, 12, is_signed);
}
