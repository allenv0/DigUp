#ifndef STB_IMAGE_SHIM_H
#define STB_IMAGE_SHIM_H
#include <stddef.h>
#include <stdint.h>

// Decodes an image file into packed RGB, 8-bit, exactly like mtmd's stb_image path (same decoder, same
// channel order), so the MLX vision tower sees the pixels llama.cpp sees. Returns malloc'd bytes (free(3)),
// or NULL when the file can't be read.
uint8_t *stb_image_load_rgb(const char *path, int *width, int *height);
#endif
