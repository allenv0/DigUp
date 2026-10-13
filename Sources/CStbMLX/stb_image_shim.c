#include "include/stb_image_shim.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "../../Vendor/llama.cpp/vendor/stb/stb_image.h"

uint8_t *stb_image_load_rgb(const char *path, int *width, int *height) {
    int channels = 0;
    return stbi_load(path, width, height, &channels, 3);
}
