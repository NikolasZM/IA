#include "PatchEmbedder.hpp"
#include <cstdlib>
#include <cmath>
#include <iostream>

PatchEmbedder::PatchEmbedder(int patch_size, int embed_dim)
    : patch_size(patch_size),
      embed_dim(embed_dim),
      input_dim(patch_size * patch_size) {
    initialize_weights();
}

std::vector<std::vector<float>> PatchEmbedder::extract_patches(const float* image48x48) const {
    const int image_size = 48;
    const int filas = (image_size / patch_size);
    const int total_size = filas * filas;
    std::vector<std::vector<float>> patches;
    patches.reserve(total_size);

    for (int y = 0; y < image_size; y += patch_size) {
        for (int x = 0; x < image_size; x += patch_size) {
            std::vector<float> patch;
            patch.reserve(input_dim);
            for (int dy = 0; dy < patch_size; ++dy) {
                for (int dx = 0; dx < patch_size; ++dx) {
                    int idx = (y + dy) * image_size + (x + dx);
                    patch.push_back(image48x48[idx]);
                }
            }
            patches.push_back(patch);
        }
    }

    return patches;
}

float PatchEmbedder::ternarize(float x, float threshold) {
    if (x > threshold) return 1.0f;
    if (x < -threshold) return -1.0f;
    return 0.0f;
}

std::vector<float> PatchEmbedder::ternarize_vector(const std::vector<float>& input, float threshold) {
    std::vector<float> output;
    output.reserve(input.size());
    for (float x : input)
        output.push_back(ternarize(x, threshold));
    return output;
}

void PatchEmbedder::initialize_weights(float threshold) {
    weights.resize(embed_dim * input_dim);
    for (auto& w : weights) {
        float raw = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
        w = ternarize(raw, threshold);
    }
}

std::vector<std::vector<float>> PatchEmbedder::embed_patches(const std::vector<std::vector<float>>& patches) const {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(patches.size());

    for (const auto& patch : patches) {
        std::vector<float> embedded(embed_dim, 0.0f);
        std::vector<float> tern_patch = ternarize_vector(patch);

        for (int i = 0; i < embed_dim; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < input_dim; ++j) {
                acc += tern_patch[j] * weights[i * input_dim + j];
            }
            embedded[i] = acc;
        }

        embeddings.push_back(embedded);
    }

    return embeddings;
}

std::vector<std::vector<float>> PatchEmbedder::process(const float* image48x48) {
    auto patches = extract_patches(image48x48);
    return embed_patches(patches);
}
