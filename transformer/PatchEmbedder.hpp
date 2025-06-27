#pragma once
#include <vector>

using namespace std;

class PatchEmbedder {
public:
    PatchEmbedder(int patch_size, int embed_dim);

    vector<vector<float>> extract_patches(const float* image48x48) const;
    vector<vector<float>> embed_patches(const vector<vector<float>>& patches) const;
    vector<vector<float>> process(const float* image48x48);

private:
    int patch_size;
    int embed_dim;
    int input_dim;
    vector<float> weights;

    static float ternarize(float x, float threshold = 0.333333f);
    static vector<float> ternarize_vector(const vector<float>& input, float threshold = 0.333333f);
    void initialize_weights(float threshold = 0.333333f);
};
