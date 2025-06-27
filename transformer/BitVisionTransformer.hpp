#pragma once

#include "PatchEmbedder.hpp"
#include "PositionalEmbedding.hpp"
#include "BitTransformerEncoderLayer.hpp"
#include "GlobalAveragePooling.hpp"
#include "BitLinear.hpp"
#include "ClassifierUtils.hpp"

#include <iostream>
using namespace std;

class BitVisionTransformer {
public:
    BitVisionTransformer(int patch_size, int d_model, int num_classes, int num_layers, float threshold = 0.33f);

    int predict(const float* image_data);

private:
    int d_model;
    int num_layers;

    PatchEmbedder patch_embedder;
    PositionalEmbedding pos_embedding;
    vector<BitTransformerEncoderLayer> encoders;
    GlobalAveragePooling pooling;
    BitLinear classifier;
};
