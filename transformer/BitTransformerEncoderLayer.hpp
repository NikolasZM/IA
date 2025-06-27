#pragma once
#include "BitAttention.hpp"
#include <stdexcept>
#include "BitMLP.hpp"

class BitTransformerEncoderLayer {
public:
    BitTransformerEncoderLayer(int dim, int hidden_dim = 128, float threshold = 0.33f);

    vector<vector<float>> forward(const vector<vector<float>>& tokens);

private:
    BitAttention attention;
    BitMLP mlp;

    vector<float> add(const vector<float>& a, const vector<float>& b) const;
};
