#pragma once
#include "BitLinear.hpp"
#include <vector>

class BitMLP {
public:
    BitMLP(int dim, int hidden_dim, float threshold = 0.33f);

    vector<float> forward(const vector<float>& x) const;

private:
    BitLinear fc1;
    BitLinear fc2;

    float relu(float x) const;
};
