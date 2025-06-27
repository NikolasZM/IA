#pragma once
#include <vector>

#include "BitLinear.hpp"
#include <string>
using namespace std;

class BitAttention {
public:
    BitAttention(int dim, float threshold = 0.33f);

    vector<vector<float>> forward(const vector<vector<float>>& X);

private:
    int dim;

    BitLinear W_q;
    BitLinear W_k;
    BitLinear W_v;
    BitLinear W_o;

    vector<float> softmax(const vector<float>& input) const;
};
