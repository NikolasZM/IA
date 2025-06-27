#pragma once
#include <vector>
#include <string>
using namespace std;

class BitLinear {
public:
    BitLinear(int in_features, int out_features, float threshold, string filename) ;

    vector<float> forward(const vector<float>& input) const;
    void initialize_weights(float threshold);

private:
    int in_features;
    int out_features;
    vector<vector<float>> weights;
    string filename;

    float ternarize(float x, float threshold = 0.33333f) const;
};
