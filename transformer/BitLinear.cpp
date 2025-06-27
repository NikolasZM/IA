#include "BitLinear.hpp"
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

//const string DEFAULT_WEIGHT_PATH = "classifier_weight.bin";

BitLinear::BitLinear(int in_features, int out_features, float threshold, string filename)
    : in_features(in_features), out_features(out_features), filename(filename) {
    weights.resize(out_features, vector<float>(in_features));
    initialize_weights(threshold);
}

void BitLinear::initialize_weights(float threshold) {
    ifstream file(filename, ios::binary);
    
    if (file) {
        vector<float> raw_weights(out_features * in_features);
        file.read(reinterpret_cast<char*>(raw_weights.data()), 
                 out_features * in_features * sizeof(float));

        std::streamsize bytes_read = file.gcount();
cout << "Bytes leídos: " << bytes_read << endl;
cout << "Esperados:    " << (out_features * in_features * sizeof(float)) << endl;

        
        if (file) {
            for (int i = 0; i < out_features; ++i) {
                for (int j = 0; j < in_features; ++j) {
                    weights[i][j] = ternarize(raw_weights[i * in_features + j], threshold);
                }
            }
            return;
        }
    }
    
    cerr << "Advertencia: No se encontraron pesos pre-entrenados en " 
         << filename << ". Usando inicialización aleatoria." << endl;
    
    for (auto& row : weights) {
        for (auto& w : row) {
            float raw = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
            w = ternarize(raw, threshold);
        }
    }
}

float BitLinear::ternarize(float x, float threshold) const {
    if (x > threshold) return 1.0f;
    if (x < -threshold) return -1.0f;
    return 0.0f;
}

vector<float> BitLinear::forward(const vector<float>& input) const {
    if ((int)input.size() != in_features)
        throw runtime_error("Dimensión de entrada incorrecta en BitLinear");

    vector<float> output(out_features, 0.0f);
    vector<float> bin_input(in_features);

    for (int i = 0; i < in_features; ++i)
        bin_input[i] = ternarize(input[i]);

    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j)
            output[i] += bin_input[j] * weights[i][j];
    }

    return output;
}