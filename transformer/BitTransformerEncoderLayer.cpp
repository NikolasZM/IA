#include "BitTransformerEncoderLayer.hpp"

BitTransformerEncoderLayer::BitTransformerEncoderLayer(int dim, int hidden_dim, float threshold)
    : attention(dim, threshold), mlp(dim, hidden_dim, threshold) {}

vector<vector<float>> BitTransformerEncoderLayer::forward(const vector<vector<float>>& tokens) {
    int n_tokens = tokens.size();

    vector<vector<float>> attended = attention.forward(tokens);

    vector<vector<float>> residual(n_tokens);
    for (int i = 0; i < n_tokens; ++i)
        residual[i] = add(tokens[i], attended[i]);

    //MLP
    vector<vector<float>> output(n_tokens);
    for (int i = 0; i < n_tokens; ++i)
        output[i] = mlp.forward(residual[i]);

    return output;
}

vector<float> BitTransformerEncoderLayer::add(const vector<float>& a, const vector<float>& b) const {
    if (a.size() != b.size())
        throw runtime_error("Dimensiones incompatibles en residual.");
    vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}
