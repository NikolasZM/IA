#include "BitVisionTransformer.hpp"


BitVisionTransformer::BitVisionTransformer(int patch_size, int d_model, int num_classes, int num_layers, float threshold)
    : d_model(d_model),
      num_layers(num_layers),
      patch_embedder(patch_size, d_model),
      pos_embedding(64, d_model, threshold),
      classifier(d_model, num_classes, threshold, "model_weights/classifier_weight.bin") {
    cout << "Creando desde BitVisionTransformer.hpp arriba \n";
    cout << "num_layers: " << num_layers << endl;
    for (int i = 0; i < num_layers; ++i)
        encoders.emplace_back(BitTransformerEncoderLayer(d_model, d_model * 2, threshold));
}

int BitVisionTransformer::predict(const float* image_data) {
    vector<vector<float>> tokens = patch_embedder.process(image_data);

    pos_embedding.apply(tokens);

    for (int i = 0; i < num_layers; ++i)
        tokens = encoders[i].forward(tokens);

    vector<float> pooled = pooling.forward(tokens);

    vector<float> logits = classifier.forward(pooled);

    return argmax(logits);
}
