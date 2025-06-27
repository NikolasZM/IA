#include "BitVisionTransformer.hpp"
#include <fstream>
#include <iostream>

int main() {
    std::ifstream in("prePros/X_test.bin", std::ios::binary);
    float image[48 * 48];
    in.read(reinterpret_cast<char*>(image), sizeof(image));
    in.close();

    BitVisionTransformer model(6, 64, 7, 2);  // patch 6x6, d_model=64, 7 clases, 2 capas encoder

    int predicted = model.predict(image);
    cout << "Emoción predicha: " << predicted << "\n";

    return 0;
}
