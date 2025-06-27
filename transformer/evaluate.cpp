#include "BitVisionTransformer.hpp"
#include <fstream>
#include <iostream>
#include <vector>


int main() {

 const char* path = "classifier_weight.bin";
    ifstream file(path, ios::binary);

    if (!file) {
        cerr << " ERROR: No se encontro el archivo: " << path << endl;
    } else {
        cout << " EXITO: Archivo encontrado correctamente.\n";

        float buffer[5];
        file.read(reinterpret_cast<char*>(buffer), sizeof(buffer));

        cout << "Primeros 5 valores float del archivo:\n";
        for (int i = 0; i < 5; ++i) {
            cout << buffer[i] << " ";
        }
        cout << "\n";
    }


    const int img_size = 48 * 48;
    const int num_classes = 7;

    ifstream x_file("X_test.bin", ios::binary);
    ifstream y_file("y_test.bin", ios::binary);

    if (!x_file || !y_file) {
        cerr << "Error abriendo los archivos X_train.bin o Y_train.bin.\n";
        return 1;
    }
    cout <<"Iniciando proceso" << endl;

    x_file.seekg(0, ios::end);
    size_t x_bytes = x_file.tellg();
    size_t num_samples = x_bytes / (img_size * sizeof(float));
    x_file.seekg(0, ios::beg);
    cout << "xbtes: " << x_bytes << " - num_samples: " << num_samples << endl;
    vector<float> image(img_size);
    vector<int> labels(num_samples);
    y_file.read(reinterpret_cast<char*>(labels.data()), num_samples * sizeof(int));

    BitVisionTransformer model(6, 64, num_classes, 2);  // patch=6x6, d_model=64, clases=7, capas=2

    int correct = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        cout << "Dato i:" << i << endl;
        x_file.read(reinterpret_cast<char*>(image.data()), img_size * sizeof(float));
        int prediction = model.predict(image.data());
        if (prediction == labels[i])
            correct++;
    }

    float accuracy = (float)correct / num_samples * 100.0f;
    cout << "Precision: " << accuracy << "% (" << correct << "/" << num_samples << ")\n";

    return 0;
}
