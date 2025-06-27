import numpy as np

def process_csv_to_bin(csv_path, x_bin_path, y_bin_path):
    print(f"Procesando: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

    y = data[:, 0].astype(np.uint8)     
    X = data[:, 1:] / 255.0         

    print(f"→ {X.shape[0]} muestras, {X.shape[1]} características por muestra")

    X.tofile(x_bin_path)
    y.tofile(y_bin_path)

    print(f"✅ Guardado: {x_bin_path}, {y_bin_path}\n")

train_csv = "fer2013_train_shuffled.csv"
test_csv  = "fer2013_test_shuffled.csv"

process_csv_to_bin(train_csv, "X_train.bin", "y_train.bin")
process_csv_to_bin(test_csv,  "X_test.bin",  "y_test.bin")
