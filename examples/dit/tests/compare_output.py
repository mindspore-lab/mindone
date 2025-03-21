import numpy as np


def load_npy_file(file_path):
    return np.load(file_path)


def calculate_mse(output1, output2):
    return np.mean((output1 - output2) ** 2)


def main():
    ms_output = load_npy_file("ms_output.npy")
    pt_output = load_npy_file("pt_output.npy")

    mse = calculate_mse(ms_output, pt_output)

    print(f"Mean Squared Error (MSE): {mse}")

    if mse < 0.001:
        print("The mse is less than 0.001, the model is correct.")


if __name__ == "__main__":
    main()
