import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


def load_index_file(index_file):
    with open(index_file, "r") as f:
        return json.load(f)


def _load_huggingface_safetensor(ckpt_file):
    db_state_dict = {}
    with safe_open(ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            db_state_dict[key] = f.get_tensor(key)
    return db_state_dict


def merge_safetensors(input_folder, index_file, output_file):
    # Load the index file
    index_data = load_index_file(index_file)
    # Iterate through the files specified in the index
    weight_map = index_data.get("weight_map", {})
    weight_names = []
    file_paths = []
    for weight_name in weight_map.keys():
        file_paths.append(weight_map[weight_name])
        weight_names.append(weight_name)
    file_paths = set(file_paths)
    weight_names = set(weight_names)

    sd = []
    for file_path in file_paths:
        if file_path:
            file_path = os.path.join(input_folder, file_path)
            partial_sd = _load_huggingface_safetensor(file_path)
            sd.append(partial_sd)

    # Merge all tensors together
    merged_tensor = sd[0]
    for tensor in sd[1:]:
        merged_tensor.update(tensor)

    # Save the merged tensor to a new Safetensor file
    save_file(merged_tensor, output_file)
    print(f"Merged Safetensors saved as: {output_file}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Merge multiple Safetensors files into one using an index.")
    parser.add_argument("--input_folder", "-i", type=str, help="Path to the folder containing Safetensors files.")
    parser.add_argument("--index_file", "-f", type=str, help="Path to the index JSON file.")
    parser.add_argument("--output_file", "-o", type=str, help="Path to the output merged Safetensors file.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the merge function
    assert args.output_file.endswith(".safetensors")
    merge_safetensors(args.input_folder, args.index_file, args.output_file)


if __name__ == "__main__":
    main()
