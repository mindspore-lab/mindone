import argparse
import csv
import hashlib
import io
import json
import os
import tarfile

from PIL import Image


def create_laion_webdataset(csv_path, image_folder, output_tar, samples_per_tar=1000):
    """Convert dataset to WebDataset format following LAION dataset structure

    Args:
        csv_path: Path to the CSV file containing image paths and captions
        image_folder: Root folder containing the images
        output_tar: Output path template, e.g. "dataset_{}.tar"
        samples_per_tar: Number of samples to store in each tar file
    """
    with open(csv_path, "r") as f:
        reader = list(csv.DictReader(f))

    num_samples = len(reader)
    num_tars = (num_samples + samples_per_tar - 1) // samples_per_tar

    print(f"Processing {num_samples} samples into {num_tars} tar files...")

    for tar_idx in range(num_tars):
        # Generate tar filename with index if multiple tars
        if num_tars > 1:
            tar_path = (
                output_tar.format(tar_idx) if "{}" in output_tar else output_tar.replace(".tar", f"_{tar_idx:05d}.tar")
            )
        else:
            tar_path = output_tar

        start_idx = tar_idx * samples_per_tar
        end_idx = min((tar_idx + 1) * samples_per_tar, num_samples)

        print(f"Creating tar file {tar_path} with samples {start_idx} to {end_idx}")

        with tarfile.open(tar_path, "w") as tar:
            for idx, row in enumerate(reader[start_idx:end_idx], start=start_idx):
                # Get image path and caption
                image_path = os.path.join(image_folder, row["image_path"])
                caption = row.get("text_en", "")

                try:
                    # Load and process image
                    img = Image.open(image_path)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    # Calculate image hash
                    img_hash = hashlib.sha256(img_bytes.getvalue()).hexdigest()

                    # Get image dimensions
                    width, height = img.size

                    # Create LAION-style metadata
                    metadata = {
                        "caption": caption,
                        "hash": img_hash,
                        "original_height": height,
                        "original_width": width,
                        "height": height,
                        "width": width,
                        "url": f"local:{image_path}",
                        "format": "png",
                        "size": len(img_bytes.getvalue()),
                        "sha256": img_hash,
                    }

                    # Add optional fields if available in CSV
                    if "NSFW" in row:
                        metadata["NSFW"] = row["NSFW"]
                    if "similarity" in row:
                        metadata["similarity"] = float(row["similarity"])
                    if "punsafe" in row:
                        metadata["punsafe"] = float(row["punsafe"])

                    key = f"{idx:08d}"

                    # Save image
                    img_info = tarfile.TarInfo(f"{key}.png")
                    img_info.size = len(img_bytes.getvalue())
                    tar.addfile(img_info, img_bytes)

                    # Save caption
                    txt_content = caption.encode("utf-8")
                    txt_info = tarfile.TarInfo(f"{key}.txt")
                    txt_info.size = len(txt_content)
                    txt_bytes = io.BytesIO(txt_content)
                    tar.addfile(txt_info, txt_bytes)

                    # Save metadata
                    json_content = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
                    json_info = tarfile.TarInfo(f"{key}.json")
                    json_info.size = len(json_content)
                    json_bytes = io.BytesIO(json_content)
                    tar.addfile(json_info, json_bytes)

                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue

        print(f"Finished creating {tar_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert image dataset to WebDataset format")
    parser.add_argument(
        "--csv-path", type=str, required=True, help="Path to the CSV file containing image paths and captions"
    )
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder containing the images")
    parser.add_argument(
        "--output-tar", type=str, required=True, help='Output path template, e.g. "dataset_{}.tar" or "dataset.tar"'
    )
    parser.add_argument(
        "--samples-per-tar", type=int, default=1000, help="Number of samples to store in each tar file (default: 1000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_laion_webdataset(
        csv_path=args.csv_path,
        image_folder=args.image_folder,
        output_tar=args.output_tar,
        samples_per_tar=args.samples_per_tar,
    )
    print("Conversion completed successfully!")
