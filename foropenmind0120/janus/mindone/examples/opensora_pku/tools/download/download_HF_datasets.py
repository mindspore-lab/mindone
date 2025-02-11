import argparse
import os

from huggingface_hub import hf_hub_download
from tqdm import tqdm

mixkit_file_names = [
    "Airplane.tar",
    "Baby.tar",
    "Bicycle.tar",
    "Birds.tar",
    "Business.tar",
    "Car.tar",
    "Cats.tar",
    "City.tar",
    "Couple.tar",
    "Dance.tar",
    "Dogs.tar",
    "Drive.tar",
    "Family.tar",
    "Fashion.tar",
    "Fish.tar",
    "Girl.tar",
    "House.tar",
    "Life.tar",
    "Man.tar",
    "Monkey.tar",
    "Motocycle.tar",
    "Music.tar",
    "Party.tar",
    "People.tar",
    "Pets.tar",
    "Reptiles.tar",
    "Road.tar",
    "Safari.tar",
    "Shark.tar",
    "Sport.tar",
    "Street.tar",
    "Taxi.tar",
    "Traffic.tar",
    "Trains.tar",
    "Truck.tar",
    "Wildlife.tar",
    "Woman.tar",
    "Zoo.tar",
    "beach.tar",
    "clouds.tar",
    "earth.tar",
    "fire.tar",
    "flower.tar",
    "forest.tar",
    "mountain.tar",
    "night.tar",
    "rain.tar",
    "sea.tar",
    "sky.tar",
    "smoke.tar",
    "snow.tar",
    "space.tar",
    "sun.tar",
    "sunset.tar",
    "water.tar",
]

anno_json_files = [
    "anytext_en_1886137.json",
    "human_images_162094.json",
    "sam_image_11185255.json",
    "video_mixkit_513f_1997.json",
    "video_mixkit_65f_54735.json",
    "video_pexel_65f_3832666.json",
    "video_pexels_513f_271782.json",
    "video_pixabay_513f_51483.json",
    "video_pixabay_65f_601513.json",
]


def download_files(repo_id, local_dir, subfolder, filenames):
    for filename in tqdm(filenames, total=len(filenames)):
        target_fp = os.path.join(local_dir, subfolder, filename)
        if os.path.exists(target_fp):
            print(f"{target_fp} exist, skip downloading...")
            continue
        else:
            hf_hub_download(
                repo_id=repo_id, repo_type="dataset", subfolder=subfolder, local_dir=local_dir, filename=filename
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="The local directory where the downloaded datasets will be saved.",
    )
    args = parser.parse_args()
    #################################################################################
    #                       Download Datasets from HF                               #
    #################################################################################
    local_dir = args.output_dir
    repo_id = "LanguageBind/Open-Sora-Plan-v1.1.0"

    # download json files
    subfolder = "anno_jsons"
    download_files(repo_id, local_dir, subfolder, anno_json_files)

    # # download human images
    subfolder = "human_image"
    download_files(repo_id, local_dir, subfolder, ["images.tar.gz"])

    # # download_mixkit
    subfolder = "all_mixkit"
    download_files(repo_id, local_dir, subfolder, mixkit_file_names)

    # download pixabay_v2
    file_names = [f"folder_{index:02d}.tar.gz" for index in range(1, 51)]
    subfolder = "pixabay_v2_tar"
    download_files(repo_id, local_dir, subfolder, file_names)

    # download pexels
    file_names = []
    subfolder = "pexels"
    for index in range(0, 96):
        for part in range(5):
            file_name = f"5000-{index}.tar.part{part}"
            file_names += [file_name]

    download_files(repo_id, local_dir, subfolder, file_names)
