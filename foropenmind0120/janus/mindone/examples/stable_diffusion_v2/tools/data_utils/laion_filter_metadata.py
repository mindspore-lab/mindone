"""
Available fields to filter:
for laion-art:
['URL',
 'TEXT',
 'WIDTH',
 'HEIGHT',
 'similarity',
 'LANGUAGE',
 'hash',
 'pwatermark',
 'punsafe',
 'aesthetic']

 for laion2b-en:
 ['URL', 'TEXT', 'WIDTH', 'HEIGHT', 'similarity', 'punsafe', 'pwatermark', 'AESTHETIC_SCORE', 'hash', '__index_level_0__']
"""
import argparse
import glob
import os
import shutil

from pyspark.sql import SparkSession, functions
from pyspark.sql.functions import rand

# predefined filter conditions for sd 2.1 base training
filter_width = 512
filter_height = 512
lang = "en"
filter_aes = 4.5
filter_punsafe = 0.98  # used in sd2.1 base training


def filter_metadata(
    data_path_or_dir, width=None, height=None, num_repartitions=1, lang="en", output_dir=None, dataset_name="laion2b_en"
):
    assert os.path.exists(data_path_or_dir), f"{data_path_or_dir} not exists"
    spark = (
        SparkSession.builder.config("spark.driver.memory", "16G")
        .master("local[16]")
        .appName("spark-stats")
        .getOrCreate()
    )
    # spark = SparkSession.builder.config("spark.driver.memory", "32G") .master("local[*]").appName('spark-stats').getOrCreate()
    df = spark.read.parquet(data_path_or_dir)
    num_ori = df.count()
    print("Num samples: ", num_ori)
    print("Examples: ", df.show(10))
    print("Availabe fields to filter: ", df.schema.names)

    if dataset_name.startswith("laion_art"):
        df = df.filter(
            (df.WIDTH >= filter_width)
            & (df.HEIGHT >= filter_height)
            & (df.LANGUAGE == lang)
            & (df.punsafe <= filter_punsafe)
            & (df.aesthetic >= filter_aes)
        )
    else:
        df = df.filter(
            (df.WIDTH >= filter_width)
            & (df.HEIGHT >= filter_height)
            & (df.punsafe <= filter_punsafe)
            & (df.AESTHETIC_SCORE >= filter_aes)
        )

    df = df.orderBy(rand())  # this line is important to have a shuffled dataset
    if num_repartitions > 1:
        df.repartition(num_repartitions).write.parquet(output_dir)
    else:
        df.repartition(num_repartitions).write.parquet(output_dir, mode="overwrite")

    num_after_filtered = df.count()
    print("Num samples after filtering: ", num_after_filtered)
    print(df.select(functions.min("WIDTH")).collect())
    print(df.select(functions.min("HEIGHT")).collect())

    return num_after_filtered


def check_filtered_metadata(data_dir, dataset_name="laion2b_en"):
    assert os.path.exists(data_dir), f"{data_dir} not exists"
    spark = (
        SparkSession.builder.config("spark.driver.memory", "16G")
        .master("local[16]")
        .appName("spark-stats")
        .getOrCreate()
    )
    df = spark.read.parquet(data_dir)
    num_samples = df.count()
    print("Num samples : ", num_samples)
    # print('Min w: ', df.select(functions.min("WIDTH")).collect())
    # print('Min h: ', df.select(functions.min("HEIGHT")).collect())
    if dataset_name.startswith("laion2b"):
        min_aes = df.select(functions.min("AESTHETIC_SCORE")).collect()
        max_punsafe = df.select(functions.max("punsafe")).collect()
        num_punsafe_ood = df.select().where((df.punsafe > 0.5)).count()
        print("Min aes: ", min_aes)
        print("Max punsafe: ", max_punsafe)
        print("Num punsafe > 0.5: ", num_punsafe_ood, num_punsafe_ood / num_samples)
    min_sim = df.select(functions.min("similarity")).collect()
    num_sim_ood = (
        df.select().where((df.similarity < 0.28)).count()
    )  # according to laion doc, samples with similariy < 0.28 should have been dropped
    print("Min similarity: ", min_sim)
    print("Num sim < 0.28: ", num_sim_ood, num_sim_ood / num_samples)
    # from laion: we recommend 0.5 for safety and 0.8 for watermark
    max_watermark = df.select(functions.max("pwatermark")).collect()
    num_watermark_ood = df.select().where((df.pwatermark > 0.8)).count()
    print("Max pwatermark: ", max_watermark)
    print("Num pwatermark > 0.8: ", num_watermark_ood, num_watermark_ood / num_samples)


def rename_parquet_files(data_dir):
    dirs = os.listdir(data_dir)
    for dn in dirs:
        # 2B-en-4.5_1
        if dn.startswith("2B-en"):
            part_id = int(dn.split("_")[-1])
            save_fn = f"part_{part_id}.parquet"
            parquet_files = glob.glob(os.path.join(data_dir, dn) + "/*.parquet")
            assert os.path.exists(os.path.join(data_dir, dn) + "/_SUCCESS"), f"{dn} not successfully written"
            assert len(parquet_files) == 1, "Expecting one parquet file in each part folder"
            save_fp = os.path.join(data_dir, save_fn)
            print("Move ", parquet_files[0], " to ", save_fp)
            shutil.move(parquet_files[0], save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metadata Filter")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/Extreme_SSD/LAION/2b_en_ae4.5_meta",
        help="dir containing source parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Volumes/Extreme_SSD/laion2b_en/sd2.1_base_train/metadata_filtered",
        help="dir to saved the filtered metadata",
    )
    parser.add_argument("--dataset_name", type=str, default="laion2b_en", help="laion_art or laion2b_en")
    parser.add_argument("--task", type=str, default="filter", help="filter or check")
    args = parser.parse_args()

    # data_path_or_dir='/data3/datasets/laion_art_metadata'
    data_path_or_dir = args.data_dir
    output_dir = args.output_dir
    dataset_name = args.dataset_name
    num_repartitions = 1

    if args.task == "filter":
        if dataset_name.startswith("laion_art"):
            res_num = filter_metadata(
                data_path_or_dir,
                filter_width,
                filter_height,
                num_repartitions,
                lang,
                output_dir=output_dir,
                dataset_name=dataset_name,
            )
        else:
            # TODO: due to limited memory size, we need to do it one by one. Just set num_repartition=64 to finish the filtering on a machine with large memory.
            # num_repartitions = 64
            fps = sorted(glob.glob(data_path_or_dir + "/*.parquet"))
            res_num = 0
            for i, data_path in enumerate(fps):
                output_path = output_dir + "/" + os.path.basename(data_path).replace(".parquet", "")
                # output_path = output_dir + '/' + os.path.basename(data_path)
                print("=> Start filtering ", data_path)
                cur_num = filter_metadata(
                    data_path,
                    filter_width,
                    filter_height,
                    num_repartitions,
                    lang,
                    output_dir=output_path,
                    dataset_name=dataset_name,
                )
                res_num += cur_num

            rename_parquet_files(output_dir)

            print(f"{res_num} samples in total after filtering")

    check_filtered_metadata(output_dir, dataset_name)
