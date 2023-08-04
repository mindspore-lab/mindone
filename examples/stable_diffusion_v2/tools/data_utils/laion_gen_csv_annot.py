import argparse
import glob
import os
from img2dataset import download
from tqdm import tqdm
import json
import pandas as pd
#from pyspark.sql import SparkSession

# check completeness of download images
filter_width = 512

def check_download_result(data_dir='/data3/datasets/laion_art', img_fmt='jpg', download_fmt='files'):
    assert os.path.exists(data_dir), f'{data_dir} not exists'
    img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    num_imgs = len(img_paths)
    print("Get image num: ", num_imgs)

    # check total fails

    # check parquets in download image folder
    #spark = SparkSession.builder.config("spark.driver.memory", "2G") .master("local[4]").appName('spark-stats').getOrCreate()
    #df = spark.read.parquet(data_dir)
    fp = data_dir+ "/00000.parquet"
    print(fp)
    df = pd.read_parquet(fp)
    print(df.count())
    print(df.show())


def convert(data_dir, img_fmt='jpg', one_csv_per_part=True, check_data=False, folder_prefix=""):
    assert os.path.exists(data_dir), f'{data_dir} not exists'
    #img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    #num_imgs = len(img_paths)
    #print("Get image num: ", num_imgs)
    num_imgs = 0
    len_postfix = len(img_fmt) + 1

    #num_parts = len(glob.glob(os.path.join(data_dir, "part_*")))
    #part_folders = [fp for fp in glob.glob(os.path.join(data_dir, "part_*") if os.path.isdir(fp)]
    folders = [fp for fp in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fp))]
    if folder_prefix != "":
        folders = [f for f in folders if f.startswith(folder_prefix)]

    if check_data:
        log = open('laion_to_csv_log.txt', 'w')
        stat = {"min_h": 10e5, "min_w":10e5, "max_punsafe":-1, "min_aes": 10e5}
        num_small = 0

    #for part_id in range(1, num_parts+1):
    def _gather_img_text_in_folder(root_dir, folder, check_data=False):
        img_paths = sorted(glob.glob(os.path.join(root_dir, folder, f'*.{img_fmt}')))
        print('Image folder: ', folder, ', num imgs: ', len(img_paths))
        rel_img_paths = []
        texts = []
        if len(img_paths) > 0:
            for img_fp in tqdm(img_paths):
                text_fp = img_fp[:-len_postfix] + '.txt'
                json_fp = img_fp[:-len_postfix] + '.json'

                rel_img_paths.append(os.path.join(folder, os.path.basename(img_fp)))

                #with open(text_fp, 'r') as f:
                #    text = f.read()
                with open(json_fp, 'r') as f:
                    meta = json.load(f)
                    text = meta['caption']

                    if check_data:
                        stat['min_h'] = min(meta['original_height'], stat['min_h'])
                        stat['min_w'] = min(meta['original_width'], stat['min_w'])
                        stat['min_aes'] = min(meta['aesthetic'], stat['min_aes'])
                        stat['max_punsafe'] = max(meta['punsafe'], stat['max_punsafe'])

                        if meta['original_width'] < filter_width or meta['original_height'] < filter_width :
                        #if meta['aesthetic'] < 8.0:
                            print('Abnormal sample: ', meta['url'], meta['original_height'], meta['original_width'])
                            num_small += 1
                            log.write(f"{meta['original_height']}x{meta['original_width']}, {meta['url']} \n")

                texts.append(text)

        return rel_img_paths, texts

    def _save_to_csv(img_paths, texts, save_fp): 
        assert len(img_paths) == len(texts), f'{len(img_paths)} != {len(texts)}'
        frame = pd.DataFrame({"dir": img_paths, "text": texts})
        frame.to_csv(save_fp, index=False, sep=",")
        print('csv saved in ', save_fp)


    for folder in folders:
        # try to get image-text from level one folder
        rel_img_paths, texts = _gather_img_text_in_folder(data_dir, folder)
        if len(texts) > 0:
            save_fp = os.path.join(data_dir, folder + '.csv')
            _save_to_csv(rel_img_paths, texts, save_fp)
            num_imgs += len(rel_img_paths)
            
        # second level
        subfolders = [dn for dn in sorted(os.listdir(os.path.join(data_dir, folder))) if os.path.isdir(os.path.join(data_dir, folder, dn))]
        #img_folders = [os.path.join(folder, dn)  for dn in subfolders]
        print("Folder: ", folder, ", num sub folders: ", len(subfolders))

        rel_img_paths_all = []
        texts_all = []
        for subfolder in subfolders:
            rel_img_paths, texts = _gather_img_text_in_folder(os.path.join(data_dir, folder), subfolder)
            
            if len(rel_img_paths) > 0:
                if one_csv_per_part:
                    # csv saved along with folder
                    rel_img_paths= [folder+'/'+p for p in rel_img_paths] 
                    rel_img_paths_all.extend(rel_img_paths)
                    texts_all.extend(texts)
                else:
                    # csv saved under folder
                    #rel_img_paths= [os.path.join(p.split("/")[1:]) for p in rel_img_paths] 
                    save_fp = os.path.join(data_dir, folder, subfolder + '.csv')
                    _save_to_csv(rel_img_paths, texts, save_fp)

                num_imgs += len(rel_img_paths)

        if len(rel_img_paths_all) > 0:
            print("Saving csv...")
            save_fp = os.path.join(data_dir, folder + '.csv')
            print(len(rel_img_paths_all), len(texts_all))
            _save_to_csv(rel_img_paths_all, texts_all, save_fp)

    print("Num text-image pairts: ", num_imgs)
    print('All csv files are saved in ', data_dir)

    if check_data:
        log.close()
        print("Num too small: ", num_small )
        print("Stat: ", stat)
        print("Abnormal rate: ", num_small / num_imgs)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Save csv")
    parser.add_argument("--data_dir", type=str, default='/Volumes/Extreme_SSD/LAION/sd2.1_base_train', help="dir containing the downloaded images")
    parser.add_argument("--folder_prefix", type=str, default='', help="folder prefix to filter unwanted folders. e.g. part")
    parser.add_argument("--save_csv_per_img_folder", type=bool, default=False, help="If False, save a csv file for each part, which will result in a large csv file (~400MB). If True, save a csv file for each image folder, which will result in hundreads of csv files for one part of dataset.")
    args = parser.parse_args()

    #data_dir = '/data3/datasets/laion_art_filtered'
    data_dir = args.data_dir
    #check_download_result(data_dir)
    convert(data_dir, one_csv_per_part=not args.save_csv_per_img_folder, folder_prefix=args.folder_prefix)
