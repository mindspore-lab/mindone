import glob
import os
import pandas as pd
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


def convert(data_dir, output_dir, img_fmt='jpg'):
    #assert os.path.exists(data_dir), f'{data_dir} not exists'
    #img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    #num_imgs = len(img_paths)
    #print("Get image num: ", num_imgs)

    folders = sorted(os.listdir(data_dir))
    len_postfix = len(img_fmt) + 1
    stat = {"min_h": 10e5, "min_w":10e5, "max_punsafe":-1, "min_aes": 10e5}
    num_imgs = 0
    num_small = 0
    log = open('laion_to_csv_log.txt', 'w')
    for folder in folders:
        img_paths = sorted(glob.glob(os.path.join(data_dir, f'{folder}/*.{img_fmt}')))
        if len(img_paths) > 0:
            relative_paths = []
            texts = []
            
            print('Folder: ', folder)
            for img_fp in tqdm(img_paths):
                text_fp = img_fp[:-len_postfix] + '.txt'
                json_fp = img_fp[:-len_postfix] + '.json'

                #with open(text_fp, 'r') as f:
                #    text = f.read()
                with open(json_fp, 'r') as f:
                    meta = json.load(f)
                    text = meta['caption']
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
                rel_path = folder + '/' + img_fp.split('/')[-1]
                relative_paths.append(rel_path) 

            
            frame = pd.DataFrame({"dir": relative_paths, "text": texts})
            csv_fn = folder + '.csv' 
            frame.to_csv(os.path.join(output_dir, csv_fn), index=False, sep=",")
        num_imgs += len(img_paths)
        print("Stat: ", stat)
    print("Num text-image pairts: ", num_imgs)
    print('Saved in ', output_dir)
    log.close()
    print("Num too small: ", num_small )
    print("Abnormal rate: ", num_small / num_imgs)

if __name__=='__main__':
    data_dir = '/data3/datasets/laion_art_filtered'
    #check_download_result(data_dir)
    convert(data_dir, data_dir)
