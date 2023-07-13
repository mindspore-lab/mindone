import glob
import os
import pandas as pd
from img2dataset import download
from tqdm import tqdm
import json

# check completeness of download images

def check(data_dir='/data3/datasets/laion_art', img_fmt='jpg', download_fmt='files'):
    assert os.path.exists(data_dir), f'{data_dir} not exists'
    img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    num_imgs = len(img_paths)
    print("Get image num: ", num_imgs)


def convert(data_dir, output_dir, img_fmt='jpg'):
    #assert os.path.exists(data_dir), f'{data_dir} not exists'
    #img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    #num_imgs = len(img_paths)
    #print("Get image num: ", num_imgs)

    folders = sorted(os.listdir(data_dir))
    len_postfix = len(img_fmt) + 1
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
                    text = json.load(f)['caption']
                texts.append(text) 
                rel_path = folder + '/' + img_fp.split('/')[-1]
                relative_paths.append(rel_path) 

            
            frame = pd.DataFrame({"dir": relative_paths, "text": texts})
            csv_fn = folder + '.csv' 
            frame.to_csv(os.path.join(output_dir, csv_fn), index=False, sep=",")


if __name__=='__main__':
    data_dir = '/data3/datasets/laion_art_filtered'
    #check(data_dir)
    convert(data_dir, data_dir)
