import csv
import os
from subprocess import Popen, PIPE


def download_from_url(cmd, url, filename):
    if os.path.exists(filename):
        return
    try:
        process = Popen(
            f'{cmd} {url}',
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            shell=True
        )
        print('running:', f'{cmd} {url}')
        std_out, std_err = process.communicate()
    except Exception as e:
        print('download failed', e)
        raise e

def get_downloader():
    url = 'https://raw.githubusercontent.com/openimages/dataset/master/downloader.py'
    filename = 'downloader.py'
    download_from_url('wget', url, filename)

def get_metadata(split='test'):
    filename = {
        'train': 'train-images-boxable-with-rotation.csv',
        'test': 'test-images-with-rotation.csv',
        'validation': 'validation-images-with-rotation.csv',
    }[split]
    url = f'https://storage.googleapis.com/openimages/2018_04/{split}/{filename}',
    download_from_url('wget', url, filename)
    return filename

def get_images(folder='openimage_test', split='test'):
    os.makedirs(folder, exist_ok=True)
    get_downloader()
    filename = get_metadata(split=split)
    temp = 'temp_ids.txt'
    with open(temp, 'w', newline='\n') as fw, open(filename, newline='\n') as fr:
        reader = csv.DictReader(fr)
        i = 0
        for line in reader:
            i += 1
            Subset = line['Subset']
            ImageID = line['ImageID']
            s = f'{Subset}/{ImageID}\n'
            fw.write(s)
    print('downloading %d images...' % i)
    download_from_url(f'python downloader.py {temp}', f'--download_folder {folder} --num_processes 8', 'ghost_file')

if __name__ == '__main__':
    get_images(folder='openimage_train', split='train')
    get_images(folder='openimage_validation', split='validation')
    get_images(folder='openimage_test', split='test')
