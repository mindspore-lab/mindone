from vc.data.dataset_train import VideoDatasetForTrain, build_dataset
from vc.data.transforms import create_transforms 
from configs.train_config import cfg
import time
import mindspore as ms
from mindspore import dataset as ds

import sys
sys.path.append("../stable_diffusion_v2/")
from tools._common.clip import CLIPTokenizer


def test_dataset():
    data_dir = './demo_video'
    #cfg.max_frames = 16
    #cfg.batch_size = 1

    tokenizer = CLIPTokenizer("./model_weights/bpe_simple_vocab_16e6.txt.gz")
    '''
    infer_transforms, misc_transforms, mv_transforms, vit_transforms = create_transforms(cfg)
    dataset = VideoDatasetForTrain(
        root_dir=data_dir,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_transforms,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=vit_transforms,
        vit_image_size=cfg.vit_image_size,
        misc_size=cfg.misc_size,
	mvs_visual=cfg.mvs_visual,
        tokenizer=tokenizer,
    )

    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["video_data", "cap_tokens", "feature_framerate", "vit_image", "mv_data", "single_image", "mask", "misc_data"],
    )
    dl = dataloader.batch(cfg.batch_size)
    '''
    
    dl = build_dataset(cfg, 1, 0, tokenizer)
    dl.get_dataset_size()

    ms.set_context(mode=0)

    num_tries = 4*100
    start = time.time()
    times = []
    iterator = dl.create_dict_iterator()
    for i, batch in enumerate(iterator):
        for k in batch:
            print(k, batch[k].shape, batch[k].min(), batch[k].max())
            if k in ['cap_tokens', 'feature_framerate']:
                print(batch[k])
        times.append(time.time() - start)
        if i >= num_tries:
            break
        start = time.time()

    WU = 2
    tot = sum(times[WU:])  # skip warmup
    mean = tot / (num_tries - WU)
    print("Avg batch loading time: ", mean)


if __name__ == "__main__":
    test_dataset()



