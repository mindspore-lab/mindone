import os, sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from lvdm.data.dataset import create_dataloader

data_dir = "/root/lhy/data/data/mixkit-100videos/mixkit"
# csv_path = "/root/lhy/data/data/mixkit-100videos/mixkit/video_caption_train.csv"
csv_path = "/root/lhy/data/data/mixkit-100videos/mixkit/video_caption_test.csv"

device_num = 1
rank_id = 0
epochs = 2

data_config = dict(
        csv_path=csv_path,
        data_dir=data_dir,
        # column_names=['video', 'caption'],
        column_names=['video', 'caption', 'path', 'fps', 'frame_stride'],
        # subsample=None,
        batch_size=1,
        video_length=16,
        resolution=[576, 1024],
        # resolution=None,
        frame_stride=6,
        # frame_stride_min=1,
        spatial_transform="resize_center_crop",
        # spatial_transform=None,
        # crop_resolution=None,
        # fps_max=None,
        load_raw_resolution=True,
        # fixed_fps=None,
        random_fs=True,
        shuffle=True,
        num_parallel_workers=10,
        max_rowsize=64,
    )

# tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenize
dataset = create_dataloader(data_config, device_num=device_num, rank_id=rank_id)
dataset_size = dataset.get_dataset_size()
ds_iter = dataset.create_tuple_iterator(num_epochs=epochs)
# ds_iter = dataset.create_dict_iterator(num_epochs=epochs)
for i, data in enumerate(ds_iter):
    print("---: ", i, len(data), data[0].shape, type(data[0]), data[0].dtype)