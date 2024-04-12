import glob
import logging
import os
import os.path as osp

from mindspore.mindrecord import FileWriter

logger = logging.getLogger()


class MindRecordEmbeddingCacheWriter:
    def __init__(
        self,
        cache_folder,
        dataset_name,
        schema,
        start_video_index,
        overwrite=False,
        max_page_size=128,
        dump_every_n_lines=10,
    ):
        self.cache_folder = cache_folder
        self.dataset_name = dataset_name
        assert osp.exists(cache_folder)
        self.start_video_index = start_video_index

        if not overwrite:
            existing_files = glob.glob(osp.join(cache_folder, dataset_name + "*.mindrecord"))
            logger.info(
                f"Found existing {len(existing_files)} mindrecord files. \nStart saving embedding cache to {len(existing_files)+1} mindrecord file."
            )
            self.start_file_index = len(existing_files)
        else:
            self.start_file_index = 0
        self.current_file_index = self.start_file_index

        self.num_saved_files = 0
        self.num_saved_videos = 0
        self.max_page_size = max_page_size
        self.schema = schema

        # initiate writer
        self.initiate_writer()
        self.num_saved_files += 1
        self.dump_every_n_lines = dump_every_n_lines

    @property
    def current_file_path(self):
        filepath = os.path.join(self.cache_folder, self.dataset_name + str(self.current_file_index) + ".mindrecord")
        return filepath

    def initiate_writer(self):
        self.writer = FileWriter(self.current_file_path, shard_num=1, overwrite=True)
        if self.max_page_size == 64:
            self.writer.set_page_size(1 << 26)
        elif self.max_page_size == 128:
            self.writer.set_page_size(1 << 27)
        elif self.max_page_size == 256:
            self.writer.set_page_size(1 << 28)
        else:
            raise ValueError("Not supported max page size!")
        self.writer.add_schema(self.schema, "Preprocessed {} dataset.".format(self.dataset_name))

    def handle_saving_exception(self, e):
        self.writer.commit()
        self.get_status()
        raise e  # end the process

    def get_status(self):
        logger.info(
            "MindRecord embedding cache writer status:\n"
            f"Start Video Index: {self.start_video_index}.\n"
            f"Number of saved mindrecord files {self.num_saved_files}\n"
            f"Number of saved videos {self.num_saved_videos}\n"
        )

    def save_data_and_close_writer(self, data):
        if data:
            try:
                self.writer.write_raw_data(data)
                self.num_saved_videos += len(data)
            except Exception as e:
                self.handle_saving_exception(e)
            data = []
        self.writer.commit()
        return data

    def save(self, data):
        # measure the current file size before saving
        # if it exceeds 19GB, close the current file writer and initiate another file writer
        # if it does not exceeds, dump raw data if number of lines >=dump_every_n_lines
        if os.path.isfile(self.current_file_path):
            mindrecord_size = os.stat(self.current_file_path).st_size
            mindrecord_size = mindrecord_size / 1024 / 1024 / 1024
            if mindrecord_size > 19:
                # close last filewriter when it exceeds 19GB
                data = self.save_data_and_close_writer(data)
                self.current_file_index += 1
                self.initiate_writer()
                self.num_saved_files += 1

        if len(data) >= self.dump_every_n_lines or not os.path.isfile(self.current_file_path):
            try:
                self.writer.write_raw_data(data)
                self.num_saved_videos += len(data)
            except Exception as e:
                self.handle_saving_exception(e)
            data = []
        return data
