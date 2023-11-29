import os
import random
from tqdm import tqdm
import scipy.io as scio
import numpy as np
import mindspore as ms
import utils.audio as audio


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode="constant", constant_values=0)
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames, 1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id + start + 9 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 9, 0] = [
                0.5,
                0.6,
                0.7,
                0.9,
                1,
                0.9,
                0.7,
                0.6,
                0.5,
            ]
            frame_id = frame_id + start + 9
        else:
            break
    return ratio


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10, num_frames), min(int(num_frames / 2), 70)))
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 5, 0] = [
                0.5,
                0.9,
                1.0,
                0.9,
                0.5,
            ]
            frame_id = frame_id + start + 5
        else:
            break
    return ratio


class TestDataset:
    def __init__(
        self,
        args,
        preprocessor,
        save_dir,
        syncnet_mel_step_size=16,
        fps=25,
        idlemode=False,
        length_of_audio=False,
        use_blink=True,
    ):
        self.args = args
        self.preprocessor = preprocessor
        self.save_dir = save_dir
        self.first_frame_dir = os.path.join(self.save_dir, "first_frame_dir")

        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.fps = fps
        self.still = self.args.still
        self.idlemode = idlemode
        self.length_of_audio = length_of_audio
        self.use_blink = use_blink

    def crop_and_extract(self):
        os.makedirs(self.first_frame_dir, exist_ok=True)
        print("3DMM Extraction for source image")

        first_coeff_path, crop_pic_path, crop_info = self.preprocessor.generate(
            self.args.source_image,
            self.first_frame_dir,
            self.args.preprocess,
            source_image_flag=True,
            pic_size=self.args.size,
        )

        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if self.args.ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(self.args.ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(self.save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing eye blinking")
            ref_eyeblink_coeff_path, _, _ = self.preprocessor.generate(
                self.args.ref_eyeblink,
                ref_eyeblink_frame_dir,
                self.args.preprocess,
                source_image_flag=False,
            )
        else:
            ref_eyeblink_coeff_path = None

        if self.args.ref_pose is not None:
            if self.args.ref_pose == self.args.ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(self.args.ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(self.save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print("3DMM Extraction for the reference video providing pose")
                ref_pose_coeff_path, _, _ = self.preprocessor.generate(
                    self.args.ref_pose,
                    ref_pose_frame_dir,
                    self.args.preprocess,
                    source_image_flag=False,
                )
        else:
            ref_pose_coeff_path = None

        return (
            first_coeff_path,
            crop_pic_path,
            crop_info,
            ref_eyeblink_coeff_path,
            ref_pose_coeff_path,
        )

    def __next__(self):
        if self._index >= 1:
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # 1. crop and extract 3dMM coefficients
        (
            first_coeff_path,
            crop_pic_path,
            crop_info,
            ref_eyeblink_coeff_path,
            ref_pose_coeff_path,
        ) = self.crop_and_extract()

        # 2. process audio
        pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
        audio_name = os.path.splitext(os.path.split(self.args.driven_audio)[-1])[0]

        if self.idlemode:
            num_frames = int(self.length_of_audio * self.fps)
            indiv_mels = np.zeros((num_frames, num_frames, self.syncnet_mel_step_size))
        else:
            wav = audio.load_wav(self.args.driven_audio, 16000)
            wav_length, num_frames = parse_audio_length(len(wav), 16000, self.fps)
            wav = crop_pad_audio(wav, wav_length)
            orig_mel = audio.melspectrogram(wav).T
            spec = orig_mel.copy()  # nframes 80
            indiv_mels = []

            for i in tqdm(range(num_frames), "mel:"):
                start_frame_num = i - 2
                start_idx = int(80.0 * (start_frame_num / float(self.fps)))
                end_idx = start_idx + self.syncnet_mel_step_size
                seq = list(range(start_idx, end_idx))
                seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
                m = spec[seq, :]
                indiv_mels.append(m.T)
            indiv_mels = np.asarray(indiv_mels)  # T 80 16

        indiv_mels = ms.Tensor(indiv_mels, ms.float32).unsqueeze(1).unsqueeze(0)  # bs T 1 80 16

        # 3. generate ref coeffs

        ratio = generate_blink_seq_randomly(num_frames)  # T
        source_semantics_path = first_coeff_path
        source_semantics_dict = scio.loadmat(source_semantics_path)
        ref_coeff = source_semantics_dict["coeff_3dmm"][:1, :70]  # 1 70
        ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

        if ref_eyeblink_coeff_path is not None:
            ratio[:num_frames] = 0
            refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
            refeyeblink_coeff = refeyeblink_coeff_dict["coeff_3dmm"][:, :64]
            refeyeblink_num_frames = refeyeblink_coeff.shape[0]
            if refeyeblink_num_frames < num_frames:
                div = num_frames // refeyeblink_num_frames
                re = num_frames % refeyeblink_num_frames
                refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
                refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
                refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
                print(refeyeblink_coeff.shape[0])

            ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64]

        if self.use_blink:
            ratio = ms.Tensor(ratio, ms.float32).unsqueeze(0)  # bs T
        else:
            ratio = ms.Tensor(ratio, ms.float32).unsqueeze(0).fill_(0.0)
            # bs T

        ref_coeff = np.asarray(ref_coeff).astype("float32")
        ref_coeff = ms.Tensor(ref_coeff).unsqueeze(0)  # bs 1 70

        return {
            "indiv_mels": indiv_mels,
            "ref": ref_coeff,
            "num_frames": num_frames,
            "ratio_gt": ratio,
            "audio_name": audio_name,
            "pic_name": pic_name,
            "ref_pose_coeff_path": ref_pose_coeff_path,
            "first_coeff_path": first_coeff_path,
            "crop_pic_path": crop_pic_path,
            "crop_info": crop_info,
        }
