from gm.models.trainer_factory import TrainOneStepCell as TrainOneStepCellOriginal


class TrainOneStepCell(TrainOneStepCellOriginal):
    def construct(self, x, *tokens):
        num_frames = x.shape[1]
        cond_frames_without_noise, fps_id, motion_bucket_id, cond_frames, cond_aug = tokens
        # merge batch dimension with num_frames
        x = x.reshape(-1, *x.shape[2:])
        fps_id = fps_id.reshape(-1, *fps_id.shape[2:])
        motion_bucket_id = motion_bucket_id.reshape(-1, *motion_bucket_id.shape[2:])
        cond_aug = cond_aug.reshape(-1, *cond_aug.shape[2:])

        tokens = (cond_frames_without_noise, fps_id, motion_bucket_id, cond_frames, cond_aug)

        return super().construct(x, *tokens, num_video_frames=num_frames)
