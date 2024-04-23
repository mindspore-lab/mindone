from sklearn.metrics.pairwise import cosine_similarity

from mindspore import Tensor, ops

VIDEO_EXTENSIONS = {".mp4"}


class ClipScoreText:
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def score(self, frames, prompt):
        inputs = self.processor(text=[prompt], images=frames)
        inputs = {k: Tensor(v) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        score = outputs[0].asnumpy().mean()

        return score


class ClipScoreFrame:
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.fill_diagonal = ops.FillDiagonal(0.0)

    def score(self, frames):
        inputs = self.processor(images=frames)
        inputs = {k: Tensor(v) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs).asnumpy()
        cosine_sim_matrix = cosine_similarity(image_features)
        cosine_sim_matrix = self.fill_diagonal(Tensor(cosine_sim_matrix))  # set diagonal elements to 0
        score = cosine_sim_matrix.sum() / (len(frames) * (len(frames) - 1))
        return score
