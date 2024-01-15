import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor
from mindspore import dataset as ds
from mindspore import ops
from mindspore.dataset import vision

from .inception_v3 import inception_v3_fid


class ImagePathDataset:
    """Image files dataload."""

    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        # if self.transforms is not None:
        #    img = self.transforms(img)
        # return (img,)
        img = vision.ToTensor()(img)

        # TODO: use other numpy resize ops, used in torchmetrics
        img = ms.Tensor(img).expand_dims(0)
        img = ops.ResizeBilinearV2()(img, (299, 299))  # it gives smaller error than vision.Resize or ops.interpolate
        img = img.squeeze()

        return img


def get_activations(files, model, batch_size=64, dims=2048):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if batch_size > len(files):
        print(("Warning: batch size is bigger than the data size. " "Setting batch size to data size"))
        batch_size = len(files)
    # dataset = ImagePathDataset(files, transforms=vision.ToTensor())
    dataset = ImagePathDataset(files)
    dataloader = ds.GeneratorDataset(dataset, ["image"], shuffle=False)
    """
    transforms = [vision.Resize(size=(299, 299), interpolation=vision.Inter.BILINEAR),
                  vision.ToTensor(),
                  ]
    dataloader = dataloader.map(operations=transforms, input_columns="image")
    """

    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    with tqdm(total=dataloader.get_dataset_size()) as p_bar:
        for batch in dataloader:
            batch = Tensor(batch[0])
            pred = model(batch).asnumpy()
            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]
            p_bar.update(1)
    return pred_arr


class FrechetInceptionDistance:
    def __init__(self, ckpt_path=None, batch_size=64):
        # TODO: set context
        if ckpt_path is not None:
            self.model = inception_v3_fid(pretrained=False, ckpt_path=ckpt_path)
        else:
            self.model = inception_v3_fid(pretrained=True)

        self.model.set_train(False)
        self.batch_size = batch_size

    def calculate_activation_stat(self, act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        eigenvalues, eigenvectors = np.linalg.eig(sigma1.dot(sigma2))
        sqrt_diagonal_matrix = np.diag(np.sqrt(eigenvalues))
        covmean = np.dot(np.dot(eigenvectors, sqrt_diagonal_matrix), np.linalg.inv(eigenvectors))
        if not np.isfinite(covmean).all():
            msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            eigenvalues, eigenvectors = np.linalg.eig((sigma1 + offset).dot(sigma2 + offset))
            sqrt_diagonal_matrix = np.diag(np.sqrt(eigenvalues))
            covmean = np.dot(np.dot(eigenvectors, sqrt_diagonal_matrix), np.linalg.inv(eigenvectors))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute(self, gen_images, gt_images):
        """
        gen_images: list of generated image file paths
        gt_images: list of GT image file paths
        """
        gen_images = sorted(gen_images)
        gt_images = sorted(gt_images)

        gen_feats = get_activations(gen_images, self.model, self.batch_size)
        gen_mu, gen_sigma = self.calculate_activation_stat(gen_feats)

        gt_feats = get_activations(gt_images, self.model, self.batch_size)
        gt_mu, gt_sigma = self.calculate_activation_stat(gt_feats)

        fid_value = self.calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)

        return fid_value

    def update(images, real=True):
        pass

    def reset(
        self,
    ):
        pass


if __name__ == "__main__":
    gen_imgs = ["data/img_1.jpg", "data/img_2.jpg"]
    gt_imgs = [
        "data/img_10.jpg",
        "data/img_11.jpg",
    ]

    # fid_scorer = FrechetInceptionDistance("./inception_v3_fid.ckpt")
    fid_scorer = FrechetInceptionDistance()
    score = fid_scorer.compute(gen_imgs, gt_imgs)
    print("ms FID: ", score)
