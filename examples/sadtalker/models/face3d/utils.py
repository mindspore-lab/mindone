import warnings
from PIL import Image
from scipy.io import loadmat, savemat
import os.path as osp
import numpy as np
from array import array


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# load expression basis
def LoadExpBasis(bfm_folder="BFM"):
    n_vertex = 53215
    Expbin = open(osp.join(bfm_folder, "Exp_Pca.bin"), "rb")
    exp_dim = array("i")
    exp_dim.fromfile(Expbin, 1)
    expMU = array("f")
    expPC = array("f")
    expMU.fromfile(Expbin, 3 * n_vertex)
    expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(osp.join(bfm_folder, "std_exp.txt"))

    return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(bfm_folder="BFM"):
    print("Transfer BFM09 to BFM_model_front......")
    original_BFM = loadmat(osp.join(bfm_folder, "01_MorphableModel.mat"))
    shapePC = original_BFM["shapePC"]  # shape basis
    shapeEV = original_BFM["shapeEV"]  # corresponding eigen value
    shapeMU = original_BFM["shapeMU"]  # mean face
    texPC = original_BFM["texPC"]  # texture basis
    texEV = original_BFM["texEV"]  # eigen value
    texMU = original_BFM["texMU"]  # mean texture

    expPC, expEV = LoadExpBasis(bfm_folder)

    # transfer BFM09 to our face model

    idBase = shapePC * np.reshape(shapeEV, [-1, 199])
    idBase = idBase / 1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC * np.reshape(expEV, [-1, 79])
    exBase = exBase / 1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC * np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat(osp.join(bfm_folder, "BFM_front_idx.mat"))
    index_exp = index_exp["idx"].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat(osp.join(bfm_folder, "BFM_exp_idx.mat"))
    index_shape = index_shape["trimIndex"].astype(np.int32) - 1  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.

    other_info = loadmat(osp.join(bfm_folder, "facemodel_info.mat"))
    frontmask2_idx = other_info["frontmask2_idx"]
    skinmask = other_info["skinmask"]
    keypoints = other_info["keypoints"]
    point_buf = other_info["point_buf"]
    tri = other_info["tri"]
    tri_mask2 = other_info["tri_mask2"]

    # save our face model
    savemat(
        osp.join(bfm_folder, "BFM_model_front.mat"),
        {
            "meanshape": meanshape,
            "meantex": meantex,
            "idBase": idBase,
            "exBase": exBase,
            "texBase": texBase,
            "tri": tri,
            "point_buf": point_buf,
            "tri_mask2": tri_mask2,
            "keypoints": keypoints,
            "frontmask2_idx": frontmask2_idx,
            "skinmask": skinmask,
        },
    )


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):
    Lm3D = loadmat(osp.join(bfm_folder, "similarity_Lm3D_all.mat"))
    Lm3D = Lm3D["lm"]

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack(
        [
            Lm3D[lm_idx[0], :],
            np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
            np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
            Lm3D[lm_idx[5], :],
            Lm3D[lm_idx[6], :],
        ],
        axis=0,
    )
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0 : 2 * npts - 1 : 2, 0:3] = x.transpose()
    A[0 : 2 * npts - 1 : 2, 3] = 1

    A[1 : 2 * npts : 2, 4:7] = x.transpose()
    A[1 : 2 * npts : 2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


# resize and crop images for face reconstruction


def resize_n_crop_img(img, lm, t, s, target_size=224.0, mask=None):
    w0, h0 = img.size
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    return img, lm, mask


# utils for face reconstruction


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack(
        [
            lm[lm_idx[0], :],
            np.mean(lm[lm_idx[[1, 2]], :], 0),
            np.mean(lm[lm_idx[[3, 4]], :], 0),
            lm[lm_idx[5], :],
            lm[lm_idx[6], :],
        ],
        axis=0,
    )
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


# utils for face reconstruction


def align_img(img, lm, lm3D, mask=None, target_size=224.0, rescale_factor=102.0):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor / s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new
