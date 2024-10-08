import cv2
import numpy as np
import trimesh
from PIL import Image


def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3,
        faces=facenp_fx3,
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, "obj")


def save_glb(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    pointnp_px3 = pointnp_px3 @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3,
        faces=facenp_fx3,
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, "glb")


def save_obj_with_mtl(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, texmap_hxwx3, fname):
    import os

    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = "%s/%s.mtl" % (fol, na)
    fid = open(matname, "w")
    fid.write("newmtl material_0\n")
    fid.write("Kd 1 1 1\n")
    fid.write("Ka 0 0 0\n")
    fid.write("Ks 0.4 0.4 0.4\n")
    fid.write("Ns 10\n")
    fid.write("illum 2\n")
    fid.write("map_Kd %s.png\n" % na)
    fid.close()
    ####

    fid = open(fname, "w")
    fid.write("mtllib %s.mtl\n" % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write("v %f %f %f\n" % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write("vt %f %f\n" % (pp[0], pp[1]))

    fid.write("usemtl material_0\n")
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1
        f2 = facetex_fx3[i] + 1
        fid.write("f %d/%d %d/%d %d/%d\n" % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    fid.close()

    # save texture map
    lo, hi = 0, 1
    img = np.asarray(texmap_hxwx3, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(np.float32), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(np.float32)
    kernel = np.ones((3, 3), "uint8")
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), "RGB").save(f"{fol}/{na}.png")


def loadobj(meshfile):
    v = []
    f = []
    meshfp = open(meshfile, "r")
    for line in meshfp.readlines():
        data = line.strip().split(" ")
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            continue
        if data[0] == "v":
            v.append([float(d) for d in data[1:]])
        if data[0] == "f":
            data = [da.split("/")[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3


def loadobjtex(meshfile):
    v = []
    vt = []
    f = []
    ft = []
    meshfp = open(meshfile, "r")
    for line in meshfp.readlines():
        data = line.strip().split(" ")
        data = [da for da in data if len(da) > 0]
        if not ((len(data) == 3) or (len(data) == 4) or (len(data) == 5)):
            continue
        if data[0] == "v":
            assert len(data) == 4

            v.append([float(d) for d in data[1:]])
        if data[0] == "vt":
            if len(data) == 3 or len(data) == 4:
                vt.append([float(d) for d in data[1:3]])
        if data[0] == "f":
            data = [da.split("/") for da in data]
            if len(data) == 4:
                f.append([int(d[0]) for d in data[1:]])
                ft.append([int(d[1]) for d in data[1:]])
            elif len(data) == 5:
                idx1 = [1, 2, 3]
                data1 = [data[i] for i in idx1]
                f.append([int(d[0]) for d in data1])
                ft.append([int(d[1]) for d in data1])
                idx2 = [1, 3, 4]
                data2 = [data[i] for i in idx2]
                f.append([int(d[0]) for d in data2])
                ft.append([int(d[1]) for d in data2])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    ftnp_fx3 = np.array(ft, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    uvs = np.array(vt, dtype=np.float32)
    return pointnp_px3, facenp_fx3, uvs, ftnp_fx3
