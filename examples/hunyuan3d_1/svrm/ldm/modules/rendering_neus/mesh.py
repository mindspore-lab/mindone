from __future__ import annotations

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, ops

# from ...utils.typing import *
from ...utils.typing import Any, Dict, Float, Integer, Optional, Union


def dot(x, y):
    return mint.sum(x * y, -1, keepdim=True)


class Mesh:
    def __init__(self, v_pos: Tensor, t_pos_idx: Tensor, v_rgb: Tensor, **kwargs) -> None:
        # v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], v_rgb: Integer[Tensor, "Nf 3"]

        self.v_pos: Tensor = v_pos
        self.t_pos_idx: Tensor = t_pos_idx
        self.v_rgb: Optional[Float, Tensor] = v_rgb  # Float[Tensor, "Nv 3"]

        self._v_nrm: Optional[Float, Tensor] = None  # Float[Tensor, "Nv 3"]
        self._v_tng: Optional[Float, Tensor] = None  # Float[Tensor, "Nv 3"]
        self._v_tex: Optional[Float, Tensor] = None  # Float[Tensor, "Nt 3"]
        self._t_tex_idx: Optional[Float, Tensor] = None  # Float[Tensor, "Nf 3"]
        # self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer, Tensor] = None  # Integer[Tensor, "Ne 2"]
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]) -> Mesh:
        if self.requires_grad:
            print("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.asnumpy(),
            faces=self.t_pos_idx.asnumpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        print("Mesh has {} components, with faces: {}".format(len(components), [c.faces.shape[0] for c in components]))

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold)
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        print("Removing components with less than {} faces".format(n_faces_threshold))

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        print(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = ms.Tensor(mesh.vertices).to(self.v_pos.dtype)
        t_pos_idx = ms.Tensor(mesh.faces).to(self.t_pos_idx.dtype)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            print(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self):
        if self._t_tex_idx is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._t_tex_idx

    # @property
    # def v_rgb(self):
    #     return self._v_rgb

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = ops.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = mint.zeros_like(self.v_pos)
        v_nrm = mint.scatter_add(v_nrm, 0, i0[:, None].tile((1, 3)), face_normals)
        v_nrm = mint.scatter_add(v_nrm, 0, i1[:, None].tile((1, 3)), face_normals)
        v_nrm = mint.scatter_add(v_nrm, 0, i2[:, None].tile((1, 3)), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = mint.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, Tensor([0.0, 0.0, 1.0]).to(v_nrm.dtype))
        v_nrm = v_nrm / ops.norm(v_nrm, dim=1)  # F.normalize(v_nrm, dim=1) #TODO: test accuracy

        assert mint.all(mint.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = mint.zeros_like(self.v_nrm)
        tansum = mint.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / mint.where(denom > 0.0, mint.clamp(denom, min=1e-6), mint.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].tile((1, 3))
            tangents = mint.scatter_add(tangents, 0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum = mint.scatter_add(tansum, 0, idx, mint.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = tangents / ops.norm(tangents, dim=1)
        tangents = tangents - dot(tangents, self.v_nrm) * self.v_nrm
        tangents = tangents / ops.norm(tangents)
        # tangents = F.normalize(tangents, dim=1)
        # tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        assert mint.all(mint.isfinite(tangents))

        return tangents

    def _unwrap_uv(self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}):
        print("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(
            self.v_pos.asnumpy(),
            self.t_pos_idx.asnumpy(),
        )
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        atlas.generate(co, po)
        vmapping, indices, uvs = atlas.get_mesh(0)
        vmapping = ms.Tensor(vmapping.astype(np.uint64, casting="same_kind").view(np.int64)).long()
        uvs = ms.Tensor(uvs).float()
        indices = ms.Tensor(indices.astype(np.uint64, casting="same_kind").view(np.int64)).long()
        return uvs, indices

    def unwrap_uv(self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}):
        self._v_tex, self._t_tex_idx = self._unwrap_uv(xatlas_chart_options, xatlas_pack_options)

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb

    def _compute_edges(self):
        # Compute edges
        edges = mint.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = mint.unique(edges, dim=0)
        return edges

    def normal_consistency(self) -> Tensor:
        edge_nrm = self.v_nrm[self.edges]  # Float[Tensor, "Ne 2 3"]
        nc = (1.0 - ops.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)).mean()
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]
        # F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten(start_dim=0)
        jj = faces[:, [2, 0, 1]].flatten(start_dim=0)
        adj = mint.unique(mint.stack([mint.cat([ii, jj]), mint.cat([jj, ii])], dim=0), dim=1)
        adj_values = mint.ones(adj.shape[1]).to(verts.dtype)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = mint.cat((adj, mint.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = mint.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return ms.COOTensor(idx, values, (V, V)).coalesce()

    def laplacian(self) -> Tensor:
        L = ops.stop_gradient(self._laplacian_uniform())
        loss = L.to_csr.mm(self.v_pos)  # COOTensor has no mm operator
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss
