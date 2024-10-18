import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import mint, ops

from .tables import check_table, dmc_table, num_vd_table, tet_table

__all__ = ["FlexiCubes"]


class FlexiCubes(nn.Cell):
    """
    This class implements the FlexiCubes method for extracting meshes from scalar fields.
    It maintains a series of lookup tables and indices to support the mesh extraction process.
    FlexiCubes, a differentiable variant of the Dual Marching Cubes (DMC) scheme, enhances
    the geometric fidelity and mesh quality of reconstructed meshes by dynamically adjusting
    the surface representation through gradient-based optimization.

    During instantiation, the class loads DMC tables from a file and transforms them into
    PyTorch tensors on the specified device.

    Attributes:
        dmc_table (ms.Tensor): Dual Marching Cubes (DMC) table that encodes the edges
            associated with each dual vertex in 256 Marching Cubes (MC) configurations.
        num_vd_table (ms.Tensor): Table holding the number of dual vertices in each of
            the 256 MC configurations.
        check_table (ms.Tensor): Table resolving ambiguity in cases C16 and C19
            of the DMC configurations.
        tet_table (ms.Tensor): Lookup table used in tetrahedralizing the isosurface.
        quad_split_1 (ms.Tensor): Indices for splitting a quad into two triangles
            along one diagonal.
        quad_split_2 (ms.Tensor): Alternative indices for splitting a quad into
            two triangles along the other diagonal.
        quad_split_train (ms.Tensor): Indices for splitting a quad into four triangles
            during training by connecting all edges to their midpoints.
        cube_corners_idx (ms.Tensor): Cube corners indexed as powers of 2, used
            to retrieve the case id.
        edge_dir_table (ms.Tensor): A mapping tensor that associates edge indices with
            their corresponding axis. For instance, edge_dir_table[0] = 0 indicates that the
            first edge is oriented along the x-axis.
        dir_faces_table (ms.Tensor): A tensor that maps the corresponding axis of shared edges
            across four adjacent cubes to the shared faces of these cubes. For instance,
            dir_faces_table[0] = [5, 4] implies that for four cubes sharing an edge along
            the x-axis, the first and second cubes share faces indexed as 5 and 4, respectively.
            This tensor is only utilized during isosurface tetrahedralization.
        adj_pairs (ms.Tensor):
            A tensor containing index pairs that correspond to neighboring cubes that share the same edge.
        qef_reg_scale (float):
            The scaling factor applied to the regularization loss to prevent issues with singularity
            when solving the QEF. This parameter is only used when a 'grad_func' is specified.
        weight_scale (float):
            The scale of weights in FlexiCubes. Should be between 0 and 1.
        cube_edges (np.array): Edge connections in a cube, listed in pairs.
            Used to retrieve edge vertices in DMC.
        cube_corners (np.array): Defines the positions of a standard unit cube's
            eight corners in 3D space, ordered starting from the origin (0,0,0),
            moving along the x-axis, then y-axis, and finally z-axis.
            Used as a blueprint for generating a voxel grid.
    """

    def __init__(self, qef_reg_scale=1e-3, weight_scale=0.99):
        super().__init__()
        self.dmc_table = ms.Tensor(dmc_table, dtype=ms.int64)
        self.num_vd_table = ms.Tensor(num_vd_table, dtype=ms.int64)
        self.check_table = ms.Tensor(check_table, dtype=ms.int64)
        self.tet_table = ms.Tensor(tet_table, dtype=ms.int64)
        self.quad_split_1 = ms.Tensor([0, 1, 2, 0, 2, 3], dtype=ms.int64)
        self.quad_split_2 = ms.Tensor([0, 1, 3, 3, 1, 2], dtype=ms.int64)
        self.quad_split_train = ms.Tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=ms.int64)
        self.cube_corners_idx = ops.pow(2, ops.arange(8))
        self.cube_edges = ms.Tensor(
            [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4], dtype=ms.int8
        )
        self.edge_dir_table = ms.Tensor([0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1], dtype=ms.int64)
        self.dir_faces_table = ms.Tensor(
            [[[5, 4], [3, 2], [4, 5], [2, 3]], [[5, 4], [1, 0], [4, 5], [0, 1]], [[3, 2], [1, 0], [2, 3], [0, 1]]],
            dtype=ms.int64,
        )
        self.adj_pairs = ms.Tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=ms.int64)
        self.qef_reg_scale = qef_reg_scale
        self.weight_scale = weight_scale

        # non-ms vars: these vars need to be used in init but not constrcut, thus cannot in ms
        self.cube_corners = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        )
        # self.cube_edges = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4]

    # an init function, needs to be done with np
    def np_construct_voxel_grid(self, res):
        """
        Generates a voxel grid based on the specified resolution.

        Args:
            res (int or list[int]): The resolution of the voxel grid. If an integer
                is provided, it is used for all three dimensions. If a list or tuple
                of 3 integers is provided, they define the resolution for the x,
                y, and z dimensions respectively.

        Returns:
            (ms.Tensor, ms.Tensor): Returns the vertices and the indices of the
                cube corners (index into vertices) of the constructed voxel grid.
                The vertices are centered at the origin, with the length of each
                dimension in the grid being one.
        """
        base_cube_f = np.arange(8)
        if isinstance(res, int):
            res = (res, res, res)
        voxel_grid_template = np.ones(res)

        # res = ms.Tensor([res], dtype=ms.float32)
        coords = np.stack([i for i in np.nonzero(voxel_grid_template)], axis=1) / res  # N, 3
        verts = (self.cube_corners[None] / res + coords[:, None]).reshape(-1, 3)
        cubes = (base_cube_f[None] + np.arange(coords.shape[0])[:, None] * 8).reshape(-1)

        verts_rounded = np.round(verts * 10**5) / (10**5)
        # ms implementation, tbr
        # unique_func = ops.UniqueConsecutive(axis=0, return_idx=True, return_counts=False)
        # verts_unique, inverse_indices, _ = unique_func(verts_rounded)
        verts_unique, inverse_indices = np.unique(verts_rounded, axis=0, return_inverse=True)
        # verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
        cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)

        return verts_unique - 0.5, cubes  # wrapped coz used in flexicubes_geo .init()

    def __call__(
        self,
        x_nx3: ms.Tensor,
        s_n: ms.Tensor,
        cube_fx8: ms.Tensor,
        res,
        beta_fx12: ms.Tensor = None,
        alpha_fx8: ms.Tensor = None,
        gamma_f: ms.Tensor = None,
        training=False,
        output_tetmesh=False,
        grad_func=None,
    ):
        r"""
        Main function for mesh extraction from scalar field using FlexiCubes. This function converts
        discrete signed distance fields, encoded on voxel grids and additional per-cube parameters,
        to triangle or tetrahedral meshes using a differentiable operation as described in
        `Flexible Isosurface Extraction for Gradient-Based Mesh Optimization`_. FlexiCubes enhances
        mesh quality and geometric fidelity by adjusting the surface representation based on gradient
        optimization. The output surface is differentiable with respect to the input vertex positions,
        scalar field values, and weight parameters.

        If you intend to extract a surface mesh from a fixed Signed Distance Field without the
        optimization of parameters, it is suggested to provide the "grad_func" which should
        return the surface gradient at any given 3D position. When grad_func is provided, the process
        to determine the dual vertex position adapts to solve a Quadratic Error Function (QEF), as
        described in the `Manifold Dual Contouring`_ paper, and employs an smart splitting strategy.
        Please note, this approach is non-differentiable.

        For more details and example usage in optimization, refer to the
        `Flexible Isosurface Extraction for Gradient-Based Mesh Optimization`_ SIGGRAPH 2023 paper.

        Args:
            x_nx3 (ms.Tensor): Coordinates of the voxel grid vertices, can be deformed.
            s_n (ms.Tensor): Scalar field values at each vertex of the voxel grid. Negative values
                denote that the corresponding vertex resides inside the isosurface. This affects
                the directions of the extracted triangle faces and volume to be tetrahedralized.
            cube_fx8 (ms.Tensor): Indices of 8 vertices for each cube in the voxel grid.
            res (int or list[int]): The resolution of the voxel grid. If an integer is provided, it
                is used for all three dimensions. If a list or tuple of 3 integers is provided, they
                specify the resolution for the x, y, and z dimensions respectively.
            beta_fx12 (ms.Tensor, optional): Weight parameters for the cube edges to adjust dual
                vertices positioning. Defaults to uniform value for all edges.
            alpha_fx8 (ms.Tensor, optional): Weight parameters for the cube corners to adjust dual
                vertices positioning. Defaults to uniform value for all vertices.
            gamma_f (ms.Tensor, optional): Weight parameters to control the splitting of
                quadrilaterals into triangles. Defaults to uniform value for all cubes.
            training (bool, optional): If set to True, applies differentiable quad splitting for
                training. Defaults to False.
            output_tetmesh (bool, optional): If set to True, outputs a tetrahedral mesh, otherwise,
                outputs a triangular mesh. Defaults to False.
            grad_func (callable, optional): A function to compute the surface gradient at specified
                3D positions (input: Nx3 positions). The function should return gradients as an Nx3
                tensor. If None, the original FlexiCubes algorithm is utilized. Defaults to None.

        Returns:
            (ms.Tensor, ms.float64Tensor, ms.Tensor): Tuple containing:
                - Vertices for the extracted triangular/tetrahedral mesh.
                - Faces for the extracted triangular/tetrahedral mesh.
                - Regularizer L_dev, computed per dual vertex.

        .. _Flexible Isosurface Extraction for Gradient-Based Mesh Optimization:
            https://research.nvidia.com/labs/toronto-ai/flexicubes/
        .. _Manifold Dual Contouring:
            https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf
        """

        surf_cubes, occ_fx8 = self._identify_surf_cubes(s_n, cube_fx8)
        if surf_cubes.sum() == 0:
            return (
                ops.zeros((0, 3)),
                ops.zeros((0, 4), dtype=ms.int64) if output_tetmesh else ops.zeros((0, 3), dtype=ms.int64),
                ops.zeros((0)),
            )
        else:
            surf_cubes = (
                surf_cubes.bool()
            )  # in order to do tensor sum in ms, this masking tensor is casted into uint8, now cast back to bool for masking
        beta_fx12, alpha_fx8, gamma_f = self._normalize_weights(beta_fx12, alpha_fx8, gamma_f, surf_cubes)

        case_ids = self._get_case_id(occ_fx8, surf_cubes, res)

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(s_n, cube_fx8, surf_cubes)

        vd, L_dev, vd_gamma, vd_idx_map = self._compute_vd(
            x_nx3, cube_fx8[surf_cubes], surf_edges, s_n, case_ids, beta_fx12, alpha_fx8, gamma_f, idx_map, grad_func
        )
        vertices, faces, s_edges, edge_indices = self._triangulate(
            s_n, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, training, grad_func
        )
        if not output_tetmesh:
            return vertices, faces, L_dev
        else:
            vertices, tets = self._tetrahedralize(
                x_nx3,
                s_n,
                cube_fx8,
                vertices,
                faces,
                surf_edges,
                s_edges,
                vd_idx_map,
                case_ids,
                edge_indices,
                surf_cubes,
                training,
            )
            return vertices, tets, L_dev

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """
        dist = ops.norm(ue - mint.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)
        mean_l2 = ops.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - mint.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad

    def _normalize_weights(self, beta_fx12, alpha_fx8, gamma_f, surf_cubes):
        """
        Normalizes the given weights to be non-negative. If input weights are None, it creates and returns a set of weights of ones.
        """
        n_cubes = surf_cubes.shape[0]

        if beta_fx12 is not None:
            beta_fx12 = ops.tanh(beta_fx12) * self.weight_scale + 1
        else:
            beta_fx12 = ops.ones((n_cubes, 12), dtype=ms.float32)

        if alpha_fx8 is not None:
            alpha_fx8 = ops.tanh(alpha_fx8) * self.weight_scale + 1
        else:
            alpha_fx8 = ops.ones((n_cubes, 8), dtype=ms.float32)

        if gamma_f is not None:
            gamma_f = ops.sigmoid(gamma_f) * self.weight_scale + (1 - self.weight_scale) / 2
        else:
            gamma_f = ops.ones((n_cubes), dtype=ms.float32)

        return beta_fx12[surf_cubes], alpha_fx8[surf_cubes], gamma_f[surf_cubes]

    def _get_case_id(self, occ_fx8, surf_cubes, res):
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the
        supplementary material. It should be noted that this function assumes a regular grid.
        """
        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.unsqueeze(0)).sum(-1)

        problem_config = self.check_table[case_ids]
        to_check = problem_config[..., 0] == 1
        problem_config = problem_config[to_check]
        if not isinstance(res, (list, tuple)):
            res = [res, res, res]

        # The 'problematic_configs' only contain configurations for surface cubes. Next, we construct a 3D array,
        # 'problem_config_full', to store configurations for all cubes (with default config for non-surface cubes).
        # This allows efficient checking on adjacent cubes.
        problem_config_full = ops.zeros(list(res) + [5], dtype=ms.int64)
        vol_idx = ops.nonzero(problem_config_full[..., 0] == 0)  # N, 3
        vol_idx_problem = vol_idx[surf_cubes][to_check]
        problem_config_full[vol_idx_problem[..., 0], vol_idx_problem[..., 1], vol_idx_problem[..., 2]] = problem_config
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]

        within_range = (
            (vol_idx_problem_adj[..., 0] >= 0).to(ms.uint8)
            & (vol_idx_problem_adj[..., 0] < res[0]).to(ms.uint8)
            & (vol_idx_problem_adj[..., 1] >= 0).to(ms.uint8)
            & (vol_idx_problem_adj[..., 1] < res[1]).to(ms.uint8)
            & (vol_idx_problem_adj[..., 2] >= 0).to(ms.uint8)
            & (vol_idx_problem_adj[..., 2] < res[2]).to(ms.uint8)
        ).bool()

        vol_idx_problem = vol_idx_problem[within_range]
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_adj = problem_config_full[
            vol_idx_problem_adj[..., 0], vol_idx_problem_adj[..., 1], vol_idx_problem_adj[..., 2]
        ]

        # If two cubes with cases C16 and C19 share an ambiguous face, both cases are inverted.
        to_invert = problem_config_adj[..., 0] == 1
        idx = ops.arange(case_ids.shape[0])[to_check][within_range][to_invert]
        if len(idx) > 0:
            case_ids[(idx,)] = problem_config[to_invert][..., -1]
        return case_ids

    def _identify_surf_edges(self, s_n, cube_fx8, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge
        and marks the cube edges with this index.
        """
        occ_n = s_n < 0
        all_edges = cube_fx8[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        # unique_edges, _idx_map, counts = ops.unique_consecutive(all_edges, axis=0, return_idx=True, return_counts=True)
        # unique_edges, _idx_map, counts = mint.unique(all_edges, axis=0, return_inverse=True, return_counts=True)  # ms not supporting this
        unique_edges, _idx_map, counts = mint.unique(
            all_edges.to(ms.bfloat16), dim=0, return_inverse=True, return_counts=True
        )
        unique_edges = unique_edges.long()
        mask_edges = occ_n.to(ms.uint8)[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges.to(ms.uint8)[_idx_map]
        counts = counts[_idx_map]

        mapping = ops.ones((unique_edges.shape[0]), dtype=ms.int64) * -1
        mapping[mask_edges] = ops.arange(mask_edges.sum())
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    def _identify_surf_cubes(self, s_n, cube_fx8):
        """
        Identifies grid cubes that intersect with the underlying surface by checking if the signs at
        all corners are not identical.
        """
        occ_n = ms.Tensor(
            s_n < 0, dtype=ms.uint8
        )  # bool type cannot be sampled in the following line, needs to be ms tensor
        occ_fx8 = occ_n[cube_fx8.reshape(-1)].reshape(-1, 8)
        _occ_sum = ops.sum(occ_fx8, -1)  # 8 verts per cube
        surf_cubes = (_occ_sum > 0).to(ms.uint8) & (_occ_sum < 8).to(ms.uint8)
        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = ops.cat(
            [
                mint.index_select(input=edges_weight, index=ms.Tensor([1]), dim=edge_dim),
                -mint.index_select(input=edges_weight, index=ms.Tensor([0]), dim=edge_dim),
            ],
            edge_dim,
        )
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _compute_vd(
        self, x_nx3, surf_cubes_fx8, surf_edges, s_n, case_ids, beta_fx12, alpha_fx8, gamma_f, idx_map, grad_func
    ):
        """
        Computes the location of dual vertices as described in Section 4.2
        """
        alpha_nx12x2 = mint.index_select(input=alpha_fx8, index=self.cube_edges, dim=1).reshape(-1, 12, 2)
        surf_edges_x = mint.index_select(input=x_nx3, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 3)
        surf_edges_s = mint.index_select(input=s_n, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 1)
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)

        idx_map = idx_map.reshape(-1, 12)
        num_vd = mint.index_select(input=self.num_vd_table, index=case_ids, dim=0)
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []

        total_num_vd = 0
        vd_idx_map = ops.zeros((case_ids.shape[0], 12), dtype=ms.int64)

        for num in mint.unique(num_vd):
            cur_cubes = num_vd == num  # consider cubes with the same numbers of vd emitted (for batching)
            curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            curr_edge_group_to_vd = ops.arange(curr_num_vd).unsqueeze(-1).tile((1, 7)) + total_num_vd
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = (
                ops.arange(idx_map.shape[0])[cur_cubes].unsqueeze(-1).tile((1, num * 7)).reshape_as(curr_edge_group)
            )

            curr_mask = curr_edge_group != -1
            edge_group.append(ops.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(ops.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
            edge_group_to_cube.append(ops.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True))
            vd_gamma.append(ops.masked_select(gamma_f, cur_cubes).unsqueeze(-1).tile((1, num)).reshape(-1))

        edge_group = ops.cat(edge_group)
        edge_group_to_vd = ops.cat(edge_group_to_vd)
        edge_group_to_cube = ops.cat(edge_group_to_cube)
        vd_num_edges = ops.cat(vd_num_edges)
        vd_gamma = ops.cat(vd_gamma)

        vd = ops.zeros((total_num_vd, 3))
        beta_sum = ops.zeros((total_num_vd, 1))

        idx_group = mint.gather(input=idx_map.reshape(-1), dim=0, index=edge_group_to_cube * 12 + edge_group)

        x_group = mint.index_select(input=surf_edges_x, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 3)
        s_group = mint.index_select(input=surf_edges_s, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 1)

        zero_crossing_group = mint.index_select(input=zero_crossing, index=idx_group.reshape(-1), dim=0).reshape(-1, 3)

        alpha_group = mint.index_select(
            input=alpha_nx12x2.reshape(-1, 2), dim=0, index=edge_group_to_cube * 12 + edge_group
        ).reshape(-1, 2, 1)
        ue_group = self._linear_interp(s_group * alpha_group, x_group)

        beta_group = mint.gather(
            input=beta_fx12.reshape(-1), dim=0, index=edge_group_to_cube * 12 + edge_group
        ).reshape(-1, 1)
        beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
        vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum
        L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges)

        v_idx = ops.arange(vd.shape[0])  # + total_num_vd

        vd_idx_map = ops.scatter(
            input=vd_idx_map.reshape(-1),
            axis=0,
            index=edge_group_to_cube * 12 + edge_group,
            src=v_idx[edge_group_to_vd],
        )

        return vd, L_dev, vd_gamma, vd_idx_map

    def _triangulate(
        self, s_n, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, training, grad_func
    ):
        """
        Connects four neighboring dual vertices to form a quadrilateral. The quadrilaterals are then split into
        triangles based on the gamma parameter, as described in Section 4.3.
        """
        # with ops.no_grad():
        group_mask = (edge_counts == 4) & surf_edges_mask  # surface edges shared by 4 cubes.
        group = idx_map.reshape(-1)[group_mask]
        vd_idx = vd_idx_map[group_mask]
        edge_indices, indices = ops.sort(group, stable=True)
        quad_vd_idx = vd_idx[indices].reshape(-1, 4)

        # Ensure all face directions point towards the positive SDF to maintain consistent winding.
        s_edges = s_n[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
        flip_mask = s_edges[:, 0] > 0
        quad_vd_idx = ops.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]], quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))
        if grad_func is not None:
            # when grad_func is given, split quadrilaterals along the diagonals with more consistent gradients.
            # with ops.no_grad():
            vd_gamma = ops.norm(grad_func(vd), dim=-1)
            quad_gamma = mint.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
            gamma_02 = (quad_gamma[:, 0] * quad_gamma[:, 2]).sum(-1, keepdims=True)
            gamma_13 = (quad_gamma[:, 1] * quad_gamma[:, 3]).sum(-1, keepdims=True)
        else:
            quad_gamma = mint.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)
            gamma_02 = mint.index_select(input=quad_gamma, index=ms.Tensor([0]), dim=1) * mint.index_select(
                input=quad_gamma, index=ms.Tensor([2]), dim=1
            )
            gamma_13 = mint.index_select(input=quad_gamma, index=ms.Tensor([1]), dim=1) * mint.index_select(
                input=quad_gamma, index=ms.Tensor([3]), dim=1
            )
        if not training:
            mask = (gamma_02 > gamma_13).squeeze(1)
            faces = ops.zeros((quad_gamma.shape[0], 6), dtype=ms.int64)
            faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
            faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
            faces = faces.reshape(-1, 3)
        else:
            vd_quad = mint.index_select(input=vd, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
            vd_02 = (
                mint.index_select(input=vd_quad, index=ms.Tensor([0]), dim=1)
                + mint.index_select(input=vd_quad, index=ms.Tensor([2]), dim=1)
            ) / 2
            vd_13 = (
                mint.index_select(input=vd_quad, index=ms.Tensor([1]), dim=1)
                + mint.index_select(input=vd_quad, index=ms.Tensor([3]), dim=1)
            ) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = (
                (vd_02 * gamma_02.unsqueeze(-1) + vd_13 * gamma_13.unsqueeze(-1)) / weight_sum.unsqueeze(-1)
            ).squeeze(1)
            vd_center_idx = ops.arange(vd_center.shape[0]) + vd.shape[0]
            vd = ops.cat([vd, vd_center])
            faces = quad_vd_idx[:, self.quad_split_train].reshape(-1, 4, 2)
            faces = ops.cat([faces, vd_center_idx.reshape(-1, 1, 1).tile((1, 4, 1))], -1).reshape(-1, 3)
        return vd, faces, s_edges, edge_indices

    def _tetrahedralize(
        self,
        x_nx3,
        s_n,
        cube_fx8,
        vertices,
        faces,
        surf_edges,
        s_edges,
        vd_idx_map,
        case_ids,
        edge_indices,
        surf_cubes,
        training,
    ):
        """
        Tetrahedralizes the interior volume to produce a tetrahedral mesh, as described in Section 4.5.
        """
        occ_n = s_n < 0
        occ_fx8 = occ_n[cube_fx8.reshape(-1)].reshape(-1, 8)
        occ_sum = ops.sum(occ_fx8, -1)

        inside_verts = x_nx3[occ_n]
        mapping_inside_verts = ops.ones((occ_n.shape[0]), dtype=ms.int64) * -1
        mapping_inside_verts[occ_n] = ops.arange(occ_n.sum()) + vertices.shape[0]
        """
        For each grid edge connecting two grid vertices with different
        signs, we first form a four-sided pyramid by connecting one
        of the grid vertices with four mesh vertices that correspond
        to the grid edge and then subdivide the pyramid into two tetrahedra
        """
        inside_verts_idx = mapping_inside_verts[
            surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1, 2)[s_edges < 0]
        ]
        if not training:
            inside_verts_idx = inside_verts_idx.unsqueeze(1).expand((-1, 2)).reshape(-1)
        else:
            inside_verts_idx = inside_verts_idx.unsqueeze(1).expand((-1, 4)).reshape(-1)

        tets_surface = ops.cat([faces, inside_verts_idx.unsqueeze(-1)], -1)
        """
        For each grid edge connecting two grid vertices with the
        same sign, the tetrahedron is formed by the two grid vertices
        and two vertices in consecutive adjacent cells
        """
        inside_cubes = occ_sum == 8
        inside_cubes_center = x_nx3[cube_fx8[inside_cubes].reshape(-1)].reshape(-1, 8, 3).mean(1)
        inside_cubes_center_idx = ops.arange(inside_cubes_center.shape[0]) + vertices.shape[0] + inside_verts.shape[0]

        surface_n_inside_cubes = surf_cubes | inside_cubes
        edge_center_vertex_idx = ops.ones(((surface_n_inside_cubes).sum(), 13), dtype=ms.int64) * -1
        surf_cubes = surf_cubes[surface_n_inside_cubes]
        inside_cubes = inside_cubes[surface_n_inside_cubes]
        edge_center_vertex_idx[surf_cubes, :12] = vd_idx_map.reshape(-1, 12)
        edge_center_vertex_idx[inside_cubes, 12] = inside_cubes_center_idx

        all_edges = cube_fx8[surface_n_inside_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_func = ops.UniqueConsecutive(axis=0, return_idx=True, return_counts=True)
        unique_edges, _idx_map, counts = unique_func(all_edges)
        # unique_edges, _idx_map, counts = mint.unique(all_edges, dim=0, return_inverse=True, return_counts=True)
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 2
        mask = mask_edges[_idx_map]
        counts = counts[_idx_map]
        mapping = ops.ones((unique_edges.shape[0]), dtype=ms.int64) * -1
        mapping[mask_edges] = ops.arange(mask_edges.sum())
        idx_map = mapping[_idx_map]

        group_mask = (counts == 4) & mask
        group = idx_map.reshape(-1)[group_mask]
        edge_indices, indices = ops.sort(group)
        cube_idx = (
            ops.arange((_idx_map.shape[0] // 12), dtype=ms.int64).unsqueeze(1).expand((-1, 12)).reshape(-1)[group_mask]
        )
        edge_idx = (
            ops.arange((12), dtype=ms.int64).unsqueeze(0).expand((_idx_map.shape[0] // 12, -1)).reshape(-1)[group_mask]
        )
        # Identify the face shared by the adjacent cells.
        cube_idx_4 = cube_idx[indices].reshape(-1, 4)
        edge_dir = self.edge_dir_table[edge_idx[indices]].reshape(-1, 4)[..., 0]
        shared_faces_4x2 = self.dir_faces_table[edge_dir].reshape(-1)
        cube_idx_4x2 = cube_idx_4[:, self.adj_pairs].reshape(-1)
        # Identify an edge of the face with different signs and
        # select the mesh vertex corresponding to the identified edge.
        case_ids_expand = ops.ones((surface_n_inside_cubes).sum(), dtype=ms.int64) * 255
        case_ids_expand[surf_cubes] = case_ids
        cases = case_ids_expand[cube_idx_4x2]
        quad_edge = edge_center_vertex_idx[cube_idx_4x2, self.tet_table[cases, shared_faces_4x2]].reshape(-1, 2)
        mask = (quad_edge == -1).sum(-1) == 0
        inside_edge = mapping_inside_verts[unique_edges[mask_edges][edge_indices].reshape(-1)].reshape(-1, 2)
        tets_inside = ops.cat([quad_edge, inside_edge], -1)[mask]

        tets = ops.cat([tets_surface, tets_inside])
        vertices = ops.cat([vertices, inside_verts, inside_cubes_center])
        return vertices, tets

    def construct(self, *args, **kwargs):
        return super().construct(*args, **kwargs)


if __name__ == "__main__":
    ms.context.set_context(mode=1, device_target="Ascend", device_id=7)
    test_fc = FlexiCubes(weight_scale=0.5)
    print(test_fc)
    v, i = test_fc.construct_voxel_grid(res=64)
    print(f"v shape {v.shape}")
    print(f"i shape {i.shape}")
