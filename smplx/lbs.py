# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Tuple, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (rot_mat_to_euler, Tensor, Array, LBSOutput,
                    find_joint_kin_chain, Struct)

import sys
from loguru import logger


def compute_face_landmarks(
    vertices: Tensor,
    full_pose: Tensor,
    lmk_faces_idx: Tensor,
    lmk_bary_coords: Tensor,
    faces_tensor: Tensor,
    use_face_contour: bool = False,
    dynamic_lmk_faces_idx: Optional[Tensor] = None,
    dynamic_lmk_bary_coords: Optional[Tensor] = None,
    neck_kin_chain: Optional[Tensor] = None,
    pose2rot: bool = True,
) -> Tensor:

    batch_size = len(vertices)
    lmk_faces_idx = lmk_faces_idx.unsqueeze(dim=0).repeat(batch_size, 1)
    lmk_bary_coords = lmk_bary_coords.unsqueeze(dim=0).repeat(
        batch_size, 1, 1)
    if use_face_contour:
        assert dynamic_lmk_faces_idx is not None, (
            'Requested dynamic landmarks, but the face index tensor was not'
            ' given')
        assert dynamic_lmk_bary_coords is not None, (
            'Requested dynamic landmarks, but the barycentric coordinates'
            ' tensor was not given')
        lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            dynamic_lmk_faces_idx,
            dynamic_lmk_bary_coords,
            neck_kin_chain,
            pose2rot=pose2rot,
        )
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], dim=1)
        lmk_bary_coords = torch.cat(
            [lmk_bary_coords.expand(batch_size, -1, -1),
                dyn_lmk_bary_coords], 1)

    landmarks = vertices2landmarks(
        vertices, faces_tensor, lmk_faces_idx, lmk_bary_coords)
    return landmarks


def find_y_euler_lut_key(
    pose: Tensor,
    neck_kin_chain: Tensor,
    pose2rot: bool = True,
) -> Tensor:
    ''' Computes the LUT key used to select the correct
    '''
    dtype, device = pose.dtype, pose.device
    batch_size = len(pose)

    if pose2rot:
        aa_pose = torch.index_select(
            pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(
            pose.view(batch_size, -1, 3, 3), 1, neck_kin_chain)

    rel_rot_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze_(
        dim=0).repeat(batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

    return y_rot_angle


def find_dynamic_lmk_idx_and_bcoords(
    pose: Tensor,
    dynamic_lmk_faces_idx: Tensor,
    dynamic_lmk_b_coords: Tensor,
    neck_kin_chain: List[int],
    pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    y_rot_angle = find_y_euler_lut_key(pose, neck_kin_chain, pose2rot=pose2rot)

    dyn_lmk_faces_idx = torch.index_select(
        dynamic_lmk_faces_idx, 0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(
    vertices: Tensor,
    faces: Tensor,
    lmk_faces_idx: Tensor,
    lmk_bary_coords: Tensor
) -> Tensor:
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def pose_blendshapes(
    rot_mats: Tensor,
    posedirs: Tensor,
) -> Tensor:
    ''' Computes pose blendshapes from the input pose vector
    '''
    batch_size = len(rot_mats)
    dtype, device = rot_mats.dtype, rot_mats.device

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = rot_mats[:, 1:].view(batch_size, -1, 3, 3) - ident
    pose_offsets = torch.matmul(
        pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    return pose_offsets


def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
    transl: Optional[Tensor] = None,
    parallel_exec: List[List[int]] = None,
    task_group_parents: List[List[int]] = None,
    return_verts: bool = True,
) -> LBSOutput:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        transl: torch.Tensor, optional
            A tensor that contains the root translation of the body. If
            provided, it will be used to move the model joints and vertices
            to the desired location.
        Returns
        -------
            lbs_output: LBSOutput
            A dataclass that contains the vertices and joint transformations
            computed using linear blend skinning
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3))
    else:
        rot_mats = pose
    rot_mats = rot_mats.view([batch_size, -1, 3, 3])

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose_offsets = pose_blendshapes(rot_mats, posedirs=posedirs)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, rel_transforms, abs_transforms = batch_rigid_transform(
        rot_mats, J, parents, parallel_exec=parallel_exec,
        task_group_parents=task_group_parents,
    )

    verts, T = None, None
    if return_verts:
        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = J_regressor.shape[0]
        T = torch.matmul(W, rel_transforms.view(batch_size, num_joints, 16)).view(
            batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

    if transl is not None:
        # Update the translation for the
        transl_transf = transform_mat(
            torch.eye(3, dtype=dtype, device=device)[None].repeat(
                batch_size, 1, 1), transl.unsqueeze(dim=-1),)
        abs_transforms = torch.einsum(
            'bmk,bjkn->bjmn', transl_transf, abs_transforms)
        # abs_transforms[..., :3, 3] += transl.unsqueeze(dim=1)
        if verts is not None:
            verts += transl.unsqueeze(dim=1)

    return LBSOutput(
        _vertices=verts,
        _joints_transforms=abs_transforms,
        _rel_joints_transforms=rel_transforms,
        _v_shaped=v_shaped,
        _v_rest_pose=v_posed,
        _skinning_transforms=T,
    )


def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    parallel_exec: List[List[int]] = None,
    task_group_parents: List[List[int]] = None,
    transl: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    parallel_exec: List[List[int]], optional
        Contains groups of joints whose rigid transformation can be computed in
        parallel.
    task_group_parents: List[List[int]], optional
        Contains the indices of the parent joints of each joint in each group.
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    abs_transforms : torch.tensor BxNx4x4
        The rigid transformations for all the joints in world coordinates
    """

    # Get the device and data type from the joints tensor.
    device, dtype = joints.device, joints.dtype
    batch_size, num_joints = joints.shape[:2]

    # Add a dummy dimension to the joint tensor
    joints = torch.unsqueeze(joints, dim=-1)

    # Compute the parent relative coordinates of the joints.
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # Compute the transformation matrix of each joint. Note that we have not
    # traversed the kinematic tree, so these matrices are relative to the
    # parent.
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    if parallel_exec is not None and task_group_parents is not None:
        # Create the final transformation array
        transforms = torch.eye(4, dtype=dtype, device=device).reshape(
            1, 1, 4, 4).repeat(batch_size, num_joints, 1, 1)
        # Assign the root transformation
        transforms[:, 0] = transforms_mat[:, 0]
        num_groups = len(parallel_exec)
        # Iterate through the parallel groups
        # The idea here is that there joints in the kinematic tree that do not
        # depend upon each other. We can thus compute their transformation in
        # parallel.
        for ii in range(num_groups):
            # For the current group compute the transformations
            transforms[:, parallel_exec[ii]] = torch.matmul(
                transforms[:, task_group_parents[ii]],
                transforms_mat[:, parallel_exec[ii]])
    else:
        transform_chain = [transforms_mat[:, 0]]
        # Traverse the kinematic tree sequentially
        for i in range(1, parents.shape[0]):
            # Compute the absolute transformation for the current joint
            curr_res = torch.matmul(
                transform_chain[parents[i]], transforms_mat[:, i])
            transform_chain.append(curr_res)

        # Collect all transformations into a single tensor.
        transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms, transforms


class LandmarksFromVertices(nn.Module):
    ''' Computes landmarks from a set of vertices with barycentric interpolation

    '''

    def __init__(
        self,
        lmk_faces_idx: Array,
        lmk_bary_coords: Array,
        use_face_contour: bool = False,
        dynamic_lmk_faces_idx: Optional[Array] = False,
        dynamic_lmk_bary_coords: Optional[Array] = False,
        neck_index: int = 1,
        model_kin_chain: Optional[Array] = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            'lmk_faces_idx', torch.tensor(lmk_faces_idx, dtype=torch.int64))
        self.register_buffer(
            'lmk_bary_coords', torch.tensor(lmk_bary_coords, dtype=dtype))

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            if not torch.is_tensor(dynamic_lmk_faces_idx):
                dynamic_lmk_faces_idx = torch.tensor(
                    dynamic_lmk_faces_idx, dtype=torch.long)
            self.register_buffer(
                'dynamic_lmk_faces_idx', dynamic_lmk_faces_idx)

            if not torch.is_tensor(dynamic_lmk_bary_coords):
                dynamic_lmk_bary_coords = torch.tensor(
                    dynamic_lmk_bary_coords, dtype=dtype)
            self.register_buffer(
                'dynamic_lmk_bary_coords', dynamic_lmk_bary_coords)

            neck_kin_chain = find_joint_kin_chain(neck_index, model_kin_chain)
            self.register_buffer(
                'neck_kin_chain',
                torch.tensor(neck_kin_chain, dtype=torch.long))

    @property
    def num_static_landmarks(self) -> int:
        return len(self.lmk_bary_coords)

    @property
    def num_dynamic_landmarks(self) -> int:
        return len(getattr(self, 'dynamic_lmk_bary_coords', []))

    @property
    def num_landmarks(self) -> int:
        return self.num_static_landmarks + self.num_dynamic_landmarks

    def forward(
        self,
        vertices: Tensor,
        full_pose: Tensor,
        faces_tensor: Tensor,
        pose2rot: bool = True,
        **kwargs,
    ) -> Tensor:
        return compute_face_landmarks(
            vertices=vertices,
            full_pose=full_pose,
            lmk_faces_idx=self.lmk_faces_idx,
            lmk_bary_coords=self.lmk_bary_coords,
            faces_tensor=faces_tensor,
            dynamic_lmk_faces_idx=getattr(self, 'dynamic_lmk_faces_idx', None),
            dynamic_lmk_bary_coords=getattr(
                self, 'dynamic_lmk_bary_coords', None),
            neck_kin_chain=getattr(self, 'neck_kin_chain', None),
            pose2rot=pose2rot,
            use_face_contour=self.use_face_contour,
        )


class LandmarksFromJointTransforms(nn.Module):
    ''' Computes landmarks with linear blend skinning of a vertex subset
    '''

    def __init__(
        self,
        lmk_faces_idx: Array,
        lmk_bary_coords: Array,
        skinning_weights: Array,
        v_template: Array,
        blendshapes: Array,
        posedirs: Array,
        faces: Array,
        use_face_contour: bool = False,
        dynamic_lmk_faces_idx: Optional[Array] = False,
        dynamic_lmk_bary_coords: Optional[Array] = False,
        neck_index: int = 1,
        model_kin_chain: Optional[Array] = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Store the landmark barycentric coordinates
        self.register_buffer(
            'lmk_bary_coords', torch.tensor(lmk_bary_coords, dtype=dtype))

        # Get the vertex indices needed to compute the landmarks
        vertex_indices = faces[lmk_faces_idx]

        self._num_static_lmk_vertices = len(vertex_indices)

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            assert dynamic_lmk_faces_idx is not None, (
                'Requested dynamic landmarks, but the face index tensor was not'
                ' given')
            assert dynamic_lmk_bary_coords is not None, (
                'Requested dynamic landmarks, but the barycentric coordinates'
                ' tensor was not given')
            # Store the vertices for the dynamic landmarks
            dynamic_vertex_indices = faces[dynamic_lmk_faces_idx]
            self._num_dynamic_lmk_vertices = len(dynamic_vertex_indices)

            vertex_indices = np.concatenate(
                [vertex_indices.flatten(), dynamic_vertex_indices.flatten()])

            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=dtype)
            self.register_buffer(
                'dynamic_lmk_bary_coords', dynamic_lmk_bary_coords)

            # Get the kinematic chain for the neck
            neck_kin_chain = find_joint_kin_chain(neck_index, model_kin_chain)
            self.register_buffer(
                'neck_kin_chain',
                torch.tensor(neck_kin_chain, dtype=torch.long))

        # The vertex list might contain duplicates, so find the unique
        # vertices to minimize computations
        vertex_indices, unique_to_full_indices = np.unique(
            vertex_indices, return_inverse=True)
        self.vertex_indices = vertex_indices

        unique_to_full_indices_tensor = torch.tensor(
            unique_to_full_indices, dtype=torch.int64)
        self.register_buffer(
            'unique_to_full_indices', unique_to_full_indices_tensor)

        # Get the skinning weights for the selected vertices
        lmk_skinning_weights = skinning_weights[vertex_indices]
        lmk_skinning_weights_tensor = torch.tensor(
            lmk_skinning_weights, dtype=dtype)
        self.register_buffer(
            'lmk_skinning_weights', lmk_skinning_weights_tensor)

        # Store the blend shape components for the landmark vertices.
        lmk_blendshapes = blendshapes[vertex_indices]
        lmk_blendshapes_tensor = torch.tensor(lmk_blendshapes, dtype=dtype)
        self.register_buffer('lmk_blendshapes', lmk_blendshapes_tensor)

        # Store the pose blend shapes for the landmark vertices.
        posedirs = posedirs.reshape(-1, len(v_template), 3)
        lmk_posedirs = posedirs[:, vertex_indices].reshape(
            -1, len(vertex_indices) * 3)
        lmk_posedirs_tensor = torch.tensor(lmk_posedirs, dtype=dtype)
        self.register_buffer('lmk_posedirs', lmk_posedirs_tensor)

        # Store the landmark positions on the template
        lmk_template = v_template[vertex_indices]
        lmk_template_tensor = torch.tensor(lmk_template, dtype=dtype)
        self.register_buffer('lmk_template', lmk_template_tensor)

    @property
    def num_static_landmarks(self) -> int:
        return len(self.lmk_bary_coords)

    @property
    def num_dynamic_landmarks(self) -> int:
        return len(getattr(self, 'dynamic_lmk_bary_coords', []))

    @property
    def num_landmarks(self) -> int:
        return self.num_static_landmarks + self.num_dynamic_landmarks

    def forward(
        self,
        joints_transforms: Optional[Tensor] = None,
        blendshape_coefficients: Optional[Tensor] = None,
        full_pose: Optional[Tensor] = None,
        pose2rot: bool = True,
        transl: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        ''' Computes the landmarks

            Parameters
            ----------
                joints_transforms: Optional[Tensor] = None
                    A tensor with shape BxJx4x4 that contains the
                    root-relative joint transformations.
                blendshape_coefficients: Optional[Tensor] = None
                    The blendshape coefficients.
                full_pose: Optional[Tensor] = None
                    The full pose vector.
                pose2rot: bool = True
                    Whether to convert the pose vector to rotation matrices.
                transl: Optional[Tensor] = None
                    The root translation tensor.
            Returns
            -------
                The tensor that contains the compute landmarks
        '''

        assert joints_transforms is not None, (
            'Cannot compute landmarks when joint_transforms is None')

        vertices_for_landmarks_transforms = torch.einsum(
            'lj,bjmn->blmn', self.lmk_skinning_weights, joints_transforms)
        batch_size = len(joints_transforms)

        if pose2rot:
            rot_mats = batch_rodrigues(
                full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        else:
            rot_mats = full_pose
        rot_mats = rot_mats.view([batch_size, -1, 3, 3])

        # Compute the landmark vertices at rest pose
        lmk_verts_rest = self.lmk_template + blend_shapes(
            blendshape_coefficients, self.lmk_blendshapes) + pose_blendshapes(
                rot_mats, posedirs=self.lmk_posedirs)

        # Convert the vertices to homogeneous coordinates
        lmk_verts_rest_homo = F.pad(lmk_verts_rest, [0, 1], value=1.0)

        lmk_vertices_unique = torch.einsum(
            'bvmn,bvn->bvm', vertices_for_landmarks_transforms,
            lmk_verts_rest_homo)[..., :3]
        if transl is not None:
            lmk_vertices_unique += transl.unsqueeze(dim=1)

        lmk_vertices = lmk_vertices_unique[:, self.unique_to_full_indices]

        static_lmk_vertices = lmk_vertices[
            :, :self._num_static_lmk_vertices * 3].reshape(
                batch_size, self.num_static_landmarks, 3, 3)

        static_landmarks = torch.einsum(
            'lv,blvm->blm', self.lmk_bary_coords,
            static_lmk_vertices.reshape(
                batch_size, self.num_static_landmarks, 3, 3),)
        landmarks = static_landmarks
        if self.use_face_contour:
            dyn_lmk_bary_coords = self.dynamic_lmk_bary_coords
            dyn_lmk_shape = dyn_lmk_bary_coords.shape[:2]

            start = self._num_static_lmk_vertices * 3
            dyn_lmk_vertices = lmk_vertices[:, start:].reshape(
                batch_size, *dyn_lmk_shape, 3, 3)

            y_rot_angle = find_y_euler_lut_key(
                full_pose, self.neck_kin_chain, pose2rot=pose2rot)

            curr_dyn_bary_coords = dyn_lmk_bary_coords[y_rot_angle]

            ones = [1] * len(dyn_lmk_vertices.shape[1:])
            gather_indices = y_rot_angle.reshape(batch_size, *ones).repeat(
                1, 1, *dyn_lmk_vertices.shape[2:])
            curr_dyn_lmk_vertices = torch.gather(
                dyn_lmk_vertices, 1, gather_indices).flatten(end_dim=1)

            dynamic_landmarks = torch.einsum(
                'blv,blvm->blm', curr_dyn_bary_coords, curr_dyn_lmk_vertices)
            landmarks = torch.cat([static_landmarks, dynamic_landmarks], dim=1)

        return landmarks


def build_landmark_object(
    data_struct: Struct,
    blendshapes: Tensor,
    type: str = 'from_vertices',
    use_face_contour: bool = False,
    neck_index: int = 1,
):
    num_pose_basis = data_struct.posedirs.shape[-1]
    posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T

    if torch.is_tensor(blendshapes):
        blendshapes = blendshapes.detach().cpu().numpy()

    model_kin_chain = data_struct.kintree_table[0].copy()
    model_kin_chain[0] = -1

    if type == 'from_vertices':
        return LandmarksFromVertices(
            lmk_faces_idx=data_struct.lmk_faces_idx,
            lmk_bary_coords=data_struct.lmk_bary_coords,
            use_face_contour=use_face_contour,
            dynamic_lmk_faces_idx=data_struct.dynamic_lmk_faces_idx,
            dynamic_lmk_bary_coords=data_struct.dynamic_lmk_bary_coords,
            neck_index=neck_index,
            model_kin_chain=model_kin_chain,
        )
    elif type == 'from_transforms':
        return LandmarksFromJointTransforms(
            skinning_weights=data_struct.weights,
            v_template=data_struct.v_template,
            faces=data_struct.f,
            lmk_faces_idx=data_struct.lmk_faces_idx,
            lmk_bary_coords=data_struct.lmk_bary_coords,
            use_face_contour=use_face_contour,
            dynamic_lmk_faces_idx=data_struct.dynamic_lmk_faces_idx,
            dynamic_lmk_bary_coords=data_struct.dynamic_lmk_bary_coords,
            posedirs=posedirs,
            blendshapes=blendshapes,
            neck_index=neck_index,
            model_kin_chain=model_kin_chain,
        )
    else:
        raise ValueError(f'Unknown landmark computation object: {type}')
