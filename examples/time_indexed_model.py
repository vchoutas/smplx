from __future__ import annotations

from typing import List, Optional, Dict, Callable

import sys
import os
import os.path as osp
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import smplx
from smplx.body_models import SMPLLayer, SMPLOutput
from smplx import lbs as lbs
from smplx.utils import batch_size_from_tensor_list, identity_rot_mats


from tqdm import tqdm
from loguru import logger
import polyscope as ps

from scipy.spatial.transform import Rotation as R

Tensor = torch.Tensor


class SMPLIndexed(SMPLLayer):
    def __init__(
        self,
        *args,
        vertex_indices=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert vertex_indices is not None
        vertex_indices = np.asarray(vertex_indices)
        self.num_vertices_to_keep = len(vertex_indices)

        joint_template = lbs.vertices2joints(
            self.J_regressor, self.v_template[None])
        self.register_buffer('joint_template', joint_template)

        shapedirs = self.shapedirs
        joint_shapedirs = lbs.vertices2joints(
            self.J_regressor, shapedirs.permute(2, 0, 1)).permute(1, 2, 0)
        self.register_buffer('joint_shapedirs', joint_shapedirs)

        self.shapedirs = self.shapedirs[vertex_indices]

        num_vertices = len(self.v_template)
        self.v_template = self.v_template[vertex_indices]

        selected_posedirs = self.posedirs.t().reshape(
            num_vertices, 3, -1)[vertex_indices]
        self.posedirs = selected_posedirs.reshape(
            -1, selected_posedirs.shape[-1]).t()

        self.lbs_weights = self.lbs_weights[vertex_indices]

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        pose2rot: bool = True,
        v_template: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> SMPLOutput:
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [betas, global_orient, body_pose, transl]
        batch_size = batch_size_from_tensor_list(model_vars)
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        targs = {'dtype': dtype, 'device': device}

        if global_orient is None:
            global_orient = identity_rot_mats(
                batch_size=batch_size, num_matrices=1, **targs)
        if body_pose is None:
            body_pose = identity_rot_mats(
                batch_size=batch_size, num_matrices=self.NUM_BODY_JOINTS,
                **targs)

        if global_orient is None:
            global_orient = torch.eye(3, **targs).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(batch_size, -1, 3, 3),
             ],
            dim=1)
        # shape_components = torch.cat([betas, expression], dim=-1)
        # shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)
        # shape_components = torch.cat([betas, expression], dim=-1)

        joints_shaped = self.joint_template + lbs.blend_shapes(
            betas, self.joint_shapedirs
        )
        num_joints = joints_shaped.shape[1]

        v_shaped = self.v_template + lbs.blend_shapes(
            betas, self.shapedirs)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        # rot_mats = lbs.batch_rodrigues(full_pose.view(-1, 3)).view(
        #     [batch_size, -1, 3, 3])
        rot_mats = full_pose.view(batch_size, -1, 3, 3)

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        joints, rel_transforms, abs_transforms = lbs.batch_rigid_transform(
            rot_mats, joints_shaped, self.parents)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        # W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        # T = torch.matmul(W, rel_transforms.view(batch_size, num_joints, 16)).view(
        #     batch_size, -1, 4, 4)
        T = torch.einsum('vj,bjmn->bvmn', [self.lbs_weights, rel_transforms])

        # homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], **targs)
        # v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        # v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        v_homo = torch.matmul(
            T, F.pad(v_posed, [0, 1], value=1).unsqueeze(dim=-1))

        vertices = v_homo[:, :, :3, 0]

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            joints=joints,
                            betas=betas,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            v_shaped=v_shaped,
                            full_pose=full_pose if return_full_pose else None,
                            transl=transl,
                            )
        return output


def main(
    model_folder,
    model_type='smplx',
    ext='npz',
    gender='neutral',
    num_betas: int = 10,
    num_expression_coeffs: int = 10,
    use_face_contour: bool = False,
    batch_size: int = 1,
    num_verts: int = 1000,
    show: bool = False,
) -> None:

    device = torch.device('cuda')

    model = smplx.build_layer(
        model_folder, model_type=model_type,
        gender=gender, use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext)
    model = model.to(device=device)

    # Sample num verts
    vertex_indices = np.random.choice(
        np.arange(0, model.num_verts), num_verts, replace=False,
    )

    fast_model = SMPLIndexed(
        osp.join(model_folder, 'smpl'),
        model_type=model_type,
        gender=gender, use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        optim_transform=True,
        ext=ext,
        extra_joint_module_type='from_transforms',
        lmk_obj_type='from_transforms',
        vertex_indices=vertex_indices
    ).to(device=device)

    betas, expression = None, None
    #  if sample_shape:
    #  betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    #  if sample_expression:
    #  expression = torch.randn(
    #  [1, model.num_expression_coeffs], dtype=torch.float32)

    body_pose = torch.eye(3).reshape(1, 1, 3, 3).expand(
        batch_size, fast_model.NUM_BODY_JOINTS,
        -1, -1).contiguous().to(device=device)

    r = R.from_rotvec([0, np.pi / 4, 0])

    # body_pose[:, :, :, :] = torch.from_numpy(r.as_matrix())

    N = 1000

    all_v2v = []
    timings = []
    fast_timings = []
    for n in tqdm(range(N)):

        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(betas=betas, expression=expression,
                       body_pose=body_pose,
                       return_verts=True)
        # fast_output = fast_model(betas=betas, expression=expression,
        #                          body_pose=body_pose,
        #                          return_verts=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        torch.cuda.synchronize()
        start = time.perf_counter()
        fast_output = fast_model(betas=betas, expression=expression,
                                 body_pose=body_pose,
                                 return_verts=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fast_timings.append(elapsed)

        normal_verts = output.vertices.detach().cpu()[:, vertex_indices]
        fast_verts = fast_output.vertices.detach().cpu()
        # logger.info(fast_output.vertices.shape)
        # logger.info(normal_verts.shape)
        # sys.exit(0)

        v2v = (normal_verts - fast_verts).pow(2).sum(dim=-1).sqrt().mean() * 1000
        #  logger.info(f'Vertex-to-vertex error: {v2v.item():.2f} mm')
        all_v2v.append(v2v)

    logger.info(f'Vertex-to-vertex error: {np.mean(all_v2v):.2f} mm')

    logger.warning(f'Batch size: {batch_size}')
    logger.info(f'Baseline: {np.mean(timings) * 1000:.3f} (ms)')
    logger.info(f'Fast: {np.mean(fast_timings) * 1000:.3f} (ms)')

    if show:
        ps.init()

        output = model(
            betas=betas, expression=expression, body_pose=body_pose,
            return_verts=True)
        fast_output = fast_model(betas=betas, expression=expression,
                                 body_pose=body_pose,
                                 return_verts=True)

        ps.register_surface_mesh(
            'Model mesh', output.vertices[0].detach().cpu().numpy(),
            model.faces, smooth_shade=True)
        ps.register_point_cloud(
            'Fast vertices',
            fast_output.vertices[0].detach().cpu().numpy())

        ps.show()


if __name__ == '__main__':
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level='INFO', colorize=True)

    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--num-verts', type=int, default=1000,
                        dest='num_verts',
                        help='Number of vertices to use')

    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        type=int, default=[1],
                        nargs='+',
                        help='Batch size')
    parser.add_argument('--show', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Show a result')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    num_verts = args.num_verts
    show = args.show

    for batch_size in args.batch_size:
        main(model_folder, model_type, ext=ext,
             gender=gender,
             num_betas=num_betas,
             batch_size=batch_size,
             num_expression_coeffs=num_expression_coeffs,
             use_face_contour=use_face_contour,
             num_verts=num_verts, show=show)
