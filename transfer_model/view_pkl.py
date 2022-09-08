import os.path as osp
import argparse

import numpy as np
import torch

import pyrender
import trimesh

import smplx

from tqdm.auto import tqdm, trange

from pathlib import Path

def main(model_folder,
         motion_file,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_expression=True,
         num_expression_coeffs=10,
         use_face_contour=False):

    # open motion file
    motion = np.load(motion_file, allow_pickle=True)
    _motion = {}
    for k,v in motion.items():
        if isinstance(v, np.ndarray):
            print(k, motion[k].shape, motion[k].dtype)
            if motion[k].dtype in ("<U7", "<U5", "<U4", "object", "|S7"):
                _motion[k] = str(motion[k])
            else:
                _motion[k] = torch.from_numpy(motion[k]).float()
        else:
            print(k, v)
            _motion[k] = v
    motion = _motion

    if "poses" in motion:
        motion["global_orient"] = motion["root_orient"]
        motion["body_pose"] = motion["pose_body"] # seriously?
        motion["left_hand_pose"] = motion["pose_hand"][:,:45]
        motion["right_hand_pose"] = motion["pose_hand"][:,45:]

    num_betas = len(motion['betas'])
    gender = str(motion['gender'])

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         use_pca=False,
                         ext=ext)

    betas, expression = motion['betas'], None
    betas = betas.unsqueeze(0)[:, :model.num_betas]
    global_orient = motion['global_orient']
    body_pose = motion['body_pose']
    left_hand_pose = motion['left_hand_pose']
    right_hand_pose = motion['right_hand_pose']
    # if sample_expression:
    #     expression = torch.randn(
    #         [1, model.num_expression_coeffs], dtype=torch.float32)

    #print(expression)
    #print(betas.shape, body_pose.shape, expression.shape)
    for pose_idx in trange(body_pose.size(0)):
        pose_idx = [pose_idx]
        # output = model(betas=betas, # expression=expression,
        #                return_verts=True)
        # for x in [betas, global_orient, body_pose, left_hand_pose, right_hand_pose]:
        #     print(x.dtype, x.shape)
        output = model(
                betas=betas,
                global_orient=global_orient[pose_idx],
                body_pose=body_pose[pose_idx],
                left_hand_pose=left_hand_pose[pose_idx],
                right_hand_pose=right_hand_pose[pose_idx],
                # expression=expression,
                return_verts=True
                )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                    vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--motion-file', required=True, type=str,
                        help='The path to the motion file to process')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    def resolve(path):
        return osp.expanduser(osp.expandvars(path))
    model_folder = resolve(args.model_folder)
    motion_file = resolve(args.motion_file)
    ext = args.ext
    num_expression_coeffs = args.num_expression_coeffs
    sample_expression = args.sample_expression

    main(model_folder, motion_file, ext=ext,
         sample_expression=sample_expression,
         use_face_contour=args.use_face_contour)
