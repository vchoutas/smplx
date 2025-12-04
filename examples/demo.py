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

import os
import os.path as osp
import argparse

import numpy as np
import torch

import smplx


def load_smpl_file(file_path):
    """
    Load .smpl file (which is actually .npz format with specific keys)
    Returns a dictionary with SMPL-X compatible parameters

    Parameters
    ----------
    file_path : str
        Path to the .smpl file

    Returns
    -------
    dict
        Dictionary containing SMPL-X compatible parameters:
        - betas: torch.Tensor, shape [1, num_betas]
        - body_pose: torch.Tensor, shape [1, 21, 3] or None
        - global_orient: torch.Tensor, shape [1, 3] or None
        - left_hand_pose: torch.Tensor or None
        - right_hand_pose: torch.Tensor or None
        - transl: torch.Tensor, shape [1, 3] or None
        - gender: str
    """
    data = np.load(file_path, allow_pickle=True)

    params = {}

    # Shape parameters (betas)
    if 'shapeParameters' in data:
        betas = torch.from_numpy(data['shapeParameters']).float()
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)  # Add batch dimension
        params['betas'] = betas

    # Body pose (22 joints - need to convert to 21 for SMPL-X)
    if 'bodyPose' in data:
        body_pose = torch.from_numpy(data['bodyPose']).float()  # (1, 22, 3)
        # SMPL-X expects 21 body joints (excluding root)
        # The first joint in bodyPose might be root, so we take joints 1-21
        if body_pose.shape[1] == 22:
            # Skip first, take next 21
            params['body_pose'] = body_pose[:, 1:22, :]
        elif body_pose.shape[1] == 21:
            params['body_pose'] = body_pose
        else:
            # Reshape if needed: (1, 63) -> (1, 21, 3)
            if len(body_pose.shape) == 2 and body_pose.shape[1] == 63:
                params['body_pose'] = body_pose.reshape(1, 21, 3)
            else:
                params['body_pose'] = body_pose

    # Head pose - extract global orientation
    # headPose is (1, 3, 3) rotation matrices, we need axis-angle
    if 'headPose' in data:
        head_pose = torch.from_numpy(data['headPose']).float()  # (1, 3, 3)
        # Convert rotation matrix to axis-angle
        # Using Rodrigues' formula for conversion
        if head_pose.shape[1] == 3 and head_pose.shape[2] == 3:
            # Extract first rotation matrix as global_orient
            # head_pose[0] gives us the (3, 3) rotation matrix
            R = head_pose[0]  # First rotation matrix (3x3)
            # Convert rotation matrix to axis-angle using Rodrigues' formula
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
            if angle < 1e-6:
                # No rotation
                params['global_orient'] = torch.zeros(
                    1, 3, dtype=torch.float32)
            else:
                # Extract axis
                axis = torch.stack([
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]
                ]) / (2 * torch.sin(angle))
                params['global_orient'] = (angle * axis).unsqueeze(0)

    # Hand poses
    if 'leftHandPose' in data:
        left_hand = torch.from_numpy(data['leftHandPose']).float()
        if len(left_hand.shape) == 3:  # (1, 15, 3)
            # Flatten to (1, 45) for PCA or keep as (1, 15, 3) for axis-angle
            params['left_hand_pose'] = left_hand.reshape(1, -1)
        else:
            params['left_hand_pose'] = left_hand

    if 'rightHandPose' in data:
        right_hand = torch.from_numpy(data['rightHandPose']).float()
        if len(right_hand.shape) == 3:  # (1, 15, 3)
            params['right_hand_pose'] = right_hand.reshape(1, -1)
        else:
            params['right_hand_pose'] = right_hand

    # Translation
    if 'bodyTranslation' in data:
        params['transl'] = torch.from_numpy(data['bodyTranslation']).float()

    # Gender mapping
    if 'gender' in data:
        gender_val = int(data['gender'])
        # Common mapping: 0=neutral, 1=male, 2=female (may vary)
        gender_map = {0: 'neutral', 1: 'male', 2: 'female', -1: 'male'}
        params['gender'] = gender_map.get(gender_val, 'neutral')
    else:
        params['gender'] = 'neutral'

    return params


def modify_body_attributes(betas, body_fat=None, lean_mass_arms=None,
                           lean_mass_torso=None, lean_mass_legs=None):
    """
    Modify body attributes by adjusting beta values.

    NOTE: These mappings are EXPERIMENTAL and may need adjustment.
    The exact beta indices that control these attributes may vary.

    Parameters
    ----------
    betas : torch.Tensor
        Shape parameters tensor [1, num_betas]
    body_fat : float, optional
        Body fat percentage modifier (-1.0 to 1.0 range)
    lean_mass_arms : float, optional
        Arms lean mass modifier
    lean_mass_torso : float, optional
        Torso lean mass modifier
    lean_mass_legs : float, optional
        Legs lean mass modifier

    Returns
    -------
    torch.Tensor
        Modified betas tensor
    """
    # Create a copy to avoid modifying the original
    modified_betas = betas.clone()

    # Body Fat %: Typically affects overall weight/thickness
    # Experimental: betas[0] and betas[1] often control weight/thickness
    if body_fat is not None:
        # Positive values = more body fat, negative = less body fat
        modified_betas[0, 0] += body_fat * 0.5  # Primary weight parameter
        if modified_betas.shape[1] > 1:
            modified_betas[0, 1] += body_fat * \
                0.3  # Secondary weight parameter

    # Lean Mass Arms: Experimental - may affect arm muscle mass
    # Try betas[2] or betas[3] for arm-related shape
    if lean_mass_arms is not None:
        if modified_betas.shape[1] > 2:
            modified_betas[0, 2] += lean_mass_arms * 0.4
        if modified_betas.shape[1] > 3:
            modified_betas[0, 3] += lean_mass_arms * 0.2

    # Lean Mass Torso: Experimental - may affect torso muscle mass
    # Try betas[4] or betas[5] for torso-related shape
    if lean_mass_torso is not None:
        if modified_betas.shape[1] > 4:
            modified_betas[0, 4] += lean_mass_torso * 0.4
        if modified_betas.shape[1] > 5:
            modified_betas[0, 5] += lean_mass_torso * 0.2

    # Lean Mass Legs: Experimental - may affect leg muscle mass
    # Try betas[6] or betas[7] for leg-related shape
    if lean_mass_legs is not None:
        if modified_betas.shape[1] > 6:
            modified_betas[0, 6] += lean_mass_legs * 0.4
        if modified_betas.shape[1] > 7:
            modified_betas[0, 7] += lean_mass_legs * 0.2

    return modified_betas


def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=False,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False,
         output_path=None,
         params_file=None,
         body_fat=None,
         lean_mass_arms=None,
         lean_mass_torso=None,
         lean_mass_legs=None):

    # Load parameters from .smpl file if provided
    loaded_params = {}
    if params_file and osp.exists(params_file):
        print(f'Loading parameters from: {params_file}')
        loaded_params = load_smpl_file(params_file)

        # Override gender if specified in file
        if 'gender' in loaded_params:
            gender = loaded_params['gender']
            print(f'Gender from file: {gender}')

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    print(model)

    # Initialize parameters
    betas, expression = None, None
    body_pose = None
    global_orient = None
    left_hand_pose = None
    right_hand_pose = None
    transl = None

    # Load from file if provided
    if loaded_params:
        betas = loaded_params.get('betas')
        body_pose = loaded_params.get('body_pose')
        global_orient = loaded_params.get('global_orient')
        transl = loaded_params.get('transl')

        # Handle hand poses - check if model uses PCA
        # If model uses PCA and we have axis-angle format, skip hand poses
        # (converting axis-angle to PCA requires inverse transformation)
        loaded_left_hand = loaded_params.get('left_hand_pose')
        loaded_right_hand = loaded_params.get('right_hand_pose')

        if hasattr(model, 'use_pca') and model.use_pca:
            # Model expects PCA coefficients (typically 6 values), but we have axis-angle (45 values)
            # Skip hand poses and let model use defaults
            expected_pca_size = getattr(model, 'num_pca_comps', 6)
            if loaded_left_hand is not None and loaded_left_hand.shape[1] != expected_pca_size:
                print(f'Warning: Model uses PCA for hands ({expected_pca_size} components), '
                      f'but loaded hand poses are axis-angle format ({loaded_left_hand.shape[1]} values). '
                      f'Skipping hand poses.')
                left_hand_pose = None
                right_hand_pose = None
            else:
                left_hand_pose = loaded_left_hand
                right_hand_pose = loaded_right_hand
        else:
            # Model doesn't use PCA, can use axis-angle directly
            left_hand_pose = loaded_left_hand
            right_hand_pose = loaded_right_hand

        print(f'Loaded: betas={betas is not None}, body_pose={body_pose is not None}, '
              f'global_orient={global_orient is not None}, '
              f'left_hand_pose={left_hand_pose is not None}, '
              f'right_hand_pose={right_hand_pose is not None}')

    # If no file loaded, use defaults
    if betas is None:
        if sample_shape:
            betas = torch.zeros([1, model.num_betas], dtype=torch.float32)
        else:
            betas = torch.zeros([1, model.num_betas], dtype=torch.float32)

    if expression is None:
        if sample_expression:
            expression = torch.randn(
                [1, model.num_expression_coeffs], dtype=torch.float32)
        else:
            expression = torch.zeros(
                [1, model.num_expression_coeffs], dtype=torch.float32)

    # Apply body modifications if specified
    if body_fat is not None or lean_mass_arms is not None or \
       lean_mass_torso is not None or lean_mass_legs is not None:
        print('Applying body modifications...')
        betas = modify_body_attributes(
            betas,
            body_fat=body_fat,
            lean_mass_arms=lean_mass_arms,
            lean_mass_torso=lean_mass_torso,
            lean_mass_legs=lean_mass_legs
        )
        print(f'Modified betas: {betas[0, :min(8, betas.shape[1])].tolist()}')

    # Generate mesh with all parameters
    output = model(
        betas=betas,
        expression=expression,
        body_pose=body_pose,
        global_orient=global_orient,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=transl,
        return_verts=True
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    # Create trimesh object for export
    import trimesh
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
                               vertex_colors=vertex_colors)

    # Export to .glb if output path is specified
    if output_path:
        # Create exported_models directory if it doesn't exist
        export_dir = 'exported_models'
        os.makedirs(export_dir, exist_ok=True)

        # Get just the filename from the output path
        filename = osp.basename(output_path)
        # If no extension, add .glb
        if not filename.endswith('.glb'):
            filename = filename + '.glb' if '.' not in filename else filename

        # Save to exported_models folder
        full_output_path = osp.join(export_dir, filename)
        tri_mesh.export(full_output_path)
        print(f'Model exported to: {full_output_path}')

    # Visualization (optional, skip if output only and no plotting module specified)
    if plotting_module and plotting_module != 'none':
        if plotting_module == 'pyrender':
            import pyrender
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
        elif plotting_module == 'matplotlib':
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
            face_color = (1.0, 1.0, 0.9)
            edge_color = (0, 0, 0)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

            if plot_joints:
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
            plt.show()
        elif plotting_module == 'open3d':
            import open3d as o3d

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(
                vertices)
            mesh.triangles = o3d.utility.Vector3iVector(model.faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.3, 0.3, 0.3])

            geometry = [mesh]
            if plot_joints:
                joints_pcl = o3d.geometry.PointCloud()
                joints_pcl.points = o3d.utility.Vector3dVector(joints)
                joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
                geometry.append(joints_pcl)

            o3d.visualization.draw_geometries(geometry)
        else:
            raise ValueError(
                'Unknown plotting_module: {}'.format(plotting_module))


if __name__ == '__main__':
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
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d', 'none'],
                        help='The module to use for plotting the result (use "none" to skip visualization)')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for .glb file (e.g., model.glb)')
    parser.add_argument('--params-file', type=str, default=None,
                        help='Path to .smpl parameter file to load and edit')
    parser.add_argument('--body-fat', type=float, default=None,
                        help='Body fat percentage modifier (-1.0 to 1.0 range, experimental)')
    parser.add_argument('--lean-mass-arms', type=float, default=None,
                        help='Arms lean mass modifier (experimental)')
    parser.add_argument('--lean-mass-torso', type=float, default=None,
                        help='Torso lean mass modifier (experimental)')
    parser.add_argument('--lean-mass-legs', type=float, default=None,
                        help='Legs lean mass modifier (experimental)')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression
    output_path = args.output
    params_file = args.params_file
    body_fat = args.body_fat
    lean_mass_arms = args.lean_mass_arms
    lean_mass_torso = args.lean_mass_torso
    lean_mass_legs = args.lean_mass_legs

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour,
         output_path=output_path,
         params_file=params_file,
         body_fat=body_fat,
         lean_mass_arms=lean_mass_arms,
         lean_mass_torso=lean_mass_torso,
         lean_mass_legs=lean_mass_legs)


# paste this to run the script
# python examples/demo.py --model-folder "../../smpl_edit_experiment_v1/models" --gender="neutral" --output model6.glb --plotting-module none
