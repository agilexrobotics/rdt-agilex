#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange
from collections import deque
import rospy
from std_msgs.msg import Header, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
import threading
from torchvision import transforms
import cv2
from functools import partial
import ros_numpy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys
from scipy.spatial import geometric_slerp
import open3d as o3d
import random
import tf
import time
from PIL import Image as PImage
import yaml
import math
# from aloha_msgs.srv import StatusSrv, StatusSrvRequest, StatusSrvResponse

# 导入模型创建函数
from scripts.agilex_model import create_model

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None
pre_instruction = ""
pre_instruction_attention_vector = None
pre_instruction_input_ids = None
pre_instruction_attention_mask = None


class BlockingDeque:
    def __init__(self):
        self.deque = deque()
        self.not_empty = threading.Condition()

    def append(self, item):
        with self.not_empty:
            self.deque.append(item)
            self.not_empty.notify()

    def popleft(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque.popleft()
        return item

    def left(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[0]
            return item

    def right(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[-1]
            return item

    def size(self):
        with self.not_empty:
            return len(self.deque)


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def depth_to_color_projection(depth_image, color_intrinsics, depth_intrinsics, extrinsics):
    # 获取深度图像的宽度和高度
    depth_height, depth_width = depth_image.shape[:2]

    # 创建网格坐标
    u, v = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
    u = u.flatten()
    v = v.flatten()
    depth_values = depth_image.flatten()

    # 将像素坐标转换为齐次坐标
    depth_points = np.vstack((u, v, np.ones_like(u)))

    # 将深度图像中的点转换到深度相机坐标系
    X_depth = np.linalg.inv(depth_intrinsics) @ depth_points

    # 将深度相机坐标系中的点转换到彩色相机坐标系
    X_color = extrinsics @ np.vstack((X_depth, np.ones((1, X_depth.shape[1]))))

    # 将彩色相机坐标系中的点投影到彩色图像平面
    x_color = (color_intrinsics[0, 0] * (X_color[0, :] / X_color[2, :]) + color_intrinsics[0, 2]).round().astype(int)
    y_color = (color_intrinsics[1, 1] * (X_color[1, :] / X_color[2, :]) + color_intrinsics[1, 2]).round().astype(int)

    # 创建对齐后的深度图像
    aligned_depth = np.zeros_like(depth_image)

    # 将投影后的点存储到对齐后的深度图像中
    valid_indices = (x_color >= 0) & (x_color < depth_image.shape[1]) & (y_color >= 0) & (y_color < depth_image.shape[0])
    aligned_depth[y_color[valid_indices], x_color[valid_indices]] = depth_values[valid_indices]

    return aligned_depth


def color_depth_to_point_cloud(color_image, depth_image, color_intrinsic, depth_intrinsic, color_extrinsic, depth_extrinsic):
    if not np.array_equal(color_extrinsic, depth_extrinsic):
        depth_image = depth_to_color_projection(depth_image, color_intrinsic, depth_intrinsic, np.dot(np.linalg.inv(color_extrinsic), depth_extrinsic))
        # 相机内参矩阵
        fx, fy = color_intrinsic[0][0], color_intrinsic[1][1]
        cx, cy = color_intrinsic[0][2], color_intrinsic[1][2]
    else:
        # 相机内参矩阵
        fx, fy = depth_intrinsic[0][0], depth_intrinsic[1][1]
        cx, cy = depth_intrinsic[0][2], depth_intrinsic[1][2]
    # 获取图像的宽度和高度
    height, width = depth_image.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    z = depth_image.astype(np.float32) / 1000.0  # 将深度图像转换为米

    # 计算 3D 坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 提取 color 颜色值
    b = color_image[..., 0].astype(np.float32)
    g = color_image[..., 1].astype(np.float32)
    r = color_image[..., 2].astype(np.float32)

    # 合并为点云
    point_cloud = np.stack((x, y, z, r, g, b), axis=-1)

    # 跳过深度为零的点
    valid_mask = z > 0.0
    point_cloud = point_cloud[valid_mask]

    return point_cloud


def get_camera_color(camera_names, history_num, camera_color):
    colorss = []
    for cam_name in camera_names:
        colors = []
        for i in range(history_num):
            color = camera_color[cam_name][i]
            color = cv2.imencode('.jpg', color)[1].tobytes()
            color = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
            default_width = 640
            default_height = 480
            camera_width = color.shape[1]
            camera_height = color.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                color = color[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                color = color[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            color = rearrange(color, 'h w c -> c h w')
            colors.append(color)
        colorss.append(np.stack(colors, axis=0))
    colorss = np.stack(colorss, axis=0)
    colorss = colorss.astype(np.float32)[np.newaxis, :]
    return colorss  # b, num_camera, num_history, c, h, w


def get_camera_depth(camera_names, history_num, camera_depth):
    depthss = []
    for cam_name in camera_names:
        depths = []
        for i in range(history_num):
            depth = camera_depth[cam_name][i]
            default_width = 640
            default_height = 480
            camera_width = depth.shape[1]
            camera_height = depth.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                depth = depth[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                depth = depth[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            depths.append(depth)
        depthss.append(np.stack(depths, axis=0))
    depthss = np.stack(depthss, axis=0)
    depthss = depthss.astype(np.float32)[np.newaxis, :]
    return depthss  # b, num_camera, num_history, h, w


def get_camera_point_cloud(camera_names, history_num, voxel_size, use_farthest_point_down_sample,
                           point_num, use_camera_point_cloud_rgb,
                           camera_point_cloud, color_extrinsics, point_cloud_extrinsics):
    camera_point_cloudss = []
    for cam_id, cam_name in enumerate(camera_names):
        camera_point_clouds = []
        for i in range(history_num):
            pc = camera_point_cloud[cam_name][i]

            if voxel_size != 0:
                condition = pc[:, 2] < 2
                pc = pc[condition, :]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
                if pc.shape[1] != 6:
                    rgbs = pc[:, 3].view(np.uint32)
                    # rgbs = pc[:, 3].view(np.uint64)
                    r = (np.right_shift(rgbs, 16) % 256)[:, np.newaxis]
                    g = (np.right_shift(rgbs, 8) % 256)[:, np.newaxis]
                    b = (rgbs % 256)[:, np.newaxis]
                    r_g_b = np.concatenate([r, g, b], axis=-1)
                    pcd.colors = o3d.utility.Vector3dVector(r_g_b.astype(np.float64))
                else:
                    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:].astype(np.float64))

                downsampled_cloud = pcd.voxel_down_sample(voxel_size)
                if use_farthest_point_down_sample and len(downsampled_cloud.points) > point_num:
                    downsampled_cloud = downsampled_cloud.farthest_point_down_sample(point_num)

                pc = np.concatenate([downsampled_cloud.points, downsampled_cloud.colors], axis=-1)
                if pc.shape[0] > point_num:
                    idxs = np.random.choice(pc.shape[0], point_num, replace=False)
                    pc = pc[idxs]
                elif pc.shape[0] < point_num:
                    if pc.shape[0] == 0:
                        pc = np.zeros([1, 4], dtype=np.float32)
                    idxs1 = np.arange(pc.shape[0])
                    idxs2 = np.random.choice(pc.shape[0], point_num - pc.shape[0], replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                    pc = pc[idxs]
            else:
                condition = pc[:, 2] < 2
                pc = pc[condition, :]
                if pc.shape[0] >= point_num:
                    idxs = np.random.choice(pc.shape[0], point_num, replace=False)
                elif pc.shape[0] < point_num:
                    idxs1 = np.arange(pc.shape[0])
                    idxs2 = np.random.choice(pc.shape[0], point_num - pc.shape[0], replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                if pc.shape[1] != 6:
                    # rgbs = pc[idxs][:, 3].view(np.uint64)
                    rgbs = pc[idxs][:, 3].view(np.uint32)
                    r = (np.right_shift(rgbs, 16) % 256)[:, np.newaxis]
                    g = (np.right_shift(rgbs, 8) % 256)[:, np.newaxis]
                    b = (rgbs % 256)[:, np.newaxis]
                    r_g_b = np.concatenate([r, g, b], axis=-1)
                    pc = np.concatenate([pc[idxs][:, :3], r_g_b], axis=-1)
                else:
                    pc = pc[idxs]

            if not np.array_equal(color_extrinsics[cam_id], point_cloud_extrinsics[cam_id]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
                pc[:, 3:] = pc[:, 3:] / 255
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
                pcd.transform(np.dot(np.linalg.inv(color_extrinsics[cam_id]), point_cloud_extrinsics[cam_id]))
                pcd.colors = o3d.utility.Vector3dVector(
                    (np.asarray(pcd.colors) * 255).astype(np.float64))
                pc = np.concatenate([pcd.points, pcd.colors], axis=-1)

            index = 6 if use_camera_point_cloud_rgb else 3
            camera_point_clouds.append(pc[:, :index])
        camera_point_cloudss.append(np.stack(camera_point_clouds, axis=0))
    camera_point_cloudss = camera_point_cloudss.astype(np.float32)[np.newaxis, :]
    return camera_point_cloudss  # b, num_camera, num_history, point_num, 6 or 3


def point_cloud_to_numpy(point_cloud):
    pc_data2 = ros_numpy.numpify(point_cloud)
    pc_x = pc_data2.flatten()[:]["x"]
    pc_y = pc_data2.flatten()[:]["y"]
    pc_z = pc_data2.flatten()[:]["z"]
    pc_rgb = pc_data2.flatten()[:]["rgb"]
    pc_array = np.vstack([pc_x, pc_y, pc_z, pc_rgb]).T
    return pc_array


def make_policy(args):
    """创建RDT模型"""
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model


def inference_process(args, t, ros_operator, policy):
    global inference_lock
    global inference_actions
    global inference_timestep
    global pre_instruction
    global pre_instruction_attention_vector
    global pre_instruction_input_ids
    global pre_instruction_attention_mask
    bridge = CvBridge()

    (instruction,
     camera_colors, camera_depths, camera_point_clouds,
     arm_joint_states, arm_end_poses,  robot_base_vels, last_ctrl_arm_end_poses) = ros_operator.get_frame()

    camera_color_dict = {}
    camera_color = None
    camera_depth_dict = {}
    camera_depth = None
    camera_point_cloud_dict = {}
    camera_point_cloud = None
    for j in range(len(args.camera_color_names)):
        camera_color_dict[args.camera_color_names[j]] = [bridge.imgmsg_to_cv2(camera_colors[i][j], 'rgb8') for i in range(args.obs_history_num)]
    if args.use_camera_color:
        camera_color = get_camera_color(args.camera_color_names, args.obs_history_num, camera_color_dict)  # b, num_camera, history, c, h, w

    if args.use_camera_color_depth_to_point_cloud or args.use_camera_depth:
        for j in range(len(args.camera_depth_names)):
            if np.array_equal(ros_operator.get_camera_color_extrinsic(j), ros_operator.get_camera_depth_extrinsic(j)):
                camera_depth_dict[args.camera_depth_names[j]] = [bridge.imgmsg_to_cv2(camera_depths[i][j], 'passthrough') for i in range(args.obs_history_num)]
            else:
                camera_depth_dict[args.camera_depth_names[j]] = [depth_to_color_projection(bridge.imgmsg_to_cv2(camera_depths[i][j], 'passthrough'),
                                                                                            ros_operator.get_camera_color_intrinsic(j), ros_operator.get_camera_depth_intrinsic(j),
                                                                                            np.dot(np.linalg.inv(ros_operator.get_camera_color_extrinsic(j)), ros_operator.get_camera_depth_extrinsic(j))) for i in range(args.obs_history_num)]
        if args.use_camera_depth:
            camera_depth = get_camera_depth(args.camera_depth_names, args.obs_history_num, camera_depth_dict)  # b, num_camera, num_history, h, w

        color_extrinsics = []
        point_cloud_extrinsics = []
        if args.use_camera_point_cloud:
            if not args.use_camera_color_depth_to_point_cloud:
                for j in range(len(args.camera_point_cloud_names)):
                    camera_point_cloud_dict[args.camera_point_cloud_names[j]] = [point_cloud_to_numpy(camera_point_clouds[i][j]) for i in range(args.obs_history_num)]
                    color_extrinsics.append(ros_operator.get_camera_color_extrinsic(j))
                    point_cloud_extrinsics.append(ros_operator.get_camera_point_cloud_extrinsic(j))
            else:
                for j in range(len(args.camera_point_cloud_names)):
                    color_intrinsic = ros_operator.get_camera_color_intrinsic(j)
                    depth_intrinsic = ros_operator.get_camera_color_intrinsic(j)
                    color_extrinsic = ros_operator.get_camera_color_extrinsic(j)
                    depth_extrinsic = ros_operator.get_camera_color_extrinsic(j)
                    color_extrinsics.append(color_extrinsic)
                    point_cloud_extrinsics.append(depth_extrinsic)
                    camera_point_cloud_dict[args.camera_point_cloud_names[j]] = [color_depth_to_point_cloud(camera_color_dict[args.camera_color_names[j]][i], camera_depth_dict[args.camera_depth_names[j]][i],
                                                                                                            color_intrinsic, depth_intrinsic,
                                                                                                            color_extrinsic, depth_extrinsic) for i in range(args.obs_history_num)]
            camera_point_cloud = get_camera_point_cloud(args.camera_point_cloud_names, args.obs_history_num,
                                                        args.camera_point_cloud_voxel_size, args.use_farthest_point_down_sample,
                                                        args.camera_point_cloud_point_num,
                                                        args.use_camera_point_cloud_rgb, camera_point_cloud_dict, 
                                                        color_extrinsics, point_cloud_extrinsics)  # b, num_camera, num_history, point_num, 6 or 3

    robot_state = []
    if args.use_arm_joint_state % 2 == 1:
        qpos_joint_state = [np.concatenate([np.array(arm_joint_states[i][j].position) for j in range(len(args.arm_joint_state_names))], axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_joint_state = np.concatenate(qpos_joint_state, axis=0)  # obs_history_num, num_arm * 7
        robot_state.append(qpos_joint_state)
    if args.use_arm_end_pose % 2 == 1:
        qpos_end_pose = [np.concatenate(
            [np.array([arm_end_poses[i][j].pose.position.x, arm_end_poses[i][j].pose.position.y, arm_end_poses[i][j].pose.position.z,
                        arm_end_poses[i][j].pose.orientation.x, arm_end_poses[i][j].pose.orientation.y, arm_end_poses[i][j].pose.orientation.z, arm_end_poses[i][j].pose.orientation.w])
                for j in range(len(args.arm_end_pose_names))],
            axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_end_pose = np.concatenate(qpos_end_pose, axis=0)  # obs_history_num, num_arm * 7
        robot_state.append(qpos_end_pose)
    if args.use_robot_base % 2 == 1:
        qpos_robot_base = [np.array([robot_base_vels[i][0].twist.twist.linear.x,
                                        robot_base_vels[i][0].twist.twist.linear.y,
                                        robot_base_vels[i][0].twist.twist.angular.z])[np.newaxis, :]
                            for i in range(args.obs_history_num)]
        qpos_robot_base = np.concatenate(qpos_robot_base, axis=0)  # obs_history_num, 3
        robot_state.append(qpos_robot_base)
    robot_state = np.concatenate(robot_state, axis=-1)[np.newaxis, :]  # b, obs_history_num, n
    
    # 准备图像数据，格式要与agilex_model.py中的step方法兼容
    # 需要准备PIL图像列表，顺序为[前一帧的图像, 当前帧的图像]
    images = [None, None, None]
    if camera_color is not None:
        # camera_color shape: b, num_camera, history, c, h, w
        camera_color = camera_color.squeeze(0)  # num_camera, history, c, h, w
        num_cameras = camera_color.shape[0]
        history_len = camera_color.shape[1]
        
        # 转换为PIL图像列表，顺序为[前一帧的3个相机, 当前帧的3个相机]
        for hist_idx in range(history_len):
            for cam_idx in range(num_cameras):
                # 转换为numpy数组并调整维度顺序
                img_tensor = camera_color[cam_idx, hist_idx]  # c, h, w
                img_array = img_tensor.transpose(1, 2, 0)  # h, w, c
                img_array = img_array.astype(np.uint8)
                # 转换为PIL图像
                pil_image = PImage.fromarray(img_array)
                images.append(pil_image)
        
        # 重新排列图像顺序 (如果需要)
        if len(images) >= 6:
            images[3], images[4], images[5] = images[4], images[5], images[3]
    
    # 准备proprioception数据 (最新的关节状态)
    proprio = robot_state[0, -1, :]  # 取最新的状态
    proprio = torch.from_numpy(proprio).float().cuda()
    test_embeds = torch.from_numpy(pre_instruction_attention_vector).float().cuda()
    start_time = time.time()
    
    action = policy.step(
        proprio=proprio,
        images=images,
        text_embeds=test_embeds,
    )
    
    print(f"Model inference time: {time.time() - start_time:.3f}s")
    
    inference_lock.acquire()
    inference_actions = action.cpu().numpy()  # 1, chunk_size, action_dim
    inference_timestep = t
    inference_lock.release()


def model_inference(args, ros_operator):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    global pre_instruction_attention_vector
    # instruction = np.load(self.args.lang_embeddings_path, allow_pickle=True).item()
    # instructions_vectors = instruction[f'full_instruction/vector']
    # instruction_index = random.randint(0, len(instructions_vectors) - 1)
    # pre_instruction_attention_vector = instructions_vectors[instruction_index]
    # pre_instruction_attention_vector = pre_instruction_attention_vector[np.newaxis, :]
    sample = np.load(self.args.lang_embeddings_path, allow_pickle=True).item()
    instruction_vector = sample['vector']
    instruction_input_ids = sample['input_ids']
    instruction_attention_mask = sample['attention_mask']
    pre_instruction_attention_vector = instruction_vector[np.newaxis, :]
    # 正确初始化policy
    policy = make_policy(args)

    joint_state0 = [[0, 0, 0, 0, 0, 0, 0.08] for _ in range(len(args.arm_joint_state_ctrl_topics))]
    joint_state1 = [[0, 0, 0, 0, 0, 0, 0.0] for _ in range(len(args.arm_joint_state_ctrl_topics))]
    end_pose0 = [[0.05, 0, 0.2, 0, 0, 0, 1.6] for _ in range(len(args.arm_end_pose_ctrl_topics))]
    end_pose1 = [[0.05, 0, 0.2, 0, 0, 0, 0.0] for _ in range(len(args.arm_end_pose_ctrl_topics))]
    robot_base_vel0 = [0, 0, 0]
    robot_base_vel1 = [0, 0, 0]
    if args.use_arm_joint_state > 1:
        ros_operator.arm_joint_state_ctrl_linear_interpolation_thread(joint_state0, True, calc_step=True)
    if args.use_arm_end_pose > 1:
        ros_operator.arm_end_pose_ctrl(end_pose0)
        ros_operator.arm_joint_state_ctrl(joint_state0)
        # ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(end_pose0, True, calc_step=True)
    if args.use_robot_base > 1:
        ros_operator.robot_base_vel_ctrl(robot_base_vel0)
    pre_inference_status = -1
    ctrl_rate = rospy.Rate(10)

    all_actions = None

    with torch.inference_mode():
        while not rospy.is_shutdown():
            t = 0
            max_t = 0
            if inference_thread is not None:
                inference_thread.join()
                inference_thread = None
            while t < args.max_publish_step and not rospy.is_shutdown():
                inference_status = ros_operator.get_inference_status()
                if inference_status == -1:
                    input("Please press any key to start inference:")
                    ros_operator.set_inference_status(0)
                is_new_action = False
                if args.asynchronous_inference:
                    if inference_thread is None and (args.pos_lookahead_step == 0 or t % args.pos_lookahead_step == 0 or t >= max_t) and ros_operator.check_frame():
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, t, ros_operator, policy))
                        inference_thread.start()

                    if inference_thread is not None and (not inference_thread.is_alive() or t >= max_t):
                        inference_thread.join()
                        inference_lock.acquire()
                        inference_thread = None
                        all_actions = inference_actions
                        t_start = inference_timestep
                        max_t = t_start + args.chunk_size
                        is_new_action = True
                        inference_lock.release()
                if t >= max_t:
                    print("inference time error")
                    continue
                raw_action = all_actions[:, t - t_start]
                raw_action = raw_action[0]
                if args.use_arm_joint_state > 1:
                    action_joint_state = raw_action[:args.arm_joint_state_dim * len(args.arm_joint_state_names)]
                    raw_action = raw_action[args.arm_joint_state_dim * len(args.arm_joint_state_names):]
                    action_joint_state = [action_joint_state[i*args.arm_joint_state_dim:(i+1)*args.arm_joint_state_dim] for i in range(len(args.arm_joint_state_names))]
                    ros_operator.arm_joint_state_ctrl_interpolation_thread(action_joint_state, args.blocking_publish, is_new_action if args.asynchronous_inference else False)
                if args.use_arm_end_pose > 1:
                    action_end_pose = raw_action[:args.arm_end_pose_dim * len(args.arm_end_pose_names)]
                    raw_action = raw_action[args.arm_end_pose_dim * len(args.arm_end_pose_names):]
                    action_end_pose = [action_end_pose[i*args.arm_end_pose_dim:(i+1)*args.arm_end_pose_dim] for i in range(len(args.arm_end_pose_names))]
                    ros_operator.arm_end_pose_ctrl_linear_interpolation_thread(action_end_pose, args.blocking_publish, calc_step=False)
                    # ros_operator.arm_end_pose_ctrl(action_end_pose)
                    # ctrl_rate.sleep()
                if args.use_robot_base > 1:
                    action_robot_base = raw_action[:args.robot_base_dim]
                    ros_operator.robot_base_vel_ctrl(action_robot_base)
                print("t:", t)
                t += 1


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        self.instruction = args.instruction
        self.camera_color_deques = [BlockingDeque() for _ in range(len(args.camera_color_names))]
        self.camera_depth_deques = [BlockingDeque() for _ in range(len(args.camera_depth_names))]
        self.camera_point_cloud_deques = [BlockingDeque() for _ in range(len(args.camera_point_cloud_names))]
        self.arm_joint_state_deques = [BlockingDeque() for _ in range(len(args.arm_joint_state_names))]
        self.arm_end_pose_deques = [BlockingDeque() for _ in range(len(args.arm_end_pose_names))]
        self.robot_base_vel_deques = [BlockingDeque() for _ in range(len(args.robot_base_vel_names))]

        self.camera_color_intrinsics = [None for _ in range(len(self.args.camera_color_names))]
        self.camera_depth_intrinsics = [None for _ in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_intrinsics = [None for _ in range(len(self.args.camera_point_cloud_names))]
        self.camera_color_extrinsics = [None for _ in range(len(self.args.camera_color_names))]
        self.camera_depth_extrinsics = [None for _ in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_extrinsics = [None for _ in range(len(self.args.camera_point_cloud_names))]

        self.all_config_exist = False

        self.camera_color_history_list = []
        self.camera_depth_history_list = []
        self.camera_point_cloud_history_list = []
        self.arm_joint_state_history_list = []
        self.arm_end_pose_history_list = []
        self.robot_base_vel_history_list = []

        rospy.init_node('inference', anonymous=True)
        self.instruction_subscriber = rospy.Subscriber(self.args.instruction_topic, String, self.instruction_callback)
        self.camera_color_subscriber = [rospy.Subscriber(self.args.camera_color_topics[i], Image, partial(self.camera_color_callback, i), queue_size=1) for i in range(len(self.args.camera_color_names))]
        self.camera_depth_subscriber = [rospy.Subscriber(self.args.camera_depth_topics[i], Image, partial(self.camera_depth_callback, i), queue_size=1) for i in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_subscriber = [rospy.Subscriber(self.args.camera_point_cloud_topics[i], PointCloud2, partial(self.camera_point_cloud_callback, i), queue_size=1) for i in range(len(self.args.camera_point_cloud_names))]
        self.arm_joint_state_subscriber = [rospy.Subscriber(self.args.arm_joint_state_topics[i], JointState, partial(self.arm_joint_state_callback, i), queue_size=1) for i in range(len(self.args.arm_joint_state_names))]
        self.arm_end_pose_subscriber = [rospy.Subscriber(self.args.arm_end_pose_topics[i], PoseStamped, partial(self.arm_end_pose_callback, i), queue_size=1) for i in range(len(self.args.arm_end_pose_names))]
        self.robot_base_vel_subscriber = [rospy.Subscriber(self.args.robot_base_vel_topics[i], Odometry, partial(self.robot_base_vel_callback, i), queue_size=1) for i in range(len(self.args.robot_base_vel_names))]

        # self.camera_color_config_subscriber = [rospy.Subscriber(self.args.camera_color_config_topics[i], CameraInfo, partial(self.camera_color_config_callback, i), queue_size=1) for i in range(len(self.args.camera_color_names))]
        # self.camera_depth_config_subscriber = [rospy.Subscriber(self.args.camera_depth_config_topics[i], CameraInfo, partial(self.camera_depth_config_callback, i), queue_size=1) for i in range(len(self.args.camera_depth_names))]
        # self.camera_point_cloud_config_subscriber = [rospy.Subscriber(self.args.camera_point_cloud_config_topics[i], CameraInfo, partial(self.camera_point_cloud_config_callback, i), queue_size=1) for i in range(len(self.args.camera_point_cloud_names))]

        self.arm_joint_state_ctrl_publisher = [rospy.Publisher(self.args.arm_joint_state_ctrl_topics[i], JointState, queue_size=1) for i in range(len(self.args.arm_joint_state_ctrl_topics))]
        self.arm_end_pose_ctrl_publisher = [rospy.Publisher(self.args.arm_end_pose_ctrl_topics[i], PoseStamped, queue_size=1) for i in range(len(self.args.arm_end_pose_ctrl_topics))]
        self.robot_base_vel_ctrl_publisher = rospy.Publisher(self.args.robot_base_vel_ctrl_topic, Twist, queue_size=1)

        # self.inference_status_service = rospy.Service(self.args.aloha_inference_status_service, StatusSrv, self.change_inference_status)
        self.inference_status = -1

        self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread_return_lock = threading.Lock()
        self.arm_joint_state_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_joint_state = None

        self.k = 3
        self.times = np.array([i for i in range(self.k)])
        self.arm_joint_state_ctrl_history_list = []

        self.arm_end_pose_ctrl_thread = None
        self.arm_end_pose_ctrl_thread_return_lock = threading.Lock()
        self.arm_end_pose_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_end_poses = []
        self.arm_end_pose_ctrl_history_list = []

    def get_camera_color_intrinsic(self, index):
        return self.camera_color_intrinsics[index]

    def get_camera_depth_intrinsic(self, index):
        return self.camera_depth_intrinsics[index]

    def get_camera_point_cloud_intrinsic(self, index):
        return self.camera_point_cloud_intrinsics[index]

    def get_camera_color_extrinsic(self, index):
        return self.camera_color_extrinsics[index]

    def get_camera_depth_extrinsic(self, index):
        return self.camera_depth_extrinsics[index]

    def get_camera_point_cloud_extrinsic(self, index):
        return self.camera_point_cloud_extrinsics[index]

    def instruction_callback(self, msg):
        self.instruction = msg.data

    def camera_color_callback(self, index, msg):
        if self.camera_color_deques[index].size() >= 200:
            self.camera_color_deques[index].popleft()
        self.camera_color_deques[index].append(msg)

    def camera_depth_callback(self, index, msg):
        if self.camera_depth_deques[index].size() >= 200:
            self.camera_depth_deques[index].popleft()
        self.camera_depth_deques[index].append(msg)

    def camera_point_cloud_callback(self, index, msg):
        if self.camera_point_cloud_deques[index].size() >= 200:
            self.camera_point_cloud_deques[index].popleft()
        self.camera_point_cloud_deques[index].append(msg)

    def arm_joint_state_callback(self, index, msg):
        if self.arm_joint_state_deques[index].size() >= 200:
            self.arm_joint_state_deques[index].popleft()
        self.arm_joint_state_deques[index].append(msg)

    def arm_end_pose_callback(self, index, msg):
        if self.arm_end_pose_deques[index].size() >= 200:
            self.arm_end_pose_deques[index].popleft()
        self.arm_end_pose_deques[index].append(msg)

    def robot_base_vel_callback(self, index, msg):
        if self.robot_base_vel_deques[index].size() >= 200:
            self.robot_base_vel_deques[index].popleft()
        self.robot_base_vel_deques[index].append(msg)

    def camera_color_config_callback(self, index, msg):
        self.camera_color_intrinsics[index] = np.array(msg.K).reshape(3, 3)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(self.args.camera_color_parent_frame_ids[index], msg.header.frame_id, rospy.Time(), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform(self.args.camera_color_parent_frame_ids[index], msg.header.frame_id, rospy.Time())
                rot = euler_from_quaternion(rot)
                self.camera_color_extrinsics[index] = create_transformation_matrix(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(f'Failed to get transform: {e}')
                continue
        self.camera_color_config_subscriber[index].unregister()

    def camera_depth_config_callback(self, index, msg):
        self.camera_depth_intrinsics[index] = np.array(msg.K).reshape(3, 3)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(self.args.camera_depth_parent_frame_ids[index], msg.header.frame_id, rospy.Time(), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform(self.args.camera_depth_parent_frame_ids[index], msg.header.frame_id, rospy.Time())
                rot = euler_from_quaternion(rot)
                self.camera_depth_extrinsics[index] = create_transformation_matrix(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(f'Failed to get transform: {e}')
                continue
        self.camera_depth_config_subscriber[index].unregister()

    def camera_point_cloud_config_callback(self, index, msg):
        self.camera_point_cloud_intrinsics[index] = np.array(msg.K).reshape(3, 3)
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(self.args.camera_point_cloud_parent_frame_ids[index], msg.header.frame_id, rospy.Time(), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform(self.args.camera_point_cloud_parent_frame_ids[index], msg.header.frame_id, rospy.Time())
                rot = euler_from_quaternion(rot)
                self.camera_point_cloud_extrinsics[index] = create_transformation_matrix(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(f'Failed to get transform: {e}')
                continue
        self.camera_point_cloud_config_subscriber[index].unregister()

    def interpolation_param(self, positions):
        positions = np.array(positions)
        # 构建矩阵A和向量b
        A = [np.ones_like(self.times)]
        for i in range(self.k - 1):
            A.append(self.times ** (i + 1))
        A = np.vstack(A).T
        b = positions
        # 解线性方程组得到多项式系数
        coeffs = np.linalg.solve(A, b)
        # 使用多项式系数计算给定时间的速度
        return coeffs

    def arm_joint_state_ctrl(self, joint_states):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [f'joint{i+1}' for i in range(self.args.arm_joint_state_dim)]
        self.last_ctrl_arm_joint_state = joint_states
        for i in range(len(joint_states)):
            joint_state_msg.position = joint_states[i]
            self.arm_joint_state_ctrl_publisher[i].publish(joint_state_msg)

    def arm_joint_state_ctrl_interpolation(self, joint_states, is_new_action):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        if is_new_action or max(joint_state_diff) > 0.5:
            append_to_history_list_step = 10
            hz = 200
            if max(joint_state_diff) > 0.5:
                step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            else:
                step = 50
            rate = rospy.Rate(hz)
            joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
            for i in range(1, len(joint_state_list)):
                if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                    return
                ctrl_joint_state = [joint_state_list[i][j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
                self.arm_joint_state_ctrl(ctrl_joint_state)
                if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                    self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
                rate.sleep()
            self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
            return

        if len(self.arm_joint_state_ctrl_history_list) == 0:
            for i in range(self.k):
                self.arm_joint_state_ctrl_history_list.append([last_ctrl_joint_state[j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))])
        self.arm_joint_state_ctrl_history_list.append(joint_states)
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        coeffs = [self.interpolation_param([self.arm_joint_state_ctrl_history_list[k][i][j] for k in range(self.k)]) for i in range(len(self.args.arm_joint_state_names)) for j in range(self.args.arm_joint_state_dim)]
        hz = 200
        step = 10
        rate = rospy.Rate(hz)
        for i in range(step):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [np.polyval(coeffs[j][::-1], (self.k - 2) + (i + 1) * (1.0 / step)) for j in range(len(coeffs))]
            self.arm_joint_state_ctrl([ctrl_joint_state[j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))])
            rate.sleep()

    def arm_joint_state_ctrl_linear_interpolation(self, joint_states, calc_step):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        hz = 200
        if calc_step:
            step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            step = 1 if step == 0 else step
        else:
            step = 10
        rate = rospy.Rate(hz)
        append_to_history_list_step = 10
        joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
        for i in range(1, len(joint_state_list)):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [joint_state_list[i][j * self.args.arm_joint_state_dim: (j + 1) * self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
            self.arm_joint_state_ctrl(ctrl_joint_state)
            if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
            rate.sleep()
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        return

    def arm_joint_state_ctrl_interpolation_thread(self, joint_states, block, is_new_action):
        if self.arm_joint_state_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_joint_state_ctrl_thread_return_lock.locked():
                self.arm_joint_state_ctrl_thread_return_lock.release()
            self.arm_joint_state_ctrl_thread.join()
            self.arm_joint_state_ctrl_thread_return_lock.acquire(False)
            self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread = threading.Thread(target=self.arm_joint_state_ctrl_interpolation,
                                                            args=(joint_states, is_new_action))
        self.arm_joint_state_ctrl_thread.start()
        if block:
            self.arm_joint_state_ctrl_thread.join()

    def arm_joint_state_ctrl_linear_interpolation_thread(self, joint_states, block, calc_step):
        if self.arm_joint_state_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_joint_state_ctrl_thread_return_lock.locked():
                self.arm_joint_state_ctrl_thread_return_lock.release()
            self.arm_joint_state_ctrl_thread.join()
            self.arm_joint_state_ctrl_thread_return_lock.acquire(False)
            self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread = threading.Thread(target=self.arm_joint_state_ctrl_linear_interpolation,
                                                            args=(joint_states, calc_step))
        self.arm_joint_state_ctrl_thread.start()
        if block:
            self.arm_joint_state_ctrl_thread.join()

    def arm_end_pose_ctrl(self, end_poses):
        # print("left:", left)
        # print("right:", right)
        end_pose_msg = PoseStamped()
        end_pose_msg.header = Header()
        end_pose_msg.header.frame_id = "map"
        end_pose_msg.header.stamp = rospy.Time.now()
        self.last_ctrl_arm_end_poses.append(end_poses)
        self.last_ctrl_arm_end_poses = self.last_ctrl_arm_end_poses[-self.args.obs_history_num:]
        for i in range(len(end_poses)):
            end_pose_msg.pose.position.x = end_poses[i][0]
            end_pose_msg.pose.position.y = end_poses[i][1]
            end_pose_msg.pose.position.z = end_poses[i][2]
            # q = quaternion_from_euler(end_poses[i][3], end_poses[i][4], end_poses[i][5])
            # end_pose_msg.pose.orientation.x = q[0]
            # end_pose_msg.pose.orientation.y = q[1]
            # end_pose_msg.pose.orientation.z = q[2]
            # end_pose_msg.pose.orientation.w = q[3]
            end_pose_msg.pose.orientation.x = end_poses[i][3]
            end_pose_msg.pose.orientation.y = end_poses[i][4]
            end_pose_msg.pose.orientation.z = end_poses[i][5]
            end_pose_msg.pose.orientation.w = end_poses[i][6]
            end_pose_msg.pose.orientation.w += self.args.gripper_offset[i]
            self.arm_end_pose_ctrl_publisher[i].publish(end_pose_msg)

    def arm_end_pose_ctrl_linear_interpolation(self, end_poses, calc_step):
        if len(self.last_ctrl_arm_end_poses) == 0:
            last_ctrl_end_pose = np.concatenate([
                np.array([self.arm_end_pose_deques[i].right().pose.position.x, self.arm_end_pose_deques[i].right().pose.position.y,
                          self.arm_end_pose_deques[i].right().pose.position.z,
                          self.arm_end_pose_deques[i].right().pose.orientation.x, self.arm_end_pose_deques[i].right().pose.orientation.y,
                          self.arm_end_pose_deques[i].right().pose.orientation.z, self.arm_end_pose_deques[i].right().pose.orientation.w])
                for i in range(len(self.args.arm_end_pose_names))], axis=0)
        else:
            last_ctrl_end_pose = np.concatenate(
                [np.array(end_poses) for end_poses in self.last_ctrl_arm_end_poses[-1]], axis=0)

        hz = 200
        if calc_step:
            ctrl_end_pose = np.concatenate([np.array(end_pose) for end_pose in end_poses], axis=0)
            end_pose_diff = ctrl_end_pose - last_ctrl_end_pose

            step_position = int(max([max(abs(end_pose_diff[i*self.args.arm_end_pose_dim: i*self.args.arm_end_pose_dim+3]) / np.array(self.args.arm_steps_length[:3])) for i in range(len(self.args.arm_end_pose_names))]))
            step_grasp = int(max([abs(end_pose_diff[(i+1)*self.args.arm_end_pose_dim-1]) / np.array(self.args.arm_steps_length[self.args.arm_end_pose_dim-1]) for i in range(len(self.args.arm_end_pose_names))]))
            step = max([step_grasp, step_position])
        else:
            step = 10
        rate = rospy.Rate(hz)

        ctrl_traj_xyzg = []
        ctrl_traj_xyzw = []
        for i in range(len(self.args.arm_end_pose_names)):
            ctrl_end_pose_xyzg = [end_poses[i][0], end_poses[i][1], end_poses[i][2], end_poses[i][6]]
            ctrl_end_pose_xyzw = quaternion_from_euler(end_poses[i][3], end_poses[i][4], end_poses[i][5])
            last_ctrl_end_pose_xyzg = [last_ctrl_end_pose[i*self.args.arm_end_pose_dim+0], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+1], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+2], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+6]]
            last_ctrl_end_pose_xyzw = quaternion_from_euler(last_ctrl_end_pose[i*self.args.arm_end_pose_dim+3], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+4], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+5])
            traj_xyzg = np.linspace(last_ctrl_end_pose_xyzg, ctrl_end_pose_xyzg, step + 1)[1:]
            traj_xyzw = [geometric_slerp(np.array(last_ctrl_end_pose_xyzw), np.array(ctrl_end_pose_xyzw), (j+1) / step) for j in range(step)]
            ctrl_traj_xyzg.append(traj_xyzg)
            ctrl_traj_xyzw.append(traj_xyzw)
        for i in range(step):
            if self.arm_end_pose_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_end_poses = []
            for j in range(len(self.args.arm_end_pose_names)):
                ctrl_rpy = euler_from_quaternion([ctrl_traj_xyzw[j][i][0], ctrl_traj_xyzw[j][i][1], ctrl_traj_xyzw[j][i][2], ctrl_traj_xyzw[j][i][3]])
                ctrl_end_pose = [ctrl_traj_xyzg[j][i][0], ctrl_traj_xyzg[j][i][1], ctrl_traj_xyzg[j][i][2],
                                 ctrl_rpy[0], ctrl_rpy[1], ctrl_rpy[2],
                                 ctrl_traj_xyzg[j][i][3]]
                ctrl_end_poses.append(ctrl_end_pose)
            self.arm_end_pose_ctrl(ctrl_end_poses)
            rate.sleep()

    def arm_end_pose_ctrl_linear_interpolation_thread(self, end_poses, block, calc_step):
        if self.arm_end_pose_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_end_pose_ctrl_thread_return_lock.locked():
                self.arm_end_pose_ctrl_thread_return_lock.release()
            self.arm_end_pose_ctrl_thread.join()
            self.arm_end_pose_ctrl_thread_return_lock.acquire(False)
            self.arm_end_pose_ctrl_thread = None
        self.arm_end_pose_ctrl_thread = threading.Thread(target=self.arm_end_pose_ctrl_linear_interpolation,
                                                         args=(end_poses, calc_step))
        self.arm_end_pose_ctrl_thread.start()
        if block:
            self.arm_end_pose_ctrl_thread.join()

    def robot_base_vel_ctrl(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[2]
        self.robot_base_vel_ctrl_publisher.publish(vel_msg)

    def get_frame(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown() and not self.all_config_exist:
            self.all_config_exist = True
            # if any(item is None for item in self.camera_color_intrinsics):
            #     self.all_config_exist = False
            #     print("camera color intrinsic config not exist")
            # if any(item is None for item in self.camera_depth_intrinsics):
            #     self.all_config_exist = False
            #     print("camera depth intrinsic config not exist")
            # if not self.args.use_camera_color_depth_to_point_cloud and any(item is None for item in self.camera_point_cloud_intrinsics):
            #     self.all_config_exist = False
            #     print("camera point cloud intrinsic config not exist")
            # if any(item is None for item in self.camera_color_extrinsics):
            #     self.all_config_exist = False
            #     print("camera color extrinsic config not exist")
            # if any(item is None for item in self.camera_depth_extrinsics):
            #     self.all_config_exist = False
            #     print("camera depth extrinsic config not exist")
            # if not self.args.use_camera_color_depth_to_point_cloud and any(item is None for item in self.camera_point_cloud_extrinsics):
            #     self.all_config_exist = False
            #     print("camera point cloud extrinsic config not exist")
            rate.sleep()

        camera_colors = [self.camera_color_deques[i].right() for i in range(len(self.args.camera_color_names))]
        camera_depths = [self.camera_depth_deques[i].right() for i in range(len(self.args.camera_depth_names))]
        camera_point_clouds = [self.camera_point_cloud_deques[i].right() for i in range(len(self.args.camera_point_cloud_names))] if not self.args.use_camera_color_depth_to_point_cloud else []
        arm_joint_states = [self.arm_joint_state_deques[i].right() for i in range(len(self.args.arm_joint_state_names))]
        arm_end_poses = [self.arm_end_pose_deques[i].right() for i in range(len(self.args.arm_end_pose_names))]
        robot_base_vels = [self.robot_base_vel_deques[i].right() for i in range(len(self.args.robot_base_vel_names))]
        frame_time = max([msg.header.stamp.to_sec() for msg in (camera_colors + camera_depths + camera_point_clouds +
                                                                arm_joint_states + arm_end_poses + robot_base_vels)])
        for i in range(len(self.args.camera_color_names)):
            closer_time_diff = math.inf
            while (self.camera_color_deques[i].size() > 0 and
                   abs(self.camera_color_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.camera_color_deques[i].left().header.stamp.to_sec() - frame_time)
                camera_colors[i] = self.camera_color_deques[i].popleft()
        for i in range(len(self.args.camera_depth_names)):
            closer_time_diff = math.inf
            while (self.camera_depth_deques[i].size() > 0 and
                   abs(self.camera_depth_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.camera_depth_deques[i].left().header.stamp.to_sec() - frame_time)
                camera_depths[i] = self.camera_depth_deques[i].popleft()
        if not self.args.use_camera_color_depth_to_point_cloud:
            for i in range(len(self.args.camera_point_cloud_names)):
                closer_time_diff = math.inf
                while (self.camera_point_cloud_deques[i].size() > 0 and
                    abs(self.camera_point_cloud_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                    closer_time_diff = abs(self.camera_point_cloud_deques[i].left().header.stamp.to_sec() - frame_time)
                    camera_point_clouds[i] = self.camera_point_cloud_deques[i].popleft()
        for i in range(len(self.args.arm_joint_state_names)):
            closer_time_diff = math.inf
            while (self.arm_joint_state_deques[i].size() > 0 and
                   abs(self.arm_joint_state_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.arm_joint_state_deques[i].left().header.stamp.to_sec() - frame_time)
                arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()
        for i in range(len(self.args.arm_end_pose_names)):
            closer_time_diff = math.inf
            while (self.arm_end_pose_deques[i].size() > 0 and
                   abs(self.arm_end_pose_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.arm_end_pose_deques[i].left().header.stamp.to_sec() - frame_time)
                arm_end_poses[i] = self.arm_end_pose_deques[i].popleft()
        for i in range(len(self.args.robot_base_vel_names)):
            closer_time_diff = math.inf
            while (self.robot_base_vel_deques[i].size() > 0 and
                   abs(self.robot_base_vel_deques[i].left().header.stamp.to_sec() - frame_time) < closer_time_diff):
                closer_time_diff = abs(self.robot_base_vel_deques[i].left().header.stamp.to_sec() - frame_time)
                robot_base_vels[i] = self.robot_base_vel_deques[i].popleft()

        # for i in range(len(self.args.camera_color_names)):
        #     while self.camera_color_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.camera_color_deques[i].popleft()
        #     camera_colors[i] = self.camera_color_deques[i].popleft()
        # for i in range(len(self.args.camera_depth_names)):
        #     while self.camera_depth_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.camera_depth_deques[i].popleft()
        #     camera_depths[i] = self.camera_depth_deques[i].popleft()
        # if not self.args.use_camera_color_depth_to_point_cloud:
        #     for i in range(len(self.args.camera_point_cloud_names)):
        #         while self.camera_point_cloud_deques[i].left().header.stamp.to_sec() < frame_time:
        #             self.camera_point_cloud_deques[i].popleft()
        #         camera_point_clouds[i] = self.camera_point_cloud_deques[i].popleft()
        # for i in range(len(self.args.arm_joint_state_names)):
        #     while self.arm_joint_state_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.arm_joint_state_deques[i].popleft()
        #     arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()
        # for i in range(len(self.args.arm_end_pose_names)):
        #     while self.arm_end_pose_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.arm_end_pose_deques[i].popleft()
        #     arm_end_poses[i] = self.arm_end_pose_deques[i].popleft()
        # for i in range(len(self.args.robot_base_vel_names)):
        #     while self.robot_base_vel_deques[i].left().header.stamp.to_sec() < frame_time:
        #         self.robot_base_vel_deques[i].popleft()
        #     robot_base_vels[i] = self.robot_base_vel_deques[i].popleft()

        if len(self.camera_color_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list = self.camera_color_history_list[-self.args.obs_history_num:]

        if len(self.camera_depth_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_depth_history_list.append(camera_depths)
        self.camera_depth_history_list.append(camera_depths)
        self.camera_depth_history_list = self.camera_depth_history_list[-self.args.obs_history_num:]

        if len(self.camera_point_cloud_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_point_cloud_history_list.append(camera_point_clouds)
        self.camera_point_cloud_history_list.append(camera_point_clouds)
        self.camera_point_cloud_history_list = self.camera_point_cloud_history_list[-self.args.obs_history_num:]

        if len(self.arm_joint_state_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list = self.arm_joint_state_history_list[-self.args.obs_history_num:]

        if len(self.arm_end_pose_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_end_pose_history_list.append(arm_end_poses)
        self.arm_end_pose_history_list.append(arm_end_poses)
        self.arm_end_pose_history_list = self.arm_end_pose_history_list[-self.args.obs_history_num:]

        if len(self.robot_base_vel_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.robot_base_vel_history_list.append(robot_base_vels)
        self.robot_base_vel_history_list.append(robot_base_vels)
        self.robot_base_vel_history_list = self.robot_base_vel_history_list[-self.args.obs_history_num:]

        return (self.instruction,
                self.camera_color_history_list, self.camera_depth_history_list, self.camera_point_cloud_history_list,
                self.arm_joint_state_history_list, self.arm_end_pose_history_list,
                self.robot_base_vel_history_list, self.last_ctrl_arm_end_poses)

    def check_frame(self):
        for i in range(len(self.args.camera_color_names)):
            if self.camera_color_deques[i].size() == 0:
                print(self.args.camera_color_topics[i], "has no data")
                return False
        for i in range(len(self.args.camera_depth_names)):
            if self.camera_depth_deques[i].size() == 0:
                print(self.args.camera_depth_topics[i], "has no data")
                return False
        if not self.args.use_camera_color_depth_to_point_cloud:
            for i in range(len(self.args.camera_point_cloud_names)):
                if self.camera_point_cloud_deques[i].size() == 0:
                    print(self.args.camera_point_cloud_topics[i], "has no data")
                    return False
        for i in range(len(self.args.arm_joint_state_names)):
            if self.arm_joint_state_deques[i].size() == 0:
                print(self.args.arm_joint_state_topics[i], "has no data")
                return False
        for i in range(len(self.args.arm_end_pose_names)):
            if self.arm_end_pose_deques[i].size() == 0:
                print(self.args.arm_end_pose_topics[i], "has no data")
                return False
        for i in range(len(self.args.robot_base_vel_names)):
            if self.robot_base_vel_deques[i].size() == 0:
                print(self.args.robot_base_vel_topics[i], "has no data")
                return False
        return True

    def change_inference_status(self, request):
        response = StatusSrvResponse()
        self.inference_status = request.status
        return response

    def get_inference_status(self):
        return self.inference_status

    def set_inference_status(self, status):
        self.inference_status = status


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000,
                        required=False)

    # 模型相关参数
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                        help='Name or path to the pretrained model')
    parser.add_argument('--pretrained_vision_encoder_name_or_path', type=str, required=True, 
                        help='Name or path to the vision model')
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')

    parser.add_argument('--instruction_topic', action='store', type=str, help='instruction_topic',
                        default='/instruction', required=False)

    parser.add_argument('--camera_color_names', action='store', type=str, help='camera_color_names',
                        default=['left', 'front', 'right'],
                        required=False)
    parser.add_argument('--camera_color_parent_frame_ids', action='store', type=str, help='camera_color_parent_frame_ids',
                        default=['camera_l_link', 'camera_f_link', 'camera_r_link'],
                        required=False)
    parser.add_argument('--camera_color_topics', action='store', type=str, help='camera_color_topics',
                        default=['/camera_l/color/image_raw', '/camera_f/color/image_raw', '/camera_r/color/image_raw'],
                        required=False)
    parser.add_argument('--camera_color_config_topics', action='store', type=str, help='camera_color_config_topics',
                        default=[],
                        required=False)
    parser.add_argument('--camera_depth_names', action='store', type=str, help='camera_depth_names',
                        default=[],
                        required=False)
    parser.add_argument('--camera_depth_parent_frame_ids', action='store', type=str, help='camera_depth_parent_frame_ids',
                        default=[],
                        required=False)
    parser.add_argument('--camera_depth_topics', action='store', type=str, help='camera_depth_topics',
                        default=[],
                        required=False)
    parser.add_argument('--camera_depth_config_topics', action='store', type=str, help='camera_depth_config_topics',
                        default=[],
                        required=False)
    parser.add_argument('--use_camera_color_depth_to_point_cloud', action='store', type=bool, help='use_camera_color_depth_to_point_cloud',
                        default=True,
                        required=False)
    parser.add_argument('--camera_point_cloud_names', action='store', type=str, help='camera_point_cloud_names',
                        default=[],
                        required=False)
    parser.add_argument('--camera_point_cloud_parent_frame_ids', action='store', type=str, help='camera_point_cloud_parent_frame_ids',
                        default=[],
                        required=False)
    parser.add_argument('--camera_point_cloud_topics', action='store', type=str, help='camera_point_cloud_topics',
                        default=[],
                        required=False)
    parser.add_argument('--camera_point_cloud_config_topics', action='store', type=str, help='camera_point_cloud_config_topics',
                        default=[],
                        required=False)
    parser.add_argument('--arm_joint_state_names', action='store', type=str, help='arm_joint_state_names',
                        default=['left', 'right'],
                        required=False)
    parser.add_argument('--arm_joint_state_topics', action='store', type=str, help='arm_joint_state_topics',
                        default=['/puppet/joint_left', '/puppet/joint_right'],
                        required=False)
    parser.add_argument('--arm_end_pose_names', action='store', type=str, help='arm_end_pose_names',
                        default=[],
                        required=False)
    parser.add_argument('--arm_end_pose_topics', action='store', type=str, help='arm_end_pose_topics',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_names', action='store', type=str, help='robot_base_vel_names',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_topics', action='store', type=str, help='robot_base_vel_topics',
                        default=[],
                        required=False)
    parser.add_argument('--arm_joint_state_ctrl_topics', action='store', type=str, help='arm_joint_state_ctrl_topics',
                        default=['/master/joint_left', '/master/joint_right'],
                        required=False)
    parser.add_argument('--arm_end_pose_ctrl_topics', action='store', type=str, help='arm_end_pose_ctrl_topics',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_ctrl_topic', action='store', type=str, help='robot_base_vel_ctrl_topic',
                        default='/cmd_vel',
                        required=False)
    parser.add_argument('--gripper_offset', nargs='+', action='store', type=float, help='gripper_offset', default=[0], required=False)

    parser.add_argument('--use_camera_color', action='store', type=bool, help='use_camera_color', default=True, required=False)
    parser.add_argument('--use_camera_depth', action='store', type=bool, help='use_camera_depth', default=False, required=False)
    parser.add_argument('--use_camera_point_cloud', action='store', type=bool, help='use_camera_point_cloud', default=False, required=False)
    parser.add_argument('--use_camera_point_cloud_rgb', action='store', type=bool, help='use_camera_point_cloud_rgb', default=True, required=False)
    parser.add_argument('--use_robot_base', action='store', type=int, help='use_robot_base', default=0, required=False)
    parser.add_argument('--robot_base_dim', action='store', type=int, help='robot_base_dim', default=3, required=False)
    parser.add_argument('--use_arm_joint_state', action='store', type=int, help='use_arm_joint_state', default=3, required=False)
    parser.add_argument('--arm_joint_state_dim', action='store', type=int, help='arm_joint_state_dim', default=7, required=False)
    parser.add_argument('--use_arm_end_pose', action='store', type=int, help='use_arm_end_pose', default=0, required=False)
    parser.add_argument('--arm_end_pose_dim', action='store', type=int, help='arm_end_pose_dim', default=7, required=False)

    parser.add_argument('--obs_history_num', action='store', type=int, help='obs_history_num', default=1, required=False)
    parser.add_argument('--use_instruction', action='store', type=bool, help='use_instruction', default=False, required=False)
    parser.add_argument('--instruction', action='store', type=str, help='instruction',
                        default='null', required=False)
    parser.add_argument('--camera_point_cloud_point_num', action='store', type=int, help='camera_point_cloud_point_num', default=5000, required=False)
    parser.add_argument('--camera_point_cloud_voxel_size', action='store', type=float, help='camera_point_cloud_voxel_size', default=0.01, required=False)
    parser.add_argument('--use_farthest_point_down_sample', action='store', type=bool, help='use_farthest_point_down_sample', default=False, required=False)

    parser.add_argument('--aloha_inference_status_service', action='store', type=str,
                        help='aloha_inference_status_service',
                        default='/aloha/inference_status_service', required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=32, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.03], required=False)
    parser.add_argument('--robot_base_steps_length', action='store', type=float, help='robot_base_steps_length',
                        default=[0.1, 0.1], required=False)
    parser.add_argument('--asynchronous_inference', action='store', type=bool, help='asynchronous_inference',
                        default=True, required=False)
    parser.add_argument('--preemptive_publishing', action='store', type=bool, help='preemptive_publishing',
                        default=False, required=False)
    parser.add_argument('--blocking_publish', action='store', type=bool, help='blocking_publish',
                        default=True, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    model_inference(args, ros_operator)


if __name__ == '__main__':
    main()
