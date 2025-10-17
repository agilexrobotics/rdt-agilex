import os
import fnmatch
import json
import math
import h5py
import yaml
import cv2
import numpy as np
import random
import pickle
import glob
from configs.state_vec import STATE_VEC_IDX_MAPPING

arm_names = ['pika_l', 'pika_r']


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


def calc_relative_pose(poses, num, dim):
    results = []
    for i in range(len(poses)):
        begin_matrix = create_transformation_matrix(poses[i][0], poses[i][1], poses[i][2], poses[i][3], poses[i][4], poses[i][5])
        result = []
        for j in range(0, num):
            end_matrix = create_transformation_matrix(poses[i][j * dim + 0], poses[i][j * dim + 1], poses[i][j * dim + 2],
                                                      poses[i][j * dim + 3], poses[i][j * dim + 4], poses[i][j * dim + 5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            result += [xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], poses[i][j * dim + 6]]
        results.append(result)
    return np.array(results)


def calc_pose_incre(base_pose, pose_data, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        pose_incre_data = []
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        end_matrix = create_transformation_matrix(pose_data[0][0], pose_data[0][1], pose_data[0][2],
                                                  pose_data[0][3], pose_data[0][4], pose_data[0][5])
        result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
        xyzrpy = matrix_to_xyzrpy(result_matrix)
        pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[0][6]])
        for i in range(1, len(pose_data)):
            begin_matrix = create_transformation_matrix(pose_data[i-1][0], pose_data[i-1][1], pose_data[i-1][2],
                                                        pose_data[i-1][3], pose_data[i-1][4], pose_data[i-1][5])
            end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
                                                      pose_data[i][3], pose_data[i][4], pose_data[i][5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
        return np.array(pose_incre_data)
    elif arm_end_pose_incre_mode == 1:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(pose_data)):
            end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
                                                      pose_data[i][3], pose_data[i][4], pose_data[i][5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6] - base_pose[6]])
        return np.array(pose_incre_data)
    elif arm_end_pose_incre_mode == 2:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(pose_data)):
            end_matrix = create_transformation_matrix(pose_data[i][0], pose_data[i][1], pose_data[i][2],
                                                      pose_data[i][3], pose_data[i][4], pose_data[i][5])
            result_matrix = np.dot(np.linalg.inv(begin_matrix), end_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], pose_data[i][6] - base_pose[6]])
        return np.array(pose_incre_data)


def calc_pose_incres(base_poses, pose_datas, num, dim, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        pose_incres = []
        for i in range(num):
            pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(pose_incres, axis=-1)
    elif arm_end_pose_incre_mode == 1:
        pose_incres = []
        for i in range(num):
            pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(pose_incres, axis=-1)
    elif arm_end_pose_incre_mode == 2:
        pose_incres = []
        for i in range(num):
            pose_incres.append(calc_pose_incre(base_poses[i * dim:(i + 1) * dim], pose_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(pose_incres, axis=-1)


def decode_pose_by_incre(base_pose, incre_data, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        pose_incre_data = []
        pose_incre_data.append([base_pose[0], base_pose[1], base_pose[2], base_pose[3], base_pose[4], base_pose[5], base_pose[6]])
        for i in range(len(incre_data)):
            begin_matrix = create_transformation_matrix(pose_incre_data[-1][0], pose_incre_data[-1][1], pose_incre_data[-1][2],
                                                        pose_incre_data[-1][3], pose_incre_data[-1][4], pose_incre_data[-1][5])
            incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
                                                        incre_data[i][3], incre_data[i][4], incre_data[i][5])
            result_matrix = np.dot(begin_matrix, incre_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
        return np.array(pose_incre_data[1:])
    elif arm_end_pose_incre_mode == 1:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(incre_data)):
            incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
                                                        incre_data[i][3], incre_data[i][4], incre_data[i][5])
            result_matrix = np.dot(begin_matrix, incre_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
        return np.array(pose_incre_data)
    elif arm_end_pose_incre_mode == 2:
        begin_matrix = create_transformation_matrix(base_pose[0], base_pose[1], base_pose[2],
                                                    base_pose[3], base_pose[4], base_pose[5])
        pose_incre_data = []
        for i in range(len(incre_data)):
            incre_matrix = create_transformation_matrix(incre_data[i][0], incre_data[i][1], incre_data[i][2],
                                                        incre_data[i][3], incre_data[i][4], incre_data[i][5])
            result_matrix = np.dot(begin_matrix, incre_matrix)
            xyzrpy = matrix_to_xyzrpy(result_matrix)
            pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], incre_data[i][6]])
            # pose_incre_data.append([xyzrpy[0], xyzrpy[1], xyzrpy[2], xyzrpy[3], xyzrpy[4], xyzrpy[5], base_pose[6] + incre_data[i][6]])
        return np.array(pose_incre_data)


def decode_pose_by_incres(base_poses, incre_datas, num, dim, arm_end_pose_incre_mode):
    if arm_end_pose_incre_mode == 0:
        poses = []
        for i in range(num):
            poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(poses, axis=-1)
    elif arm_end_pose_incre_mode == 1:
        poses = []
        for i in range(num):
            poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(poses, axis=-1)
    elif arm_end_pose_incre_mode == 2:
        poses = []
        for i in range(num):
            poses.append(decode_pose_by_incre(base_poses[i * dim:(i + 1) * dim], incre_datas[:, i * dim:(i + 1) * dim], arm_end_pose_incre_mode))
        return np.concatenate(poses, axis=-1)


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIRS = ["/home/agilex/data"]
        self.DATASET_NAME = "agilex"
        self.stats_file_path = os.path.join(os.path.dirname(HDF5_DIRS[0]), f'preprocessing_stats.pkl')
        
        self.file_paths = []
        # for root, _, files in os.walk(HDF5_DIR):
        #     print(files)
        #     for filename in fnmatch.filter(files, '*.hdf5'):
        #         file_path = os.path.join(root, filename)
        #         self.file_paths.append(file_path)
        # print(self.file_paths)
        for HDF5_DIR in HDF5_DIRS:
            for f in os.listdir(HDF5_DIR):
                if f.endswith(".hdf5"):
                    self.file_paths.append(os.path.join(HDF5_DIR, f))
                if os.path.isdir(os.path.join(HDF5_DIR, f)):
                    self.file_paths.extend(glob.glob(os.path.join(HDF5_DIR, f, "*.hdf5")))
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        
        # Load preprocessing settings
        self.state_preprocessing = config['common'].get('state_preprocessing', 'none')
        self.action_preprocessing = config['common'].get('action_preprocessing', 'none')
        
        # Try to load existing statistics first
        if self._load_statistics():
            print(f"Loaded preprocessing statistics: {self.stats_file_path}")
        else:
            print("Statistics file not found, starting to compute statistics...")
            self._compute_and_save_statistics()
        
        # Get each episode's len using the statistics
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def _compute_and_save_statistics(self):
        """Compute and save preprocessing statistics"""
        episode_lens = []
        all_states = []
        all_actions = []
        
        print("Iterating through dataset to compute statistics...")
        for i, file_path in enumerate(self.file_paths):
            if i % 10 == 0:
                print(f"Processing progress: {i}/{len(self.file_paths)}")
                
            valid, res = self.parse_hdf5_file_state_only(file_path)
            if valid:
                _len = res['state'].shape[0]
                episode_lens.append(_len)
                
                # Collect states and actions for statistics computation
                # We need to extract the actual state values (before filling into unified vector)
                with h5py.File(file_path, 'r') as f:
                    qpos = np.concatenate(
                        [np.concatenate([f[f'/localization/pose/{arm_name}'][()], 
                                       f[f'/gripper/encoderDistance/{arm_name}'][()].reshape(-1, 1)], axis=1)
                            for arm_name in arm_names], axis=1)
                    
                    # Skip the first few still steps
                    EPS = 1e-2
                    qpos_delta = np.abs(qpos - qpos[0:1])
                    indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
                    if len(indices) > 0:
                        first_idx = indices[0]
                        # Process qpos similar to parse_hdf5_file
                        processed_qpos = calc_relative_pose(qpos, 2, 7)
                        processed_qpos = processed_qpos[:, 6:] / np.array(
                           [[0.098, 1, 1, 1, 1, 1, 1, 0.098]] 
                        )
                        state_data = processed_qpos[first_idx-1:]
                        all_states.append(state_data)
                        
                        # For actions, compute incremental poses  
                        target_qpos_full = qpos[first_idx-1:]
                        for t in range(len(target_qpos_full) - self.CHUNK_SIZE + 1):
                            target_chunk = target_qpos_full[t:t+self.CHUNK_SIZE]
                            action_data = calc_pose_incres(target_chunk[0], target_chunk, 2, 7, 0)
                            action_data = action_data / np.array(
                               [[1, 1, 1, 1, 1, 1, 0.098, 1, 1, 1, 1, 1, 1, 0.098]] 
                            )
                            all_actions.append(action_data)
            else:
                episode_lens.append(0)
        
        # Compute global statistics for preprocessing
        if all_states and (self.state_preprocessing != 'none' or self.action_preprocessing != 'none'):
            # For states
            if all_states and self.state_preprocessing != 'none':
                all_states_concat = np.concatenate(all_states, axis=0)
                self.state_mean = np.mean(all_states_concat, axis=0)
                self.state_std = np.std(all_states_concat, axis=0)
                self.state_min = np.min(all_states_concat, axis=0)
                self.state_max = np.max(all_states_concat, axis=0)
                
                # Avoid division by zero
                self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)
                self.state_range = self.state_max - self.state_min
                self.state_range = np.where(self.state_range == 0, 1.0, self.state_range)
                
                # If normalize_then_standardize mode, compute statistics for normalized data
                if self.state_preprocessing == 'normalize_then_standardize':
                    normalized_states = (all_states_concat - self.state_min) / self.state_range * 2 - 1
                    self.state_norm_mean = np.mean(normalized_states, axis=0)
                    self.state_norm_std = np.std(normalized_states, axis=0)
                    self.state_norm_std = np.where(self.state_norm_std == 0, 1.0, self.state_norm_std)
                else:
                    self.state_norm_mean = np.zeros_like(self.state_mean)
                    self.state_norm_std = np.ones_like(self.state_std)
            else:
                self.state_mean = np.zeros(8)  # Default for 8-dim state
                self.state_std = np.ones(8)
                self.state_min = np.zeros(8)
                self.state_max = np.ones(8)
                self.state_range = np.ones(8)
                self.state_norm_mean = np.zeros(8)
                self.state_norm_std = np.ones(8)
            
            # For actions  
            if all_actions and self.action_preprocessing != 'none':
                all_actions_concat = np.concatenate(all_actions, axis=0)
                self.action_mean = np.mean(all_actions_concat, axis=0)
                self.action_std = np.std(all_actions_concat, axis=0)
                self.action_min = np.min(all_actions_concat, axis=0)
                self.action_max = np.max(all_actions_concat, axis=0)
                
                # Avoid division by zero
                self.action_std = np.where(self.action_std == 0, 1.0, self.action_std)
                self.action_range = self.action_max - self.action_min
                self.action_range = np.where(self.action_range == 0, 1.0, self.action_range)
                
                # If normalize_then_standardize mode, compute statistics for normalized data
                if self.action_preprocessing == 'normalize_then_standardize':
                    normalized_actions = (all_actions_concat - self.action_min) / self.action_range * 2 - 1
                    self.action_norm_mean = np.mean(normalized_actions, axis=0)
                    self.action_norm_std = np.std(normalized_actions, axis=0)
                    self.action_norm_std = np.where(self.action_norm_std == 0, 1.0, self.action_norm_std)
                else:
                    self.action_norm_mean = np.zeros_like(self.action_mean)
                    self.action_norm_std = np.ones_like(self.action_std)
            else:
                self.action_mean = np.zeros(14)  # Default for 14-dim action
                self.action_std = np.ones(14)
                self.action_min = np.zeros(14)
                self.action_max = np.ones(14)
                self.action_range = np.ones(14)
                self.action_norm_mean = np.zeros(14)
                self.action_norm_std = np.ones(14)
        else:
            # Initialize default values when no preprocessing is needed
            self.state_mean = np.zeros(8)
            self.state_std = np.ones(8)
            self.state_min = np.zeros(8)
            self.state_max = np.ones(8)
            self.state_range = np.ones(8)
            self.state_norm_mean = np.zeros(8)
            self.state_norm_std = np.ones(8)
            self.action_mean = np.zeros(14)
            self.action_std = np.ones(14)
            self.action_min = np.zeros(14)
            self.action_max = np.ones(14)
            self.action_range = np.ones(14)
            self.action_norm_mean = np.zeros(14)
            self.action_norm_std = np.ones(14)
        
        # Save statistics to file
        self.save_statistics_to_path(self.stats_file_path)
    
    def _load_statistics(self):
        """Load statistics from file"""
        if not os.path.exists(self.stats_file_path):
            return False
        
        try:
            with open(self.stats_file_path, 'rb') as f:
                stats_data = pickle.load(f)
            
            # Check if preprocessing settings match
            if (stats_data.get('state_preprocessing') != self.state_preprocessing or 
                stats_data.get('action_preprocessing') != self.action_preprocessing):
                print("Preprocessing settings have changed, need to re-compute statistics...")
                return False
            
            # Load statistics
            self.state_mean = stats_data['state_mean']
            self.state_std = stats_data['state_std']
            self.state_min = stats_data['state_min']
            self.state_max = stats_data['state_max']
            self.state_range = stats_data['state_range']
            self.state_norm_mean = stats_data.get('state_norm_mean', np.zeros_like(self.state_mean))
            self.state_norm_std = stats_data.get('state_norm_std', np.ones_like(self.state_std))
            self.action_mean = stats_data['action_mean']
            self.action_std = stats_data['action_std']
            self.action_min = stats_data['action_min']
            self.action_max = stats_data['action_max']
            self.action_range = stats_data['action_range']
            self.action_norm_mean = stats_data.get('action_norm_mean', np.zeros_like(self.action_mean))
            self.action_norm_std = stats_data.get('action_norm_std', np.ones_like(self.action_std))
            print("self.state_mean:", self.state_mean)
            print("self.state_std:", self.state_std)
            print("self.state_min:", self.state_min)
            print("self.state_max:", self.state_max)
            print("self.state_range:", self.state_range)
            print("self.state_norm_mean:", self.state_norm_mean)
            print("self.state_norm_std:", self.state_norm_std)
            print("self.action_mean:", self.action_mean)
            print("self.action_std:", self.action_std)
            print("self.action_min:", self.action_min)
            print("self.action_max:", self.action_max)
            print("self.action_range:", self.action_range)
            print("self.action_norm_mean:", self.action_norm_mean)
            print("self.action_norm_std:", self.action_norm_std)
            return True
            
        except Exception as e:
            print(f"Failed to load statistics: {str(e)}")
            return False
    
    def save_statistics_to_path(self, save_path):
        """Save statistics to specified path"""
        stats_data = {
            'state_preprocessing': self.state_preprocessing,
            'action_preprocessing': self.action_preprocessing,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_min': self.state_min,
            'state_max': self.state_max,
            'state_range': self.state_range,
            'state_norm_mean': getattr(self, 'state_norm_mean', np.zeros_like(self.state_mean)),
            'state_norm_std': getattr(self, 'state_norm_std', np.ones_like(self.state_std)),
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'action_min': self.action_min,
            'action_max': self.action_max,
            'action_range': self.action_range,
            'action_norm_mean': getattr(self, 'action_norm_mean', np.zeros_like(self.action_mean)),
            'action_norm_std': getattr(self, 'action_norm_std', np.ones_like(self.action_std)),
            'dataset_name': self.DATASET_NAME,
            'num_episodes': len(self.file_paths)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(stats_data, f)
        
        print(f"Statistics saved to: {save_path}")
    
    def preprocess_state(self, state_data):
        """Apply preprocessing to state data based on configuration."""
        if self.state_preprocessing == 'standardize':
            return (state_data - self.state_mean) / self.state_std
        elif self.state_preprocessing == 'normalize':
            return (state_data - self.state_min) / self.state_range * 2 - 1  # Scale to [-1, 1]
        elif self.state_preprocessing == 'normalize_then_standardize':
            # First normalize to [-1, 1], then standardize
            normalized = (state_data - self.state_min) / self.state_range * 2 - 1
            return (normalized - self.state_norm_mean) / self.state_norm_std
        else:  # 'none'
            return state_data
    
    def preprocess_action(self, action_data):
        """Apply preprocessing to action data based on configuration."""
        if self.action_preprocessing == 'standardize':
            return (action_data - self.action_mean) / self.action_std
        elif self.action_preprocessing == 'normalize':
            return (action_data - self.action_min) / self.action_range * 2 - 1  # Scale to [-1, 1]
        elif self.action_preprocessing == 'normalize_then_standardize':
            # First normalize to [-1, 1], then standardize
            normalized = (action_data - self.action_min) / self.action_range * 2 - 1
            return (normalized - self.action_norm_mean) / self.action_norm_std
        else:  # 'none'
            return action_data
    
    def postprocess_state(self, processed_state):
        """Reverse preprocessing of state data for inference."""
        if self.state_preprocessing == 'standardize':
            return processed_state * self.state_std + self.state_mean
        elif self.state_preprocessing == 'normalize':
            return (processed_state + 1) / 2 * self.state_range + self.state_min  # Reverse [-1, 1] scaling
        elif self.state_preprocessing == 'normalize_then_standardize':
            # First denormalize, then reverse normalization
            denormalized = processed_state * self.state_norm_std + self.state_norm_mean
            return (denormalized + 1) / 2 * self.state_range + self.state_min
        else:  # 'none'
            return processed_state
    
    def postprocess_action(self, processed_action):
        """Reverse preprocessing of action data for inference."""
        if self.action_preprocessing == 'standardize':
            return processed_action * self.action_std + self.action_mean
        elif self.action_preprocessing == 'normalize':
            return (processed_action + 1) / 2 * self.action_range + self.action_min  # Reverse [-1, 1] scaling
        elif self.action_preprocessing == 'normalize_then_standardize':
            # First denormalize, then reverse normalization
            denormalized = processed_action * self.action_norm_std + self.action_norm_mean
            return (denormalized + 1) / 2 * self.action_range + self.action_min
        else:  # 'none'
            return processed_action
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = np.concatenate(
                [np.concatenate([f[f'/localization/pose/{arm_name}'][()], f[f'/gripper/encoderDistance/{arm_name}'][()].reshape(-1, 1)], axis=1)
                    for arm_name in arm_names], axis=1)
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            
            # Load the instruction
            # instruction = np.load(f[f'instruction'][()], allow_pickle=True).item()
            # instructions_vectors = instruction[f'full_instruction/vector']
            # instruction_index = random.randint(0, len(instructions_vectors) - 1)
            # instruction = instructions_vectors[instruction_index]

            instructions_vectors = None
            instructions_input_ids = None
            instructions_attention_mask = None
            time = f[f'timestamp'][step_id]
            try:
                # instruction_dir = os.path.join(os.path.dirname(file_path), f[f'instruction'][()].decode('utf-8'))
                # instruction_root = np.load(instruction_dir, allow_pickle=True).item()
                for i in range(len(f[f'instructions/segment_instructions/start_time'])):
                    start_time = f[f'instructions/segment_instructions/start_time'][i]
                    end_time = f[f'instructions/segment_instructions/end_time'][i]
                    if start_time <= time < end_time:
                        instructions_vectors = f[f'instructions/segment_instructions/{start_time}-{end_time}/vector']
                        instructions_input_ids = f[f'instructions/segment_instructions/{start_time}-{end_time}/input_ids']
                        instructions_attention_mask = f[
                            f'instructions/segment_instructions/{start_time}-{end_time}/attention_mask']
                        for j in range(start_index + 1, len(f['timestamp'])):
                            if f['timestamp'][j] > end_time:
                                end_index = j - 1
                                if end_index == start_index:
                                    end_index = -1
                                break
                        break
                rand_list = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
                random.shuffle(rand_list)
                if rand_list[0] == 0 or instructions_vectors is None or end_index == -1:
                    instructions_vectors = f[f'instructions/full_instructions/vector']
                    instructions_input_ids = f[f'instructions/full_instructions/input_ids']
                    instructions_attention_mask = f[f'instructions/full_instructions/attention_mask']
                instruction_index = random.randint(0, len(instructions_vectors) - 1)
                instruction_vector = instructions_vectors[instruction_index]
                instruction_input_ids = instructions_input_ids[instruction_index]
                instruction_attention_mask = instructions_attention_mask[instruction_index]
                instruction = instruction_vector
            except:
                current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../scripts/instruction.npy")
                sample = np.load(current_path, allow_pickle=True).item()
                instruction_vector = sample['vector']
                instruction_input_ids = sample['input_ids']
                instruction_attention_mask = sample['attention_mask']
                instruction = instruction_vector

            # dir_path = os.path.dirname(file_path)
            # with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
            #     instruction_dict = json.load(f_instr)
            # # We have 1/3 prob to use original instruction,
            # # 1/3 to use simplified instruction,
            # # and 1/3 to use expanded instruction.
            # instruction_type = np.random.choice([
            #     'instruction', 'simplified_instruction', 'expanded_instruction'])
            # instruction = instruction_dict[instruction_type]
            # if isinstance(instruction, list):
            #     instruction = np.random.choice(instruction)
            # # You can also use precomputed language embeddings (recommended)
            # # instruction = "path/to/lang_embed.pt"
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Rescale gripper to [0, 1]
            target_qpos = qpos[step_id:step_id+self.CHUNK_SIZE]
            target_qpos = calc_pose_incres(target_qpos[0], target_qpos, 2, 7, 0)
            target_qpos = target_qpos / np.array(
               [[1, 1, 1, 1, 1, 1, 0.098, 1, 1, 1, 1, 1, 1, 0.098]] 
            )
            qpos = calc_relative_pose(qpos, 2, 7)
            qpos = qpos[:, 6:] / np.array(
               [[0.098, 1, 1, 1, 1, 1, 1, 0.098]] 
            )
            
            # Parse the state and action with preprocessing
            state = qpos[step_id:step_id+1]
            # Apply preprocessing to state
            state = self.preprocess_state(state)
            
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            
            # Apply preprocessing to actions
            actions = self.preprocess_action(actions)
            
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(np.zeros_like(actions[-1:]), (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING["left_gripper_open"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_x"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_y"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_z"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_roll"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_pitch"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_yaw"],
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            def fill_in_action(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"left_arm_eef_pos_incre_x"],
                    STATE_VEC_IDX_MAPPING[f"left_arm_eef_pos_incre_y"],
                    STATE_VEC_IDX_MAPPING[f"left_arm_eef_pos_incre_z"],
                    STATE_VEC_IDX_MAPPING[f"left_arm_eef_pos_incre_roll"],
                    STATE_VEC_IDX_MAPPING[f"left_arm_eef_pos_incre_pitch"],
                    STATE_VEC_IDX_MAPPING[f"left_arm_eef_pos_incre_yaw"],
                    STATE_VEC_IDX_MAPPING["left_gripper_open"],
                    STATE_VEC_IDX_MAPPING[f"right_arm_eef_pos_incre_x"],
                    STATE_VEC_IDX_MAPPING[f"right_arm_eef_pos_incre_y"],
                    STATE_VEC_IDX_MAPPING[f"right_arm_eef_pos_incre_z"],
                    STATE_VEC_IDX_MAPPING[f"right_arm_eef_pos_incre_roll"],
                    STATE_VEC_IDX_MAPPING[f"right_arm_eef_pos_incre_pitch"],
                    STATE_VEC_IDX_MAPPING[f"right_arm_eef_pos_incre_yaw"],
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            state_indicator = fill_in_action(np.ones_like(actions[0]))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_action(actions)
            
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    if key == 'None':
                        img = np.zeros((640, 480, 3), dtype=np.uint8)
                    else:
                        if f[f'/camera/color/{key}'].ndim == 1:
                            img = cv2.imread(os.path.join(os.path.dirname(file_path), f[f'/camera/color/{key}'][i].decode('utf-8')))
                        else:
                            img = f[f'/camera/color/{key}'][i]
                    imgs.append(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img('None')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('pikaDepthCamera_l')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('pikaDepthCamera_r')
            cam_right_wrist_mask = cam_high_mask.copy()
            # cam_left_fisheye_wrist = parse_img('pikaFisheyeCamera_l')
            # cam_left_fisheye_wrist_mask = cam_high_mask.copy()
            # cam_right_fisheye_wrist = parse_img('pikaFisheyeCamera_r')
            # cam_right_fisheye_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
                # "cam_left_fisheye_wrist": cam_left_fisheye_wrist,
                # "cam_left_fisheye_wrist_mask": cam_left_fisheye_wrist_mask,
                # "cam_right_fisheye_wrist": cam_right_fisheye_wrist,
                # "cam_right_fisheye_wrist_mask": cam_right_fisheye_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = np.concatenate(
                [np.concatenate([f[f'/localization/pose/{arm_name}'][()], f[f'/gripper/encoderDistance/{arm_name}'][()].reshape(-1, 1)], axis=1)
                    for arm_name in arm_names], axis=1)
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # Rescale gripper to [0, 1]
            # Parse the state and action
            qpos = calc_relative_pose(qpos, 2, 7)
            qpos = qpos[:, 6:] / np.array(
               [[0.098, 1, 1, 1, 1, 1, 1, 0.098]] 
            )

            state = qpos[first_idx-1:]
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING["left_gripper_open"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_x"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_y"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_z"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_roll"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_pitch"],
                    STATE_VEC_IDX_MAPPING[f"arm_eef_diff_0_yaw"],
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            
            # Return the resulting sample
            return True, {
                "state": state
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
