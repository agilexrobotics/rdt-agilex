#!/usr/bin/env python3
"""
Remote Inference Server - 远程推理服务端
不使用ROS，只负责接收数据并进行模型推理
"""

import torch
import numpy as np
import pickle
import argparse
import socket
import threading
import time
import yaml
from PIL import Image as PImage
import io
import random
import zlib  # 添加压缩支持
import struct  # 添加结构化数据支持

# 导入模型创建函数
from scripts.agilex_model import create_model

class InferenceServer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.setup_model()
        
    def setup_model(self):
        """初始化模型和语言嵌入"""
        print("Initializing model...")
        
        # 创建模型
        with open(self.args.config_path, "r") as fp:
            config = yaml.safe_load(fp)
        self.args.config = config
        
        self.model = create_model(
            args=self.args.config,
            dtype=torch.bfloat16,
            pretrained=self.args.pretrained_model_name_or_path,
            pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
            control_frequency=self.args.ctrl_freq,
        )
        
        print("Model initialized successfully!")

    def process_inference_request(self, data):
        """处理推理请求"""
        try:
            # 解包数据
            camera_color = data.get('camera_color', None)
            camera_depth = data.get('camera_depth', None)
            camera_point_cloud = data.get('camera_point_cloud', None)
            robot_state = data.get('robot_state', None)
            inference_delay = data.get('inference_delay', 4)
            max_guidance_weight = data.get('max_guidance_weight', 1.0)
            pos_lookahead_step = data.get('pos_lookahead_step', 64)
            test_embeds = data.get('instruction_vector', None)
                        
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
            test_embeds = torch.from_numpy(test_embeds).float().cuda()
            
            with torch.inference_mode():
                start_time = time.time()
                
                action = self.model.step(
                    proprio=proprio,
                    images=images,
                    text_embeds=test_embeds,
                )
                
                inference_time = time.time() - start_time
                print(f"Model inference time: {inference_time:.3f}s")
                
                return {
                    'action': action.cpu().numpy(),
                    'inference_time': inference_time,
                    'success': True
                }
                
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'action': None,
                'inference_time': 0,
                'success': False,
                'error': str(e)
            }

    def _recv_exact(self, client_socket, size):
        """精确接收指定大小的数据"""
        data = b''
        while len(data) < size:
            remaining = size - len(data)
            chunk = client_socket.recv(min(self.args.chunk_size, remaining))
            if not chunk:
                raise ConnectionError("Connection broken while receiving data")
            data += chunk
            
            # 减少进度打印频率
            if len(data) % (self.args.chunk_size * 16) == 0:  # 每16个chunk打印一次进度
                print(f"Received: {len(data)/1024:.2f} KB / {size/1024:.2f} KB")
        
        return data

    def _send_exact(self, client_socket, data):
        """精确发送数据"""
        data_length = len(data)
        bytes_sent = 0
        
        while bytes_sent < data_length:
            chunk = data[bytes_sent:bytes_sent + self.args.chunk_size]
            sent = client_socket.send(chunk)
            if sent == 0:
                raise ConnectionError("Socket connection broken")
            bytes_sent += sent
            
            # 减少进度打印频率
            if bytes_sent % (self.args.chunk_size * 16) == 0:  # 每16个chunk打印一次进度
                print(f"Sent: {bytes_sent/1024:.2f} KB / {data_length/1024:.2f} KB")

    def handle_client(self, client_socket, addr):
        """处理客户端连接"""
        print(f"Connected to client: {addr}")
        
        try:
            # Socket性能优化设置
            # 1. 禁用Nagle算法，减少小包延迟
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # 2. 设置socket缓冲区大小
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.args.socket_buffer_size)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.args.socket_buffer_size)
            
            # 3. 设置TCP_QUICKACK快速确认(Linux)
            try:
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            except:
                pass  # 非Linux系统可能不支持
            
            # 设置socket超时
            client_socket.settimeout(self.args.socket_timeout)
            
            print(f"Socket optimizations applied for client {addr}")
            print(f"Buffer size: {self.args.socket_buffer_size} bytes, chunk size: {self.args.chunk_size} bytes")
            
            while True:
                # 接收请求头部 (压缩标志 + 原始长度 + 压缩长度)
                header_data = self._recv_exact(client_socket, 9)  # 1+4+4字节
                compressed_flag, original_length, compressed_length = struct.unpack('!BII', header_data)
                
                print(f"Receiving request: compressed={bool(compressed_flag)}, "
                      f"original_size={original_length/1024/1024:.2f} MB, "
                      f"compressed_size={compressed_length/1024/1024:.2f} MB")
                
                # 接收请求数据
                recv_start = time.time()
                request_data = self._recv_exact(client_socket, compressed_length)
                recv_time = time.time() - recv_start
                print(f"Request received in {recv_time:.3f}s, speed: {compressed_length/1024/recv_time:.2f} KB/s")
                
                # 解压缩数据
                if compressed_flag:
                    decompress_start = time.time()
                    request_data = zlib.decompress(request_data)
                    print(f"Decompression time: {time.time() - decompress_start:.3f}s")
                
                # 反序列化数据
                deserial_start = time.time()
                try:
                    request_obj = pickle.loads(request_data)
                    print(f"Deserialization time: {time.time() - deserial_start:.3f}s")
                except Exception as e:
                    print(f"Failed to deserialize data: {e}")
                    # 发送错误响应
                    error_result = {
                        'action': None,
                        'inference_time': 0,
                        'success': False,
                        'error': f'Deserialization failed: {str(e)}'
                    }
                    self._send_response(client_socket, error_result)
                    continue
                
                # 处理推理请求
                result = self.process_inference_request(request_obj)
                
                # 发送响应
                self._send_response(client_socket, result)
                
        except socket.timeout:
            print(f"Client {addr} timeout")
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                client_socket.close()
            except:
                pass
            print(f"Client {addr} disconnected")

    def _send_response(self, client_socket, result):
        """发送响应数据"""
        try:
            # 序列化结果
            serial_start = time.time()
            result_data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Response serialization time: {time.time() - serial_start:.3f}s")
            
            # 根据配置决定是否压缩数据
            if self.args.use_compression:
                compress_start = time.time()
                compressed_data = zlib.compress(result_data, level=self.args.compression_level)
                compression_ratio = len(compressed_data) / len(result_data)
                print(f"Response compression time: {time.time() - compress_start:.3f}s, ratio: {compression_ratio:.3f}")
                
                # 发送压缩标志(1字节) + 原始长度(4字节) + 压缩长度(4字节) + 压缩数据
                header = struct.pack('!BII', 1, len(result_data), len(compressed_data))
                data_to_send = compressed_data
                print(f"Sending compressed response: {len(compressed_data)} bytes ({len(compressed_data)/1024:.2f} KB), "
                      f"original: {len(result_data)} bytes")
            else:
                # 发送未压缩标志(1字节) + 原始长度(4字节) + 数据长度(4字节) + 数据
                header = struct.pack('!BII', 0, len(result_data), len(result_data))
                data_to_send = result_data
                print(f"Sending uncompressed response: {len(result_data)} bytes ({len(result_data)/1024:.2f} KB)")
            
            # 发送头部
            client_socket.sendall(header)
            
            # 发送数据
            send_start = time.time()
            self._send_exact(client_socket, data_to_send)
            send_time = time.time() - send_start
            print(f"Response sent in {send_time:.3f}s, speed: {len(data_to_send)/1024/send_time:.2f} KB/s")
            
        except Exception as e:
            print(f"Error sending response: {e}")
            raise

    def start_server(self):
        """启动服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 服务器socket优化
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        server_socket.bind((self.args.host, self.args.port))
        server_socket.listen(5)
        
        print(f"Inference server started on {self.args.host}:{self.args.port}")
        print(f"Server configuration:")
        print(f"  - Compression: {self.args.use_compression}")
        print(f"  - Compression level: {self.args.compression_level}")
        print(f"  - Chunk size: {self.args.chunk_size} bytes")
        print(f"  - Socket buffer size: {self.args.socket_buffer_size} bytes")
        print(f"  - Socket timeout: {self.args.socket_timeout} seconds")
        print("Waiting for client connections...")
        
        try:
            while True:
                client_socket, addr = server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server_socket.close()

def get_arguments():
    parser = argparse.ArgumentParser()
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    
    # 网络传输优化参数
    parser.add_argument('--use_compression', action='store_true', default=True, help='Enable data compression')
    parser.add_argument('--socket_timeout', type=float, default=120.0, help='Socket timeout in seconds')
    parser.add_argument('--chunk_size', type=int, default=65536, help='Data chunk size for network transfer (bytes)')
    parser.add_argument('--socket_buffer_size', type=int, default=1048576, help='Socket buffer size (bytes)')
    parser.add_argument('--compression_level', type=int, default=1, help='Compression level (1-9, 1=fastest)')
    
    # 模型相关参数
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                        help='Name or path to the pretrained model')
    parser.add_argument('--pretrained_vision_encoder_name_or_path', type=str, required=True, 
                        help='Name or path to the vision model')
    parser.add_argument('--ctrl_freq', type=int, default=25,
                        help='The control frequency of the robot')
    
    return parser.parse_args()

def main():
    args = get_arguments()
    server = InferenceServer(args)
    server.start_server()

if __name__ == '__main__':
    main() 