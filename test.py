import os
os.environ['OMP_NUM_THREADS'] = '1'
import src.headless 

import argparse
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from src.model import PPO
import torch.nn.functional as F
import numpy as np
import cv2
import subprocess as sp

# 保持 Monitor 类不变...
class Monitor:
    def __init__(self, width, height, saved_path):
        self.command = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", "60", 
            "-i", "-", "-an", "-vcodec", "mpeg4", saved_path
        ]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            self.pipe = None

    def record(self, image_array):
        if self.pipe and self.pipe.stdin:
            self.pipe.stdin.write(image_array.tobytes())

    def close(self):
        if self.pipe:
            try:
                self.pipe.stdin.close()
                self.pipe.wait()
            except Exception:
                pass

# [优化1: 同步更新]
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        # 保持 uint8
        return frame[None, :, :].astype(np.uint8)
    return np.zeros((1, 84, 84), dtype=np.uint8)

def get_args():
    parser = argparse.ArgumentParser()
    # [优化4: 默认 simple]
    parser.add_argument("--action_type", type=str, default="simple", choices=["right", "simple", "complex"])
    parser.add_argument("--saved_path", type=str, default="/kaggle/working")
    parser.add_argument("--output_path", type=str, default="/kaggle/working/output")
    parser.add_argument("--model_name", type=str, default="ppo_mario_simple_1_1.pth")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    return parser.parse_args()

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    os.makedirs(opt.output_path, exist_ok=True)
    video_path = os.path.join(opt.output_path, f"test_w{opt.world}_s{opt.stage}.mp4")
    env_id = f"SuperMarioBros-{opt.world}-{opt.stage}-v3"
    env = gym_super_mario_bros.make(env_id)
    monitor = Monitor(256, 240, video_path)
    env = JoypadSpace(env, actions)

    # 简单的 TestWrapper 用于录制
    class TestWrapper:
        def __init__(self, env, monitor):
            self.env = env
            self.monitor = monitor
        def reset(self):
            obs = self.env.reset()
            if self.monitor: self.monitor.record(obs)
            return obs
        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            if self.monitor: self.monitor.record(obs)
            return obs, reward, done, info
        def close(self):
            self.env.close()
            if self.monitor: self.monitor.close()

    env = TestWrapper(env, monitor)

    # 加载模型
    model_path = os.path.join(opt.saved_path, opt.model_name)
    num_states = 4
    num_actions = len(actions)
    model = PPO(num_states, num_actions)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    obs = env.reset()
    # 注意: process_frame 返回的是单帧 uint8 (1, 84, 84)
    # 测试时我们需要 stack 4帧。这里简化处理，直接复制4次作为初始状态
    frame = process_frame(obs)
    state = np.concatenate([frame for _ in range(4)], axis=0)
    state = torch.from_numpy(state).unsqueeze(0) # (1, 4, 84, 84)

    if torch.cuda.is_available():
        state = state.cuda()

    done = False
    step_count = 0
    while not done:
        step_count += 1
        with torch.no_grad():
            logits, _ = model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()

        obs, reward, done, info = env.step(action)
        
        # 更新状态：滚动
        next_frame = process_frame(obs)
        next_frame_torch = torch.from_numpy(next_frame).unsqueeze(0)
        if torch.cuda.is_available():
            next_frame_torch = next_frame_torch.cuda()
        
        # 移除最早的一帧，加入新的一帧
        state = torch.cat((state[:, 1:, :, :], next_frame_torch), dim=1)

        if info.get("flag_get", False):
            print("✅ Level Cleared!")
        
        if step_count > 5000: break

    print(f"🎥 Video saved to {video_path}")
    env.close()

if __name__ == "__main__":
    opt = get_args()
    test(opt)