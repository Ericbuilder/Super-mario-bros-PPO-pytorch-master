"""
@author: Viet Nguyen <nhviet1009@gmail.com>
Test PPO on Super Mario Bros with video recording (Gym v0.21 compatible)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from src.model import PPO
import torch.nn.functional as F
import numpy as np
import cv2
import subprocess as sp


class Monitor:
    def __init__(self, width, height, saved_path):
        self.command = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24", "-r", "60", "-i", "-",
            "-an", "-vcodec", "mpeg4", saved_path
        ]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            self.pipe = None

    def record(self, image_array):
        if self.pipe and self.pipe.stdin:
            self.pipe.stdin.write(image_array.tostring())

    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame
    return np.zeros((1, 84, 84))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_type", type=str, default="simple", choices=["right", "simple", "complex"])
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--model_name", type=str, default="ppo_super_mario_bros_continuous",
                        help="Model to load: e.g., 'ppo_super_mario_bros_continuous', 'ppo_world1_stage1'")
    return parser.parse_args()


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Determine action space
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    os.makedirs(opt.output_path, exist_ok=True)
    video_path = os.path.join(opt.output_path, "test_video.mp4")

    # Create environment (continuous levels)
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    monitor = Monitor(256, 240, video_path)
    env = JoypadSpace(env, actions)

    class TestWrapper:
        def __init__(self, env, monitor):
            self.env = env
            self.monitor = monitor

        def reset(self):
            obs = self.env.reset()
            if self.monitor:
                self.monitor.record(obs)
            return obs

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            if self.monitor:
                self.monitor.record(obs)
            return obs, reward, done, info

        def render(self):
            self.env.render()

        def close(self):
            self.env.close()
            if self.monitor:
                self.monitor.close()

    env = TestWrapper(env, monitor)

    # Load model
    model_path = os.path.join(opt.saved_path, opt.model_name)
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        for f in os.listdir(opt.saved_path):
            print(f"  - {f}")
        return

    num_states = 4
    num_actions = len(actions)
    model = PPO(num_states, num_actions)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"🚀 Testing with model: {opt.model_name}")
    obs = env.reset()
    state = torch.from_numpy(process_frame(obs))
    if torch.cuda.is_available():
        state = state.cuda()

    step_count = 0
    done = False
    while not done:
        step_count += 1

        with torch.no_grad():
            logits, _ = model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()

        obs, reward, done, info = env.step(action)

        # Print level completion
        if info.get("flag_get", False):
            world = info.get("world", "?")
            stage = info.get("stage", "?")
            print(f"✅ Completed level {world}-{stage} at step {step_count}")

        env.render()

        state = torch.from_numpy(process_frame(obs))
        if torch.cuda.is_available():
            state = state.cuda()

        # Safety break
        if step_count > 10000:
            print("⚠️  Step limit reached. Ending episode.")
            break

    print(f"🎥 Video saved to {video_path}")
    env.close()


if __name__ == "__main__":
    opt = get_args()
    test(opt)