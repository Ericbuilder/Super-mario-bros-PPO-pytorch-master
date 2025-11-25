"""
@author: Viet Nguyen <nhviet1009@gmail.com>
Environment setup for Super Mario Bros (pure Gym v0.21 API)
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym import Wrapper
from gym.spaces import Box
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp


# ===== Monitor for video recording =====
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


# ===== Frame preprocessing =====
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame
    return np.zeros((1, 84, 84))


# ===== Custom Reward Wrapper =====
class CustomReward(Wrapper):
    def __init__(self, env, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.monitor = monitor
        self.curr_score = 0
        self.current_x = 40
        self.world = 1
        self.stage = 1

    def step(self, action):
        # Old Gym API: (obs, reward, done, info)
        state, reward, done, info = self.env.step(action)

        if self.monitor:
            self.monitor.record(state)

        state = process_frame(state)

        # Score-based reward
        reward += (info.get("score", 0) - self.curr_score) / 40.0
        self.curr_score = info.get("score", 0)

        # 👇 Forward distance reward
        x_pos = info.get("x_pos", 0)
        if x_pos > self.current_x:
            reward += (x_pos - self.current_x) * 0.01
        elif x_pos < self.current_x:
            reward -= (self.current_x - x_pos) * 0.005

        # Flag get / death
        if done:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50

        # Update world/stage
        if "world" in info and "stage" in info:
            self.world = info["world"]
            self.stage = info["stage"]

        # Hard-coded death zones
        if self.world == 7 and self.stage == 4:
            if (506 <= x_pos <= 832 and info["y_pos"] > 127) or \
               (832 < x_pos <= 1064 and info["y_pos"] < 80) or \
               (1113 < x_pos <= 1464 and info["y_pos"] < 191) or \
               (1579 < x_pos <= 1943 and info["y_pos"] < 191) or \
               (1946 < x_pos <= 1964 and info["y_pos"] >= 191) or \
               (1984 < x_pos <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or \
               (2114 < x_pos < 2440 and info["y_pos"] < 191) or \
               x_pos < self.current_x - 500:
                reward -= 50
                done = True
        elif self.world == 4 and self.stage == 4:
            if (x_pos <= 1500 and info["y_pos"] < 127) or \
               (1588 <= x_pos < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = x_pos
        return state, reward / 10.0, done, info

    def reset(self):
        state = self.env.reset()
        self.curr_score = 0
        self.current_x = 40
        return process_frame(state)


# ===== Frame Skipping =====
class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0.0
        last_states = []
        done = False
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip // 2:
                last_states.append(state)
            if done:
                break

        if done:
            self.reset()
            return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

        max_state = np.max(np.concatenate(last_states, axis=0), axis=0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], axis=0)
        return self.states[None, :, :, :].astype(np.float32)


# ===== Create Environment (continuous levels) =====
def create_train_env(actions, output_path=None):
    # ✅ Pure Gym v0.21 API — no gymnasium, no shimmy
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    monitor = Monitor(256, 240, output_path) if output_path else None
    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env


# ===== Worker for Multiprocessing =====
def worker(conn, action_type, output_path=None, worker_id=0):
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    output = output_path if worker_id == 0 else None
    env = create_train_env(actions, output_path=output)

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, r, done, info = env.step(data.item())
                conn.send((obs, r, done, info))
            elif cmd == "reset":
                obs = env.reset()
                conn.send(obs)
            else:
                raise NotImplementedError
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    finally:
        env.close()
        conn.close()


# ===== Multiple Environments Manager =====
class MultipleEnvironments:
    def __init__(self, action_type, num_envs, output_path=None):
        self.action_type = action_type
        self.num_envs = num_envs
        self.output_path = output_path

        if action_type == "right":
            self.num_actions = len(RIGHT_ONLY)
        elif action_type == "simple":
            self.num_actions = len(SIMPLE_MOVEMENT)
        else:
            self.num_actions = len(COMPLEX_MOVEMENT)
        self.num_states = 4

        self.agent_conns, env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        for idx in range(num_envs):
            p = mp.Process(
                target=worker,
                args=(env_conns[idx], action_type, output_path, idx)
            )
            p.daemon = True
            p.start()
            env_conns[idx].close()