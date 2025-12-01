import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym import Wrapper
from gym.spaces import Box
import cv2
import numpy as np
import torch.multiprocessing as mp


# ===== Frame preprocessing =====
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame.astype(np.float32)
    return np.zeros((1, 84, 84), dtype=np.float32)


# ===== Custom Reward Wrapper =====
class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = 1
        self.stage = 1
        self.visited_positions = set()
        self.stay_counter = 0
        self.last_x = 40

    def get_level_type(self, world, stage):
        if stage == 4:
            return "maze"
        return "linear"

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)

        level_type = self.get_level_type(self.world, self.stage)
        x_pos = info.get("x_pos", 0)
        y_pos = info.get("y_pos", 0)
        score = info.get("score", 0)

        # stay-in-place penalty
        if x_pos == self.last_x:
            self.stay_counter += 1
        else:
            self.stay_counter = 0
        self.last_x = x_pos
        if self.stay_counter >= 40:
            reward -= 50
            done = True

        if level_type == "linear":
            reward += (score - self.curr_score) / 40.0
            if x_pos > self.current_x:
                reward += (x_pos - self.current_x) * 0.01
            elif x_pos < self.current_x:
                reward -= (self.current_x - x_pos) * 0.005
        elif level_type == "maze":
            pos_key = (x_pos // 10, y_pos // 10)
            if pos_key not in self.visited_positions:
                reward += 0.2
                self.visited_positions.add(pos_key)
            reward -= 0.01

        if done:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50

        if "world" in info and "stage" in info:
            self.world = info["world"]
            self.stage = info["stage"]

        self.curr_score = score
        self.current_x = x_pos
        return state, reward / 10.0, done, info

    def reset(self):
        state = self.env.reset()
        self.curr_score = 0
        self.current_x = 40
        self.visited_positions.clear()
        self.stay_counter = 0
        self.last_x = 40
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


# ===== Create Environment (Force RIGHT_ONLY actions) =====
def create_train_env(actions, world, stage):
    env_id = f"SuperMarioBros-{world}-{stage}-v3"  # 使用 v3，避免旧 API 警告
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = CustomReward(env)
    env = CustomSkipFrame(env)
    return env


# ===== Worker for Multiprocessing =====
def worker(conn, action_type, world, stage, worker_id=0):
    actions = RIGHT_ONLY
    env = create_train_env(actions, world, stage)

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, r, done, info = env.step(data.item())
                if done:
                    obs = env.reset()
                conn.send((obs, r, done, info))
            elif cmd == "reset":
                obs = env.reset()
                conn.send(obs)
            elif cmd == "close":
                conn.close()
                break
            else:
                raise NotImplementedError
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
    finally:
        env.close()


# ===== Multiple Environments Manager =====
class MultipleEnvironments:
    def __init__(self, action_type, num_envs, world=1, stage=1):
        self.action_type = action_type
        self.num_envs = num_envs
        self.num_actions = len(RIGHT_ONLY)
        self.num_states = 4

        self.agent_conns, env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        for idx in range(num_envs):
            p = mp.Process(
                target=worker,
                args=(env_conns[idx], "right", world, stage, idx)
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
            env_conns[idx].close()

    def close(self):
        for agent_conn in self.agent_conns:
            try:
                agent_conn.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join()
