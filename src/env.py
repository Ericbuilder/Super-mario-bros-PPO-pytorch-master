import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym import Wrapper
from gym.spaces import Box
import cv2
import numpy as np
import torch.multiprocessing as mp


# ===== Frame preprocessing (保持 uint8 优化) =====
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        # 保持 uint8 (0-255)，不除以 255.0，节省传输带宽
        # 注意：这意味着 model.py 中必须进行归一化 (/ 255.0)
        return frame[None, :, :].astype(np.uint8)
    return np.zeros((1, 84, 84), dtype=np.uint8)


# ===== Custom Reward Wrapper (逻辑恢复为原版) =====
class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        # 观测空间保持 uint8 以匹配 process_frame
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.curr_score = 0
        self.current_x = 40
        self.world = 1
        self.stage = 1

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)

        # [原版逻辑 1] 基于分数的奖励
        # 这种逻辑鼓励吃金币和踩怪，原版认为这对线性关卡有辅助作用
        reward += (info.get("score", 0) - self.curr_score) / 40.0
        self.curr_score = info.get("score", 0)

        # [原版逻辑 2] 向前移动的微小奖励
        # 权重设得很小 (0.01)，避免距离奖励掩盖了分数的奖励
        x_pos = info.get("x_pos", 0)
        if x_pos > self.current_x:
            reward += (x_pos - self.current_x) * 0.01
        elif x_pos < self.current_x:
            reward -= (self.current_x - x_pos) * 0.005 # 后退惩罚更小

        # [原版逻辑 3] 死亡与通关
        # 原版使用的是较大的 +/- 50
        if done:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50

        # 更新世界/关卡信息
        if "world" in info and "stage" in info:
            self.world = info["world"]
            self.stage = info["stage"]

        # [原版逻辑 4] 硬编码的死亡区域 (Hard-coded death zones)
        # 针对 7-4 和 4-4 的特定陷阱判断，保留原版逻辑
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
        
        # [原版逻辑 5] 最终缩放
        # 原版是直接除以 10.0，而不是用 np.clip
        return state, reward / 10.0, done, info

    def reset(self):
        state = self.env.reset()
        self.curr_score = 0
        self.current_x = 40
        return process_frame(state)


# ===== Frame Skipping (保持不变) =====
class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84), dtype=np.uint8)
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.uint8)

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
            return self.states[None, :, :, :], total_reward, done, info

        max_state = np.max(np.concatenate(last_states, axis=0), axis=0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :], total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], axis=0)
        return self.states[None, :, :, :]


# ===== Create Environment =====
def create_train_env(actions, world, stage):
    env_id = f"SuperMarioBros-{world}-{stage}-v3"
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, actions)
    env = CustomReward(env)
    env = CustomSkipFrame(env)
    return env


# ===== Worker for Multiprocessing =====
def worker(conn, actions, world, stage, worker_id=0):
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
        
        if action_type == "right":
            self.actions = RIGHT_ONLY
        elif action_type == "simple":
            self.actions = SIMPLE_MOVEMENT
        else:
            self.actions = COMPLEX_MOVEMENT
            
        self.num_actions = len(self.actions)
        self.num_states = 4 

        self.agent_conns, env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        for idx in range(num_envs):
            p = mp.Process(
                target=worker,
                args=(env_conns[idx], self.actions, world, stage, idx)
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
