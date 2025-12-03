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


# ===== Frame preprocessing [优化1: 保持 uint8] =====
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        # [优化1] 不除以 255.0，保持 uint8 以节省多进程传输带宽
        return frame[None, :, :].astype(np.uint8)
    return np.zeros((1, 84, 84), dtype=np.uint8)


# ===== Custom Reward Wrapper [优化2: 重构奖励] =====
class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
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

        # 获取环境信息
        x_pos = info.get("x_pos", 0)
        y_pos = info.get("y_pos", 0)
        
        # [优化2] 奖励重构
        # 1. 基础距离奖励 (只看 Delta)
        delta_x = x_pos - self.current_x
        reward = delta_x 

        # 2. 死亡判定与惩罚
        if done:
            if info.get("flag_get", False):
                reward += 15.0  # 通关奖励
            else:
                reward -= 15.0  # 死亡惩罚 (比原本的 -50 温和，配合 clip 使用)
        
        # 3. 时间/生存惩罚 (鼓励快速通关)
        reward -= 0.1

        # 4. 原地不动惩罚 (防止卡死)
        if x_pos == self.last_x:
            self.stay_counter += 1
        else:
            self.stay_counter = 0
        
        if self.stay_counter >= 40: # 连续40帧(跳帧后)不动
            reward -= 10.0 # 额外惩罚
            done = True

        self.last_x = x_pos
        self.current_x = x_pos

        # [优化2] Reward Clipping: 限制在 [-5, 5] 之间，防止梯度爆炸
        reward = np.clip(reward, -5.0, 5.0)

        # 更新关卡信息
        if "world" in info and "stage" in info:
            self.world = info["world"]
            self.stage = info["stage"]

        # 迷宫关卡特殊处理 (如 1-4)
        if self.get_level_type(self.world, self.stage) == "maze":
            pos_key = (x_pos // 10, y_pos // 10)
            if pos_key not in self.visited_positions:
                reward += 1.0 # 鼓励探索新区域
                self.visited_positions.add(pos_key)
            reward -= 0.1 # 迷宫中额外的时间压力

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.current_x = 40
        self.visited_positions.clear()
        self.stay_counter = 0
        self.last_x = 40
        return process_frame(state)


# ===== Frame Skipping =====
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
            # 注意: 如果 done, total_reward 可能包含这一步的死亡惩罚
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
    env = JoypadSpace(env, actions) # 使用传入的 actions
    env = CustomReward(env)
    env = CustomSkipFrame(env)
    return env


# ===== Worker for Multiprocessing [优化4: 修正动作传递] =====
def worker(conn, actions, world, stage, worker_id=0):
    # [优化4] 之前这里硬编码了 RIGHT_ONLY，现在使用传入的 actions
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
        
        # [优化4] 根据参数确定动作空间
        if action_type == "right":
            self.actions = RIGHT_ONLY
        elif action_type == "simple":
            self.actions = SIMPLE_MOVEMENT
        else:
            self.actions = COMPLEX_MOVEMENT
            
        self.num_actions = len(self.actions)
        self.num_states = 4 # Stacked frames

        self.agent_conns, env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        for idx in range(num_envs):
            p = mp.Process(
                target=worker,
                args=(env_conns[idx], self.actions, world, stage, idx) # 传入 self.actions
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