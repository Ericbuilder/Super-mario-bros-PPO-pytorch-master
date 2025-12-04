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
        # 保持 uint8，不除以 255.0，节省带宽
        return frame[None, :, :].astype(np.uint8)
    return np.zeros((1, 84, 84), dtype=np.uint8)


# ===== Custom Reward Wrapper (方案 A + D 实现核心) =====
class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.current_x = 40
        self.world = 1
        self.stage = 1
        
        # [方案 D] 迷宫探索记录 (记录访问过的 grid 坐标)
        self.visited_positions = set()
        
        # [方案 A] 线性关卡突破记录 (记录本局最远 X 坐标)
        self.max_x_reached = 40 
        
        self.stay_counter = 0
        self.last_x = 40

    def get_level_type(self, world, stage):
        # 定义迷宫关卡，通常是 x-4 (城堡)
        if stage == 4:
            return "maze"
        return "linear"

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)

        # 获取环境信息
        x_pos = info.get("x_pos", 0)
        y_pos = info.get("y_pos", 0)
        
        # --- 1. 基础距离奖励 (Delta X) ---
        # 即使在迷宫中，向右也是大方向，所以保留基础 delta
        delta_x = x_pos - self.current_x
        reward = delta_x 

        # --- 2. 基于关卡类型的特殊奖励 (Scheme A & D) ---
        level_type = self.get_level_type(self.world, self.stage)

        if level_type == "linear":
            # === [方案 A: Milestone Reward] ===
            # 如果当前位置超过了本局历史最远位置，给予额外奖励
            # 这对于解决 2-1 这种"掉坑恐惧症"非常有效
            if x_pos > self.max_x_reached:
                reward += 0.5  # 突破奖励
                self.max_x_reached = x_pos
                
        elif level_type == "maze":
            # === [方案 D: Maze Grid Exploration] ===
            # 将地图划分为 10x10 的网格
            # 只要进入新的网格(无论是向右还是向上/向下)，都给予奖励
            pos_key = (x_pos // 10, y_pos // 10)
            if pos_key not in self.visited_positions:
                reward += 1.0 # 发现新区域的大奖励，鼓励上下探索
                self.visited_positions.add(pos_key)
            
            # 迷宫中稍微增加一点时间压力，防止为了刷分而在两个格子里反复横跳(虽然set解决了重复，但防止磨蹭)
            reward -= 0.1

        # --- 3. 通用惩罚与奖励 ---
        
        # 基础时间压力 (所有关卡)
        reward -= 0.05 

        # 死亡与通关
        if done:
            if info.get("flag_get", False):
                reward += 15.0 # 通关大奖
            else:
                reward -= 15.0 # 死亡惩罚

        # 防卡死 (Stay Penalty)
        # 如果连续 100 帧(约 25 步)位移小于 2 像素
        if abs(x_pos - self.last_x) < 2: 
            self.stay_counter += 1
        else:
            self.stay_counter = 0
        
        if self.stay_counter >= 100: 
            reward -= 10.0 # 强制惩罚
            done = True    # 强制结束，防止无效训练

        # --- 4. 状态更新 ---
        self.last_x = x_pos
        self.current_x = x_pos
        
        # 更新 World/Stage 信息
        if "world" in info:
            self.world = info["world"]
            self.stage = info["stage"]

        # Reward Clipping: 限制奖励范围，稳定梯度
        reward = np.clip(reward, -5.0, 5.0)

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.current_x = 40
        self.max_x_reached = 40 # 重置方案 A 记录
        self.visited_positions.clear() # 重置方案 D 记录
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
    env = CustomReward(env)     # 此时加载的是包含方案 A+D 的新 Reward
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
