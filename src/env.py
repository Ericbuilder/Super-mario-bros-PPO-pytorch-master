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


# ===== Custom Reward Wrapper (回滚到基础版本) =====
class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.current_x = 40
        self.world = 1
        self.stage = 1
        
        # 移除了复杂的探索记录，回归纯粹的位置判断
        self.stay_counter = 0
        self.last_x = 40

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state)

        # 获取环境信息
        x_pos = info.get("x_pos", 0)
        
        # --- 1. 基础距离奖励 (Delta X) ---
        # 对于 1-1 这种线性关卡，这是最核心的驱动力
        delta_x = x_pos - self.current_x
        reward = delta_x 

        # --- 2. 时间与生存惩罚 ---
        # 每一帧都给予微小的惩罚，强迫 AI 跑起来，不要原地磨蹭
        reward -= 0.1 

        # --- 3. 死亡与通关 ---
        if done:
            if info.get("flag_get", False):
                reward += 15.0 # 通关奖励
            else:
                reward -= 15.0 # 死亡惩罚 (数值不需要太大，clip 会限制它)

        # --- 4. 防卡死 (Stay Penalty) ---
        if abs(x_pos - self.last_x) < 2: 
            self.stay_counter += 1
        else:
            self.stay_counter = 0
        
        if self.stay_counter >= 100: 
            reward -= 10.0 # 强制惩罚
            done = True    # 强制结束

        # --- 5. 状态更新 ---
        self.last_x = x_pos
        self.current_x = x_pos
        
        if "world" in info:
            self.world = info["world"]
            self.stage = info["stage"]

        # Reward Clipping: 限制在 [-5, 5]
        # 这对于 PPO 的稳定性至关重要，防止某个异常样本破坏梯度
        reward = np.clip(reward, -5.0, 5.0)

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.current_x = 40
        self.stay_counter = 0
        self.last_x = 40
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
