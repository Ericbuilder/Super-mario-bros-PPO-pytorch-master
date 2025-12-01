# train.py
import os
os.environ['OMP_NUM_THREADS'] = '1'

# 无头模式（防止 Kaggle 渲染相关问题）
import src.headless  # 确保 pyglet.options['headless'] = True 在任何 gym/nes_py 导入前生效

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
from gym_super_mario_bros.actions import RIGHT_ONLY
from collections import deque


def get_args():
    parser = argparse.ArgumentParser()
    # 强制只向右动作空间，保留参数但不使用用户传入值
    parser.add_argument("--action_type", type=str, default="right", choices=["right", "simple", "complex"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=int(5e6))
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=50)  # 保留但不按间隔保存
    parser.add_argument("--max_actions", type=int, default=200)
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    # 统一保存路径到 Kaggle 工作目录
    parser.add_argument("--saved_path", type=str, default="/kaggle/working")

    # 起始关卡
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)

    args = parser.parse_args()
    return args


# ===== 动态学习率设置函数 =====
def get_dynamic_lr(world, stage):
    if world <= 2:
        return 1e-3
    elif 3 <= world <= 5:
        return 1e-4
    else:
        return 5e-5


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 目录准备（只保留日志与模型保存，不涉及视频）
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    mp = _mp.get_context("spawn")

    # 强制使用 RIGHT_ONLY
    opt.action_type = "right"

    # 初始化关卡
    curr_world = opt.world
    curr_stage = opt.stage

    print(f"🚀 Starting training on World {curr_world}-{curr_stage}")
    # 注意：env 的 MultipleEnvironments 已移除 output_path，不再录制视频
    envs = MultipleEnvironments(opt.action_type, opt.num_processes, curr_world, curr_stage)

    num_actions = len(RIGHT_ONLY)
    num_states = 4

    model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()

    # 启动评估进程（评估内已禁用渲染）
    process = mp.Process(target=eval, args=(opt, model, num_states, num_actions))
    process.start()

    # 动态学习率
    curr_lr = get_dynamic_lr(curr_world, curr_stage)
    optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)
    print(f"⚙️ 初始学习率设置为 {curr_lr}")

    # 初始化状态
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states_data = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states_data, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()

    recent_passes = deque(maxlen=5)

    curr_episode = 0
    while True:
        curr_episode += 1
        old_log_policies = []
        actions_list = []
        values = []
        states = []
        rewards = []
        dones = []

        level_cleared_in_batch = False

        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions_list.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)

            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            step_results = [agent_conn.recv() for agent_conn in envs.agent_conns]
            state_list = [r[0] for r in step_results]
            reward_list = [r[1] for r in step_results]
            done_list = [r[2] for r in step_results]
            info_list = [r[3] for r in step_results]

            for info in info_list:
                if info.get("flag_get", False):
                    level_cleared_in_batch = True

            state = torch.from_numpy(np.concatenate(state_list, 0))

            reward = torch.from_numpy(np.array(reward_list, dtype=np.float32))
            done = torch.from_numpy(np.array(done_list, dtype=np.float32))

            if torch.cuda.is_available():
                state = state.cuda()
                reward = reward.cuda()
                done = done.cuda()
            else:
                reward = torch.FloatTensor(reward_list)
                done = torch.FloatTensor(done_list)

            rewards.append(reward)
            dones.append(done)
            curr_states = state

        avg_reward = torch.stack(rewards).mean().item()

        _, next_value = model(curr_states)
        next_value = next_value.squeeze()

        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions_list)
        values = torch.cat(values).detach()
        states = torch.cat(states)

        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values

        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                    int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)):
                    int((j + 1) * (opt.num_local_steps * opt.num_processes / opt.batch_size))
                ]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(
                    torch.min(
                        ratio * advantages[batch_indices],
                        torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) * advantages[batch_indices],
                    )
                )
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        print(
            f"Episode: {curr_episode}. World {curr_world}-{curr_stage}. Loss: {total_loss:.4f}. Avg Reward: {avg_reward:.2f}"
        )

        recent_passes.append(level_cleared_in_batch)
        pass_rate = sum(recent_passes) / len(recent_passes)
        print(f"📈 Recent pass rate (last {len(recent_passes)}): {pass_rate:.2f}")

        # 条件保存通用模型（仅当最近 5 个 episode 通过率 >= 0.7）
        if len(recent_passes) == recent_passes.maxlen and pass_rate >= 0.7:
            save_path = os.path.join(opt.saved_path, "ppo_super_mario_bros_continuous.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Pass rate >= 70%. General model saved to {save_path}")

        # 自动切关逻辑
        if level_cleared_in_batch:
            print(f"🎉 Level {curr_world}-{curr_stage} CLEARED! Switching level...")

            # 进入下一关
            curr_stage += 1
            if curr_stage > 4:
                curr_stage = 1
                curr_world += 1

            # 关闭旧环境
            print("🔄 Closing old environments...")
            envs.close()

            # 创建新环境（不录视频）
            print(f"🚀 Switching to World {curr_world}-{curr_stage}")
            envs = MultipleEnvironments(opt.action_type, opt.num_processes, curr_world, curr_stage)

            # 重置新环境状态
            [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
            curr_states_data = [agent_conn.recv() for agent_conn in envs.agent_conns]
            curr_states = torch.from_numpy(np.concatenate(curr_states_data, 0))
            if torch.cuda.is_available():
                curr_states = curr_states.cuda()

            # 更新学习率
            curr_lr = get_dynamic_lr(curr_world, curr_stage)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
            print(f"🔧 学习率更新为 {curr_lr} (World {curr_world}-{curr_stage})")

            # 重置计数与历史通过记录
            curr_episode = 0
            recent_passes.clear()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
