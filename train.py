import os
os.environ['OMP_NUM_THREADS'] = '1'

import src.headless # Headless mode
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


def get_args():
    parser = argparse.ArgumentParser()
    # [优化4] 默认动作改为 simple，支持更复杂的动作
    parser.add_argument("--action_type", type=str, default="simple", choices=["right", "simple", "complex"])
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
    parser.add_argument("--save_interval", type=int, default=50) # 定期保存间隔
    parser.add_argument("--max_actions", type=int, default=200)
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="/kaggle/working")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)

    args = parser.parse_args()
    return args


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

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    mp = _mp.get_context("spawn")

    # [优化4] 移除 opt.action_type = "right" 的硬编码，使用参数控制
    print(f"🚀 Starting training on World {opt.world}-{opt.stage} with Action Type: {opt.action_type}")
    
    envs = MultipleEnvironments(opt.action_type, opt.num_processes, opt.world, opt.stage)

    # 从 envs 获取正确的动作数量
    num_actions = envs.num_actions
    num_states = 4

    model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()

    process = mp.Process(target=eval, args=(opt, model, num_states, num_actions))
    process.start()

    curr_lr = get_dynamic_lr(opt.world, opt.stage)
    optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)
    print(f"⚙️ Initial Learning Rate: {curr_lr}")

    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states_data = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states_data, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda() # 此时还是 uint8/ByteTensor (如果转换了的话) 或者 FloatTensor

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

        print(f"Ep: {curr_episode}. World {opt.world}-{opt.stage}. Loss: {total_loss:.4f}. Reward: {avg_reward:.2f}")

        # 定期保存
        if curr_episode % opt.save_interval == 0:
            save_path = os.path.join(opt.saved_path, f"ppo_mario_simple_{opt.world}_{opt.stage}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model saved to {save_path}")

        # 自动切关
        if level_cleared_in_batch:
            print(f"🎉 Level {opt.world}-{opt.stage} CLEARED! Switching level...")
            opt.stage += 1
            if opt.stage > 4:
                opt.stage = 1
                opt.world += 1

            envs.close()
            print(f"🚀 Switching to World {opt.world}-{opt.stage}")
            envs = MultipleEnvironments(opt.action_type, opt.num_processes, opt.world, opt.stage)
            
            [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
            curr_states_data = [agent_conn.recv() for agent_conn in envs.agent_conns]
            curr_states = torch.from_numpy(np.concatenate(curr_states_data, 0))
            if torch.cuda.is_available():
                curr_states = curr_states.cuda()

            curr_lr = get_dynamic_lr(opt.world, opt.stage)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
            
            curr_episode = 0

if __name__ == "__main__":
    opt = get_args()
    train(opt)