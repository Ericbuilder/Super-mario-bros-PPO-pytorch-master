# src/process.py
import os
os.environ['OMP_NUM_THREADS'] = '1'

# 必须在任何 gym/nes_py 导入之前启用 headless 模式
import src.headless  # 确保 pyglet 无头模式生效

import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


def eval(opt, global_model, num_states, num_actions):
    # 固定随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 确定动作空间
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    # 创建评估环境，传入 output_path 以保证路径一致（例如 /kaggle/working/output）
    env = create_train_env(actions, opt.world, opt.stage, output_path=opt.output_path)

    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()

    # 初始 reset（env.reset() 返回预处理后的 numpy 数组）
    state_np = env.reset()
    state = torch.from_numpy(state_np)
    if torch.cuda.is_available():
        state = state.cuda()

    done = True
    curr_step = 0
    actions_deque = deque(maxlen=opt.max_actions)

    while True:
        curr_step += 1
        if done:
            # 加载最新的全局模型权重
            local_model.load_state_dict(global_model.state_dict())

        # 前向推理
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()

        # 与环境交互（旧 Gym API）
        state_np, reward, done, info = env.step(action)

        # 可选：打印关卡完成信息
        if info.get("flag_get", False):
            world = info.get("world", "?")
            stage = info.get("stage", "?")
            print(f"✅ Eval completed level {world}-{stage} at step {curr_step}")

        # 注意：已移除 env.render()，在无显示环境（如 Kaggle）中避免窗口渲染导致的 GLU/pyglet 错误

        # 跟踪动作重复以检测卡住情况
        actions_deque.append(action)
        if curr_step > opt.num_global_steps or (len(actions_deque) == actions_deque.maxlen and actions_deque.count(actions_deque[0]) == actions_deque.maxlen):
            done = True

        # 如果 episode 结束则重置
        if done:
            curr_step = 0
            actions_deque.clear()
            state_np = env.reset()

        # 为下一步准备 state 张量
        state = torch.from_numpy(state_np)
        if torch.cuda.is_available():
            state = state.cuda()
