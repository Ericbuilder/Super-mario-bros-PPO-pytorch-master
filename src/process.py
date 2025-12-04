import os
os.environ['OMP_NUM_THREADS'] = '1'

import src.headless 
import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


def eval(opt, global_model, num_states, num_actions):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = create_train_env(actions, opt.world, opt.stage)

    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()

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
            local_model.load_state_dict(global_model.state_dict())

        with torch.no_grad():
            logits, value = local_model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()

        state_np, reward, done, info = env.step(action)
        
        # 仅保留通关提示
        if info.get("flag_get", False):
            print(f"✅ Eval CLEARED Level {opt.world}-{opt.stage} at step {curr_step}!")

        actions_deque.append(action)
        if curr_step > opt.num_global_steps or (len(actions_deque) == actions_deque.maxlen and actions_deque.count(actions_deque[0]) == actions_deque.maxlen):
            done = True

        if done:
            curr_step = 0
            actions_deque.clear()
            state_np = env.reset()

        state = torch.from_numpy(state_np)
        if torch.cuda.is_available():
            state = state.cuda()
