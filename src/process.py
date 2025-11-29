import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)

    # Determine action space
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    # Create continuous-level environment (SuperMarioBros-v0)
    # ✅ Fix: Pass the initial world and stage from arguments to prevent errors
    # Note: This visualization process will stay on the initial level.
    # It does not automatically switch levels with the main training process.
    env = create_train_env(actions, opt.world, opt.stage)

    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()

    # Initial reset (old Gym API: returns obs only)
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()

    done = True
    curr_step = 0
    actions_deque = deque(maxlen=opt.max_actions)

    while True:
        curr_step += 1
        if done:
            # Load latest global model weights
            local_model.load_state_dict(global_model.state_dict())

        # Forward pass
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()

        # Step environment (old Gym API)
        state_np, reward, done, info = env.step(action)

        # Optional: print level completion
        if info.get("flag_get", False):
            world = info.get("world", "?")
            stage = info.get("stage", "?")
            print(f"✅ Eval completed level {world}-{stage} at step {curr_step}")

        # Render game window
        env.render()

        # Track action repetition
        actions_deque.append(action)
        if curr_step > opt.num_global_steps or actions_deque.count(actions_deque[0]) == actions_deque.maxlen:
            done = True

        # Reset if episode ends
        if done:
            curr_step = 0
            actions_deque.clear()
            state_np = env.reset()

        # Prepare state for next step
        state = torch.from_numpy(state_np)
        if torch.cuda.is_available():
            state = state.cuda()