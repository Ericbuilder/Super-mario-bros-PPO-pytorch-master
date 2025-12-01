import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from src.model import PPO
import torch.nn.functional as F
import numpy as np
import cv2
import shutil
import subprocess as sp


class Monitor:
    def __init__(self, width, height, saved_path, fps=60):
        self.width = width
        self.height = height
        self.saved_path = saved_path
        self.fps = fps

        # Ensure output directory exists
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)

        # Check ffmpeg availability
        self.ffmpeg_available = shutil.which("ffmpeg") is not None
        self.pipe = None

        if self.ffmpeg_available:
            self.command = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "rgb24", "-r", str(fps),
                "-i", "-",
                "-an", "-vcodec", "mpeg4", "-b:v", "2M", saved_path
            ]
            try:
                self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
                print(f"🎥 ffmpeg recording to {saved_path}")
            except Exception as e:
                print(f"⚠️ ffmpeg start failed: {e}. Falling back to OpenCV writer.")
                self.ffmpeg_available = False

        if not self.ffmpeg_available:
            # Fallback: OpenCV VideoWriter with mp4v
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv_writer = cv2.VideoWriter(saved_path, fourcc, fps, (width, height))
            if not self.cv_writer.isOpened():
                print("❌ OpenCV VideoWriter failed to open.")
                self.cv_writer = None

    def record(self, image_array):
        # image_array expected shape (H, W, 3) in RGB
        if image_array is None:
            return

        if self.pipe and self.pipe.stdin:
            # Write raw bytes to ffmpeg
            try:
                self.pipe.stdin.write(image_array.tobytes())
            except Exception as e:
                print(f"⚠️ ffmpeg write failed: {e}")
        elif hasattr(self, "cv_writer") and self.cv_writer is not None:
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            self.cv_writer.write(bgr)

    def close(self):
        # Close ffmpeg or OpenCV writer
        if self.pipe:
            try:
                self.pipe.stdin.close()
                self.pipe.wait()
            except Exception:
                pass
        if hasattr(self, "cv_writer") and self.cv_writer is not None:
            self.cv_writer.release()


def process_frame(frame):
    # Convert obs to 84x84 grayscale [1, 84, 84], normalized to [0,1]
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return (resized[None, :, :] / 255.0).astype(np.float32)
    return np.zeros((1, 84, 84), dtype=np.float32)


def get_args():
    parser = argparse.ArgumentParser()
    # 强制只向右动作空间，保留参数但固定为 right
    parser.add_argument("--action_type", type=str, default="right", choices=["right", "simple", "complex"])
    # Save and output paths set for Kaggle working directory
    parser.add_argument("--saved_path", type=str, default="/kaggle/working",
                        help="Directory where trained models (.pth) are stored")
    parser.add_argument("--output_path", type=str, default="/kaggle/working/output",
                        help="Directory to save test videos")
    # Default model name includes .pth
    parser.add_argument("--model_name", type=str, default="ppo_super_mario_bros_continuous.pth",
                        help="Model filename to load (e.g., ppo_super_mario_bros_continuous.pth)")
    # Test specific level
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    return parser.parse_args()


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 强制使用 RIGHT_ONLY（只右走与右跳）
    actions = RIGHT_ONLY
    opt.action_type = "right"

    os.makedirs(opt.output_path, exist_ok=True)

    # Video filename includes level info
    video_path = os.path.join(opt.output_path, f"test_video_w{opt.world}_s{opt.stage}.mp4")

    # Create environment for specific level, fallback to v3
    env_id = f"SuperMarioBros-{opt.world}-{opt.stage}-v0"
    try:
        print(f"🚀 Loading environment: {env_id}")
        env = gym_super_mario_bros.make(env_id)
    except Exception as e:
        print(f"❌ Error loading {env_id}: {e}")
        print("Fallback to standard SuperMarioBros-v3")
        env = gym_super_mario_bros.make("SuperMarioBros-v3")

    monitor = Monitor(256, 240, video_path, fps=60)
    env = JoypadSpace(env, actions)

    class TestWrapper:
        def __init__(self, env, monitor):
            self.env = env
            self.monitor = monitor

        def reset(self):
            obs = self.env.reset()
            if self.monitor:
                self.monitor.record(obs)
            return obs

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            if self.monitor:
                self.monitor.record(obs)
            return obs, reward, done, info

        def render(self):
            # Optional: NES Py headless; rendering may be a no-op in Kaggle
            pass

        def close(self):
            self.env.close()
            if self.monitor:
                self.monitor.close()

    env = TestWrapper(env, monitor)

    # Load model
    model_path = os.path.join(opt.saved_path, opt.model_name)
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        if os.path.exists(opt.saved_path):
            for f in os.listdir(opt.saved_path):
                print(f"  - {f}")
        return

    num_states = 4
    num_actions = len(actions)
    model = PPO(num_states, num_actions)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"🚀 Testing model {opt.model_name} on World {opt.world}-{opt.stage} (RIGHT_ONLY actions)")
    obs = env.reset()
    state = torch.from_numpy(process_frame(obs))
    if torch.cuda.is_available():
        state = state.cuda()

    step_count = 0
    done = False
    while not done:
        step_count += 1
        with torch.no_grad():
            logits, _ = model(state)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()

        obs, reward, done, info = env.step(action)

        # Print level completion
        if info.get("flag_get", False):
            world = info.get("world", opt.world)
            stage = info.get("stage", opt.stage)
            print(f"✅ Completed level {world}-{stage} at step {step_count}")

        # Update state
        state = torch.from_numpy(process_frame(obs))
        if torch.cuda.is_available():
            state = state.cuda()

        # Safety break
        if step_count > 10000:
            print("⚠️ Step limit reached. Ending episode.")
            break

    print(f"🎥 Video saved to {video_path}")
    env.close()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
