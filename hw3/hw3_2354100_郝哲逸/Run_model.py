import numpy as np
import torch
import imageio
from PIL import Image
import os
from game import wrapped_flappy_bird as game
from Train_DQN import DQN, preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """加载训练好的模型"""
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def run_demo_with_gif(model_path, max_frames):
    """运行训练好的模型并生成GIF演示"""
    print("Loading model...")
    model = load_model(model_path)

    # 从模型路径提取训练次数
    model_name = os.path.basename(model_path).split('.')[0]  # 去掉.pth后缀
    training_iter = model_name.split('_')[-1]  # 提取次数

    # 创建GIF文件夹
    gif_dir = "GIF"
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f"FB_{training_iter}.gif")

    print("Initializing game...")
    game_state = game.GameState()

    # 初始化状态
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = preprocess_image(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    print("Starting demo and recording GIF...")
    total_reward = 0
    frames = []

    while len(frames) < max_frames:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.expand_dims(s_t, 0)).to(device)
            q_values = model(state_tensor)
            action_index = torch.argmax(q_values).item()

        action = np.zeros(2)
        action[action_index] = 1
        x_t1, r_t, terminal = game_state.frame_step(action)
        total_reward += r_t

        # 修复图像方向问题
        frame = Image.fromarray(x_t1)
        frame = frame.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        frame = frame.rotate(90, expand=True)  # expand=True保持完整图像
        frames.append(frame)

        x_t1 = preprocess_image(x_t1)
        s_t = np.append(np.expand_dims(x_t1, 0), s_t[:3, :, :], axis=0)

        if terminal:
            print(f"Game Over! Total Reward: {total_reward}")
            break

    print(f"Saving GIF with {len(frames)} frames...")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF saved to {gif_path}")


if __name__ == "__main__":
    MODEL_PATH = "saved_models/flappy_bird_dqn_100000.pth"
    MAX_FRAMES = 180  # (30fps每秒)

    run_demo_with_gif(MODEL_PATH, MAX_FRAMES)