import os
import cv2
import random
import numpy as np
import argparse # 参数
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from game import wrapped_flappy_bird as game

def get_args():
    """定义参数，可以在命令行输入"""
    # 创建解析器
    parser = argparse.ArgumentParser(description='Flappy Bird DQN 训练参数')
    # 训练参数
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子 γ（未来奖励的重要性）')
    parser.add_argument('--initial_epsilon', type=float, default=0.3, help='初始探索率 ε，表示以随机方式探索的概率')
    parser.add_argument('--final_epsilon', type=float, default=0.001, help='最终探索率 ε，训练后期的最小随机概率')
    parser.add_argument('--replay_memory_size', type=int, default=100000, help='经验回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=128, help='每次训练采样的经验数量')
    parser.add_argument('--observe_steps', type=int, default=20000, help='仅观察游戏、不进行训练的步数')
    parser.add_argument('--explore_steps', type=int, default=200000, help='从初始探索率衰减到最终探索率所需的步数')
    parser.add_argument('--max_steps', type=int, default=1000000, help='最大训练步数')
    # 日志与模型保存
    parser.add_argument('--save_interval', type=int, default=50000, help='每训练多少步保存一次模型')
    parser.add_argument('--log_interval', type=int, default=5000, help='每多少步打印一次训练日志')
    # 优化器参数
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='优化器类型：sgd 或 adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率（越小学习越慢，但可能更稳定）')
    # 路径参数
    parser.add_argument('--saved_path', type=str, default='saved_models', help='保存模型的文件夹路径')
    # 游戏参数
    parser.add_argument('--image_size', type=int, default=80, help='将游戏画面调整为的图像尺寸（如 80 表示 80x80）')
    # 解析参数
    args = parser.parse_args()
    return args

class DQN(nn.Module):
    """DQN（Deep Q-Network）深度强化学习模型，用于估计每个动作的Q值。输入是处理后的图像序列，输出是每个动作的Q值。"""
    def __init__(self, image_size=80):
        super(DQN, self).__init__()
        # --- 卷积层部分 ---
        # 输入图像通道为4（通常是连续4帧游戏画面），输出通道为32，卷积核大小8x8，步幅为4
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        # 第二层卷积：输入32通道，输出64通道，卷积核大小4x4，步幅为2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 第三层卷积：输入64通道，输出64通道，卷积核大小3x3，步幅为1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # --- 动态计算全连接层输入大小 ---
        # 用一个假的输入图像，通过卷积网络，模拟一遍前向传播，然后看输出展平之后有多少个神经元
        with torch.no_grad():  # 禁用梯度计算 —— 因为我们只是临时模拟前向传播
            dummy = torch.zeros(1, 4, image_size, image_size)  # batch=1，通道=4，图像尺寸=image_size
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            self.fc_input_size = dummy.view(1, -1).size(1)  # 计算展平后的特征数量（也就是全连接层输入）

        # --- 全连接层部分 ---
        # 第一个全连接层，将卷积输出映射到 512 维隐藏层
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        # 第二个全连接层，输出 2 个动作的 Q 值（例如：跳 or 不跳）
        self.fc2 = nn.Linear(512, 2)
        # 初始化网络权重
        self._init_weights()

    def _init_weights(self):
        """使用 He 初始化法（Kaiming）初始化卷积层和全连接层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 对权重进行 He 初始化，适合 ReLU 激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 对偏置进行常数初始化
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        前向传播过程：输入是图像张量，输出是每个动作的Q值，x.shape: (batch_size, 4, image_size, image_size)
        """
        # --- 卷积层 + ReLU 激活 ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # --- 展平为向量 ---
        x = x.view(x.size(0), -1)
        # --- 全连接层 + ReLU + 输出层 ---
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 表示每个动作的Q值


def preprocess_image(image, image_size=80):
    """预处理游戏画面: 调整大小、灰度化、二值化"""
    image = cv2.cvtColor(cv2.resize(image, (image_size, image_size)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image.astype(np.float32) / 255.0  # 归一化到[0,1]

def train(args):
    """主训练函数，使用DQN算法训练模型玩游戏"""
    # 设备配置：优先使用 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 打印训练参数
    print("Training parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # 初始化 DQN 模型
    model = DQN(image_size=args.image_size).to(device)
    # 根据参数选择优化器：Adam 或 SGD，学习率为 args.lr
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # 损失函数使用均方误差（MSE）
    criterion = nn.MSELoss()

    # 初始化游戏环境
    game_state = game.GameState()
    # 初始化动作：开始时选择“什么都不做”动作（全零向量，只有第一个元素置1）
    do_nothing = np.zeros(2)
    do_nothing[0] = 1

    # 游戏环境执行“无动作”操作，得到初始画面 x_t，奖励 r_0，是否结束 terminal
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # 对初始画面进行预处理（灰度化，缩放等）
    x_t = preprocess_image(x_t, args.image_size)
    # 堆叠4帧作为输入状态（DQN输入4张连续画面），初始时用4张同样的图像堆叠
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    # 经验回放缓冲区，存储游戏状态转换，用于采样训练，容量为replay_memory_size
    replay_memory = deque(maxlen=args.replay_memory_size)
    # 初始化探索率 ε（epsilon），用来控制探索（随机动作）与利用（模型动作）的平衡
    epsilon = args.initial_epsilon

    # 计步器 t，记录已经训练多少步
    t = 0
    # 用来保存当前最高得分，方便观察训练进展
    max_score = 0

    # 创建保存模型和日志的目录（如果不存在则创建）
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    # --- 主训练循环 ---
    while t < args.max_steps:
        # 1. 选择动作（ε-greedy策略）
        # 以概率epsilon或在观察期内，随机选动作（探索），否则根据模型预测的Q值选择最大Q值对应的动作（利用）
        if random.random() <= epsilon or t <= args.observe_steps:
            action_index = random.randint(0, 1)  # 随机选动作（0或1）
        else:
            with torch.no_grad():
                # 将状态转成tensor送入模型计算Q值
                state_tensor = torch.FloatTensor(np.expand_dims(s_t, 0)).to(device)
                q_values = model(state_tensor)
                # 选择最大Q值对应的动作索引
                action_index = torch.argmax(q_values).item()

        # 2. 执行动作（动作编码为one-hot向量）
        action = np.zeros(2)
        action[action_index] = 1
        # 送入游戏环境，执行动作，得到下一帧画面 x_t1，奖励 r_t，是否结束 terminal
        x_t1, r_t, terminal = game_state.frame_step(action)
        # 对新画面进行预处理
        x_t1 = preprocess_image(x_t1, args.image_size)
        # 3. 更新状态 s_t1：把新画面放在最前面，保留之前的3帧，共4帧构成新的状态
        s_t1 = np.append(np.expand_dims(x_t1, 0), s_t[:3, :, :], axis=0)
        # 4. 将这次经验 (s_t, action, r_t, s_t1, terminal) 存入经验回放缓冲区
        replay_memory.append((s_t, action, r_t, s_t1, terminal))

        # 5. 训练阶段
        if t > args.observe_steps and len(replay_memory) >= args.batch_size:
            # 随机采样一个小批量经验
            minibatch = random.sample(replay_memory, args.batch_size)
            # 分别提取状态，动作，奖励，下一状态，终止标志
            batch_states = np.array([d[0] for d in minibatch])
            batch_actions = np.array([np.argmax(d[1]) for d in minibatch])
            batch_rewards = np.array([d[2] for d in minibatch])
            batch_next_states = np.array([d[3] for d in minibatch])
            batch_dones = np.array([d[4] for d in minibatch])

            # 转成tensor，送到设备
            s_j = torch.FloatTensor(batch_states).to(device)
            a_batch = torch.LongTensor(batch_actions).to(device)
            r_batch = torch.FloatTensor(batch_rewards).to(device)
            s_j1 = torch.FloatTensor(batch_next_states).to(device)
            done_batch = torch.FloatTensor(batch_dones).to(device)

            # 计算当前Q值：模型预测s_j状态下所有动作的Q值，取出对应动作的Q值
            current_q = model(s_j).gather(1, a_batch.unsqueeze(1))
            # 计算下一状态的最大Q值，detach()避免梯度传播到下一状态网络
            next_q = model(s_j1).max(1)[0].detach()
            # 计算目标Q值（Bellman方程）：
            # 若done，目标为即时奖励r_batch
            # 否则为即时奖励加折扣后的下一状态最大Q值
            target_q = r_batch + (1 - done_batch) * args.gamma * next_q

            # 优化步骤：清零梯度，计算损失，反向传播，更新参数
            optimizer.zero_grad()
            loss = criterion(current_q.squeeze(), target_q)
            loss.backward()
            optimizer.step()

            # 逐步衰减探索率ε，直到最终ε
            if epsilon > args.final_epsilon:
                epsilon -= (args.initial_epsilon - args.final_epsilon) / args.explore_steps

        # 6. 更新状态为下一状态
        s_t = s_t1
        t += 1

        # 7. 记录和保存
        # 游戏结束时打印分数信息，更新最高分
        if terminal:
            current_score = game_state.score
            max_score = max(max_score, current_score)
            print(f"Step: {t} | Score: {current_score} | Max Score: {max_score} | ε: {epsilon:.4f}")
        # 定期保存模型参数
        if t % args.save_interval == 0:
            save_path = os.path.join(args.saved_path, f"flappy_bird_dqn_{t}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at step {t} to {save_path}")
        # 定期打印日志信息（步骤、ε、动作、奖励）
        if t % args.log_interval == 0:
            print(f"Step: {t} | ε: {epsilon:.4f} | Action: {action_index} | Reward: {r_t}")



if __name__ == "__main__":
    args = get_args()
    # 设置随机种子
    torch.manual_seed(42)  # 保证每次运行时模型初始化权重、随机采样等操作的一致性
    np.random.seed(42)  # 确保 NumPy 产生的随机数序列一致
    random.seed(42)  # 内置random模块的随机数种子

    # 开始训练
    train(args)