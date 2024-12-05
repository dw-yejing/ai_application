import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from util.save_load import save_ckpt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


# 定义神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 训练函数
def train(env, model, optimizer, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        done = False
        total_reward = 0
        log_probs = []
        rewards = []

        while not done:
            action_probs = model(state)
            action_probs = F.softmax(action_probs, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action) # 获取采样的概率对数
            log_probs.append(log_prob)

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)

            rewards.append(reward)

            state = next_state
            total_reward += reward

        discounted_rewards = []
        G = 0
        for r in rewards[::-1]:
            G = r + 0.99 * G
            discounted_rewards.append(G)
        discounted_rewards.reverse()
        
        discounted_rewards = torch.tensor(discounted_rewards).float()
        log_probs = torch.stack(log_probs)

        advantage = discounted_rewards 
        loss = -(log_probs * advantage).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练损失到TensorBoard
        writer.add_scalar('Training Loss', loss.item(), episode)
        writer.add_scalar('Training reward', total_reward, episode)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}; loss: {loss.item()}")
        if total_reward > checkpoint['best_reward']:
            checkpoint['best_reward'] = total_reward
            save_ckpt(model, "out", optimizer, total_reward)

if __name__ == "__main__":
    # 创建CartPole环境
    env = gym.make("CartPole-v1")

    # 设置网络参数
    input_size = env.observation_space.shape[0]
    hidden_size = 128
    output_size = env.action_space.n

    # 创建模型、优化器
    model = SimpleNet(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir='runs/train_experiment_3')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': 0.0,
        'epoch': 0  # 当前训练的轮数
    }

    # 训练模型
    train(env, model, optimizer, num_episodes=2000)

    writer.close()
    # 关闭环境
    env.close()