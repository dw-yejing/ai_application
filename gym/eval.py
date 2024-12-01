import gym
import torch
import torch.nn as nn
from util.save_load import load_ckpt
from util.visualizer import generate_video
import torch.nn.functional as F
import numpy as np

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
def eval(env, model):
    done = False
    state = env.reset()
    frames = []
    rewards = []
    while not done:
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = model(state)
        action = F.softmax(action_probs, dim=-1)
        action = torch.argmax(action)
        next_state, reward, done, _ = env.step(action.item())
        state = next_state
        res = env.render(mode="rgb_array")
        frames.append(res)
        rewards.append(reward)
    np.save("out/frames.npy", frames)
    np.save("out/rewards.npy", rewards)
    print("=== completed ===")
    

if __name__ == "__main__":
    # 创建CartPole环境
    env = gym.make("CartPole-v1")

    # 设置网络参数
    input_size = env.observation_space.shape[0]
    hidden_size = 128
    output_size = env.action_space.n

    model = SimpleNet(input_size, hidden_size, output_size)
    load_ckpt(model, "out/agent_model.pth", None)
    eval(env, model)
    env.close()
    frames = np.load("out/frames.npy")
    rewards = np.load("out/rewards.npy")
    generate_video(frames, rewards, video_name="out/a.mp4")
    print("=== video generated ===")