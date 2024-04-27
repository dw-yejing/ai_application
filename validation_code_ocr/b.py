import torch
import torch.nn.functional as F

# 定义输入序列和目标序列
logits = torch.tensor(
    [
        [-0.1, -0.3, -0.7, -1.1, -1.4, -1.8, -2.2, -2.5, -2.9],
        [-1.1, -1.4, -1.8, -2.2, -2.5, -2.9, -3.3, -3.7, -4.0],
        [-2.2, -2.5, -2.9, -3.3, -3.7, -4.0, -4.4, -4.8, -5.2],
        [-3.3, -3.7, -4.0, -4.4, -4.8, -5.2, -5.6, -6.0, -6.4],
        [-4.4, -4.8, -5.2, -5.6, -6.0, -6.4, -6.8, -7.2, -7.6],
        [-5.5, -5.9, -6.3, -6.7, -7.1, -7.5, -7.9, -8.3, -8.7],
        [-6.6, -7.0, -7.4, -7.8, -8.2, -8.6, -9.0, -9.4, -9.8],
        [-7.7, -8.1, -8.5, -8.9, -9.3, -9.7, -10.1, -10.5, -10.9],
    ],
    dtype=torch.float32,
    requires_grad=True,
)

# 为了简化，假设目标序列是 "3045"
targets = torch.tensor([3, 0, 4, 5], dtype=torch.int32)

# 在logits的第一个维度上添加时间步信息
logits = logits.unsqueeze(0).transpose(0, 1)

# 计算CTC损失，同时传递input_lengths和target_lengths
input_lengths = torch.tensor([logits.size(0)], dtype=torch.int32)
target_lengths = torch.tensor([len(targets)], dtype=torch.int32)
ctc_loss = torch.nn.CTCLoss()(logits, targets, input_lengths, target_lengths)

print("CTC Loss:", ctc_loss.item())
