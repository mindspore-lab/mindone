import numpy as np
import torch
import torch.nn as nn

# 定义输入形状
B, C, T, H, W = 2, 3, 16, 256, 256
x = np.random.normal(size=(B, C, T, H, W)).astype(np.float32)
x_tensor = torch.tensor(x)

# 定义 GroupNorm 层
group_norm = nn.GroupNorm(num_groups=3, num_channels=C)

# 第一次 GroupNorm 操作
y1 = group_norm(x_tensor)

# 重新排列形状
x_rearranged = x_tensor.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, C, T)

# 第二次 GroupNorm 操作
y2 = group_norm(x_rearranged)

# 恢复形状
# y1 = y1.view(B, C, T, H, W).permute(0, 2, 1, 3, 4).contiguous()
y2 = y2.view(B, H, W, C, T).permute(0, 3, 4, 1, 2).contiguous()

# 比较 y1 和 y2
print(y1.sum())
print(y2.sum())
print(torch.allclose(y1, y2))
