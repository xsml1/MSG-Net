import torch
from torch import nn
import torch.nn.functional as F
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)



class SGFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SGFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 3, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 3, bias=bias)
        self.sp = SpatialAttention()
        self.sig = nn.Sigmoid()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
        x1 = self.sig((F.gelu(x1)))
        x2 = self.sp((F.gelu(x2)))
        x3 = x3 * x1
        x3 = x3 * x2
        x = self.project_out(x3)
        return x