import torch.nn as nn
import torch

class DEA(nn.Module):
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()

        self.conv1 = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.conv2 = nn.Conv2d(mlp_ratio * dim, mlp_ratio * dim, 1)

        self.norm = nn.LayerNorm(dim)
        self.f1 = nn.Linear(dim, mlp_ratio * dim)

        self.f3 = nn.Linear(dim, mlp_ratio * dim)

        self.f2 = nn.Linear(dim, mlp_ratio * dim)
        self.g = nn.Linear(mlp_ratio * dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        input = x.clone()
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x1 = self.f1(x)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.f3(x1)

        x2 = self.f2(x)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        x2 = x2.permute(0, 2, 3, 1)

        x = x1 * x2

        x = self.g(x)
        x = x.permute(0, 3, 1, 2)

        x = input + x
        return x


if __name__ == '__main__':

    input = torch.randn(128, 128, 8, 8)
    dsconv = DEA(128)
    output = dsconv(input)
    print(output.shape)