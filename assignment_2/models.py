import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropChannel(nn.Module):
    def __init__(self, p=0.2):
        super(DropChannel, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        batch_size, _, channels, time = x.shape
        mask = (torch.rand(batch_size, channels, 1, device=x.device) > self.p).float()
        mask = mask.expand(-1, -1, time)
        mask = mask.unsqueeze(1)
        return x * mask


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class BaseMEGCNN(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 num_classes: int = 4,
                 dropout_prob: float = 0.5,
                 dropchannel_p: float = 0.2,
                 num_segments: int = 0,
    ) -> None:
        super(BaseMEGCNN, self).__init__()

        self.dropchannel = DropChannel(p=dropchannel_p)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_prob)

        self.fc1 = nn.Linear(32 * 62 * 25, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dropchannel(x)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.fc2(x)

        return x


class BaseMEGEEGNet(nn.Module):
    def __init__(self,
                 chans=248,
                 samples=100,
                 num_classes=4,
                 dropout_prob=0.5,
                 dropchannel_p=0.2,
                 hidden_size=64,
                 num_segments=0,
    ):
        super(BaseMEGEEGNet, self).__init__()

        self.dropchannel = DropChannel(p=dropchannel_p)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_prob)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_prob)
        )

        out_time = samples // (4 * 8)  # 100//32 = 3

        self.fc1 = nn.Linear(16 * out_time, hidden_size)

    def forward(self, x):
        x = self.dropchannel(x)

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return x


class MultiViewCNN(nn.Module):
    def __init__(self, base_cnn_cls, num_classes=4, **base_cnn_kwargs):
        super(MultiViewCNN, self).__init__()

        self.num_segments = base_cnn_kwargs.get('num_segments')
        self.alpha = 1.0

        self.heads = nn.ModuleList([
            base_cnn_cls(**base_cnn_kwargs) for _ in range(self.num_segments)
        ])

        hidden_size = base_cnn_kwargs.get('hidden_size')

        self.fc = nn.Linear(hidden_size, num_classes)

        # Domain classifier (dla rozróżniania osób)
        self.domain_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # np. 2 klasy: znana osoba vs nowa
        )

    def forward(self, x, return_domain=True):
        # x: (batch, num_segments, channels, time)
        embeddings = []

        for i in range(self.num_segments):
            xi = x[:, i:i+1, :, :]  # (batch, 1, channels, time)
            hi = self.heads[i](xi)  # (batch, hidden)
            embeddings.append(hi)

        h = torch.stack(embeddings, dim=0).mean(dim=0)  # (batch, hidden)

        y_logits = self.fc(h)

        if return_domain:
            reversed_h = grad_reverse(h, self.alpha)
            domain_logits = self.domain_fc(reversed_h)
            return y_logits, domain_logits

        return y_logits
