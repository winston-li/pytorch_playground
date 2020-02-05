import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            # feature map: 137x137 for 560x560; 67x67 for 280x280
            nn.AdaptiveAvgPool2d(output_size=(67, 67)),
            nn.Flatten(2), 
            nn.Linear(67*67, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
        )

    def forward(self, x):
        # print(f'input shape = {x.shape}')
        H = self.feature_extractor_part1(x)
        # batch_size, c, hfmp, wfmp = H.shape
        H = self.feature_extractor_part2(H)  # BxNxL (N==c)

        A = self.attention(H)                # BxNxK
        A = torch.transpose(A, 2, 1)         # BxKxN
        A = F.softmax(A, dim=-1)             # softmax over N

        M = torch.bmm(A, H).squeeze(1)      # BxKxL => BxL
        Y_logits = self.classifier(M)
        return Y_logits