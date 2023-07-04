import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFeat(nn.Module):
    """No STN module."""
    def __init__(self, global_feat=True):
        super(PointNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.shape[2]
        x = F.relu(self.bn1(self.conv1(x)))
        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, point_feat], 1)


class PointNetDiscriminator(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetDiscriminator, self).__init__()
        self.feat = PointNetFeat(global_feat=global_feat)    # point-wise feature
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 1, 1)
        # self.conv3 = torch.nn.Conv1d(256, 128, 1)
        # self.conv4 = torch.nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # batchsize = x.size()[0]
        # n_pts = x.size()[2]
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


