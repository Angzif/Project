import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Set device to CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize
def weights_init_(m):  #
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class CNN(nn.Module):    #
    '''
    Pi(s)
    Input:  State --> [b,4,100,100]
            Dim   --> [b,4*3]
    Output: mean,log_std [b,2,2], rotation_logits
            action A(s)
    '''

    def __init__(self, num_actions=2, init_w=3e-3, out_channels=[32, 32, 32], hidden_arr=[500, 100, 50, 9], num_boxes=4,
                 batch_norm=False):
        super(CNN, self).__init__()
        
        # bx4x100x100 --> bx32x48x48
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=out_channels[0], kernel_size=5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        # bx32x48x48 --> bx32x23x23
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        # bx32x23x23 --> bx32x11x11
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(out_channels[2])

        # mean
        # flatten size --> 32*11*11
        self.fc1_mean = nn.Linear(32 * 11 * 11, hidden_arr[0])
        self.fc1_mean_bn = nn.BatchNorm1d(hidden_arr[0])
        # 512  --> 128
        self.fc2_mean = nn.Linear(hidden_arr[0], hidden_arr[1])
        self.fc2_mean_bn = nn.BatchNorm1d(hidden_arr[1])
        # 128  --> 36
        self.fc3_mean = nn.Linear(hidden_arr[1], hidden_arr[2])
        self.fc3_mean_bn = nn.BatchNorm1d(hidden_arr[2])
        # 36 + 12 --> num_actions
        self.fc4_mean = nn.Linear(hidden_arr[2] + 3 * num_boxes, num_actions)

        # log-std
        # flatten size --> 32*11*11
        self.fc1_std = nn.Linear(32 * 11 * 11, hidden_arr[0])
        self.fc1_std_bn = nn.BatchNorm1d(hidden_arr[0])
        # 512  --> 128
        self.fc2_std = nn.Linear(hidden_arr[0], hidden_arr[1])
        self.fc2_std_bn = nn.BatchNorm1d(hidden_arr[1])
        # 128  --> 36
        self.fc3_std = nn.Linear(hidden_arr[1], hidden_arr[2])
        self.fc3_std_bn = nn.BatchNorm1d(hidden_arr[2])
        # 36 + 12 --> num_actions
        self.fc4_std = nn.Linear(hidden_arr[2] + 3 * num_boxes, num_actions)

        # for rotation
        self.fc1_rotation = nn.Linear(32 * 11 * 11, hidden_arr[0])
        self.fc1_rotation_bn = nn.BatchNorm1d(hidden_arr[0])
        self.fc2_rotation = nn.Linear(hidden_arr[0], hidden_arr[1])
        self.fc2_rotation_bn = nn.BatchNorm1d(hidden_arr[1])
        self.fc3_rotation = nn.Linear(hidden_arr[1], hidden_arr[2])
        self.fc3_rotation_bn = nn.BatchNorm1d(hidden_arr[2])
        self.fc4_rotation = nn.Linear(hidden_arr[2] + 3 * num_boxes, 1)

        # init weights
        self.fc4_mean.weight.data.uniform_(-init_w, init_w)
        self.fc4_mean.bias.data.uniform_(-init_w, init_w)
        self.fc4_std.weight.data.uniform_(-init_w, init_w)
        self.fc4_std.bias.data.uniform_(-init_w, init_w)
        self.fc4_rotation.weight.data.uniform_(-init_w, init_w)
        self.fc4_rotation.bias.data.uniform_(-init_w, init_w)

        self.apply(weights_init_)

        self.batch_norm = batch_norm

    def forward(self, state, dims):
        x = F.relu(self.conv1(state))
        if self.batch_norm:
            x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        if self.batch_norm:
            x = self.conv3_bn(x)
        x = x.view(-1, 32 * 11 * 11)

        # mean
        mean = F.relu(self.fc1_mean(x))
        if self.batch_norm:
            mean = self.fc1_mean_bn(mean)
        mean = F.relu(self.fc2_mean(mean))
        if self.batch_norm:
            mean = self.fc2_mean_bn(mean)
        mean = F.relu(self.fc3_mean(mean))
        if self.batch_norm:
            mean = self.fc3_mean_bn(mean)

        z_mean = torch.cat([mean, dims], dim=1)
        z_mean = self.fc4_mean(z_mean)

        # std
        std = F.relu(self.fc1_mean(x))
        if self.batch_norm:
            std = self.fc1_std_bn(std)
        std = F.relu(self.fc2_std(std))
        if self.batch_norm:
            std = self.fc2_std_bn(std)
        std = F.relu(self.fc3_std(std))
        if self.batch_norm:
            std = self.fc3_std_bn(std)

        z_std = torch.cat([std, dims], dim=1)
        z_std = self.fc4_std(z_std)
        z_std = torch.clamp(z_std, LOG_SIG_MIN, LOG_SIG_MAX)

        # rotation
        rotation = F.relu(self.fc1_rotation(x))
        if self.batch_norm:
            rotation = self.fc1_rotation_bn(rotation)
        rotation = F.relu(self.fc2_rotation(rotation))
        if self.batch_norm:
            rotation = self.fc2_rotation_bn(rotation)
        rotation = F.relu(self.fc3_rotation(rotation))
        if self.batch_norm:
            rotation = self.fc3_rotation_bn(rotation)
        rotation = torch.cat([rotation, dims], dim=1)
        rotation_logits = self.fc4_rotation(rotation)

        return z_mean, z_std, rotation_logits

    def sample(self, state, dim):
        mean, log_std, rotation_logits = self.forward(state, dim)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        rotation_probs = nn.functional.softmax(rotation_logits, dim=1)
        rotation_sample = rotation_probs.multinomial(1)

        return action, log_prob, torch.tanh(mean), rotation_sample



class CNN_task2(nn.Module):  #
    def __init__(self, num_actions=2, init_w=3e-3, out_channels=[32, 32, 32], hidden_arr=[500, 100, 50, 9], num_boxes=4,
                 batch_norm=False):
        super(CNN_task2, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=out_channels[0], kernel_size=5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(out_channels[2])

        # 自适应池化层，将不同尺寸输出统一为固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 2))

        # mean
        self.fc1_mean = nn.Linear(32 * 3 * 2, hidden_arr[0])
        self.fc1_mean_bn = nn.BatchNorm1d(hidden_arr[0])
        self.fc2_mean = nn.Linear(hidden_arr[0], hidden_arr[1])
        self.fc2_mean_bn = nn.BatchNorm1d(hidden_arr[1])
        self.fc3_mean = nn.Linear(hidden_arr[1], hidden_arr[2])
        self.fc3_mean_bn = nn.BatchNorm1d(hidden_arr[2])
        self.fc4_mean = nn.Linear(hidden_arr[2] + 3 * num_boxes, num_actions)

        # log-std
        self.fc1_std = nn.Linear(32 * 3 * 2, hidden_arr[0])
        self.fc1_std_bn = nn.BatchNorm1d(hidden_arr[0])
        self.fc2_std = nn.Linear(hidden_arr[0], hidden_arr[1])
        self.fc2_std_bn = nn.BatchNorm1d(hidden_arr[1])
        self.fc3_std = nn.Linear(hidden_arr[1], hidden_arr[2])
        self.fc3_std_bn = nn.BatchNorm1d(hidden_arr[2])
        self.fc4_std = nn.Linear(hidden_arr[2] + 3 * num_boxes, num_actions)

        # for rotation
        self.fc1_rotation = nn.Linear(32 * 3 * 2, hidden_arr[0])
        self.fc1_rotation_bn = nn.BatchNorm1d(hidden_arr[0])
        self.fc2_rotation = nn.Linear(hidden_arr[0], hidden_arr[1])
        self.fc2_rotation_bn = nn.BatchNorm1d(hidden_arr[1])
        self.fc3_rotation = nn.Linear(hidden_arr[1], hidden_arr[2])
        self.fc3_rotation_bn = nn.BatchNorm1d(hidden_arr[2])
        self.fc4_rotation = nn.Linear(hidden_arr[2] + 3 * num_boxes, 1)

        # 初始化权重
        self.fc4_mean.weight.data.uniform_(-init_w, init_w)
        self.fc4_mean.bias.data.uniform_(-init_w, init_w)
        self.fc4_std.weight.data.uniform_(-init_w, init_w)
        self.fc4_std.bias.data.uniform_(-init_w, init_w)
        self.fc4_rotation.weight.data.uniform_(-init_w, init_w)
        self.fc4_rotation.bias.data.uniform_(-init_w, init_w)

        self.apply(weights_init_)
        self.batch_norm = batch_norm

    def forward(self, state, dims):
        x = F.relu(self.conv1(state))
        if self.batch_norm:
            x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        if self.batch_norm:
            x = self.conv3_bn(x)

        x = self.adaptive_pool(x)
        x = x.view(-1, 32 * 3 * 2)

        # mean
        mean = F.relu(self.fc1_mean(x))
        if self.batch_norm:
            mean = self.fc1_mean_bn(mean)
        mean = F.relu(self.fc2_mean(mean))
        if self.batch_norm:
            mean = self.fc2_mean_bn(mean)
        mean = F.relu(self.fc3_mean(mean))
        if self.batch_norm:
            mean = self.fc3_mean_bn(mean)

        z_mean = torch.cat([mean, dims], dim=1)
        z_mean = self.fc4_mean(z_mean)

        # std
        std = F.relu(self.fc1_mean(x))
        if self.batch_norm:
            std = self.fc1_std_bn(std)
        std = F.relu(self.fc2_std(std))
        if self.batch_norm:
            std = self.fc2_std_bn(std)
        std = F.relu(self.fc3_std(std))
        if self.batch_norm:
            std = self.fc3_std_bn(std)

        z_std = torch.cat([std, dims], dim=1)
        z_std = self.fc4_std(z_std)
        z_std = torch.clamp(z_std, LOG_SIG_MIN, LOG_SIG_MAX)

        # rotation
        rotation = F.relu(self.fc1_rotation(x))
        if self.batch_norm:
            rotation = self.fc1_rotation_bn(rotation)
        rotation = F.relu(self.fc2_rotation(rotation))
        if self.batch_norm:
            rotation = self.fc2_rotation_bn(rotation)
        rotation = F.relu(self.fc3_rotation(rotation))
        if self.batch_norm:
            rotation = self.fc3_rotation_bn(rotation)
        rotation = torch.cat([rotation, dims], dim=1)
        rotation_logits = self.fc4_rotation(rotation)

        return z_mean, z_std, rotation_logits

    def sample(self, state, dim):
        mean, log_std, rotation_logits = self.forward(state, dim)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        rotation_probs = nn.functional.softmax(rotation_logits, dim=1)
        rotation_sample = rotation_probs.multinomial(1)

        return action, log_prob, torch.tanh(mean), rotation_sample
