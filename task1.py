import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from data_generator import Box
from model import CNN
import matplotlib.pyplot as plt

# Set device to CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperienceReplay:
    def __init__(self):
        self.capacity=1e4
        self.buffer = []
        self.pointer = 0

    def store(self, experience):
        if len(self.buffer) == self.capacity:
            self.buffer[int(self.pointer)] = experience
            self.pointer = (self.pointer + 1) % self.capacity
        else:
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states, dims, actions, rotations = [], [], [], []
        for idx in indices:
            state, dim, action, rotation = self.buffer[idx]
            states.append(np.array(state, copy=False))
            dims.append(np.array(dim, copy=False))
            actions.append(np.array(action, copy=False))
            rotations.append(rotation)

        return np.array(states), np.array(dims), np.array(actions), np.array(rotations)

class PolicyTrainer:
    def __init__(self):
        self.length = 100
        self.width = 100
        self.height = 100
        self.policy_model = CNN().to(device)
        self.model_name = "Task1"
        self.episodes = 20000
        self.learning_rate = 5e-3
        self.batch_size = 20
        self.save_path = "./Models"

        self.writer = SummaryWriter(log_dir=f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    def shift(self, action, rotation):
        x = action[:, 0]
        y = action[:, 1]
        r = rotation
        x = x * self.length / 2 + self.length / 2
        y = y * self.width / 2 + self.width / 2
        return x, y, r

    def train(self):
        optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        replay_buffer = ExperienceReplay()

        start_episode = 0

        for episode in tqdm(range(self.episodes)):
            self.data_generator = Box(self.height, self.width, self.length)
            data = np.array(self.data_generator.get_data_dict(flatten=False), dtype=object)
            state = np.zeros((4, self.length, self.width))
            dim = np.zeros(12)
            for i in range(len(data)):
                state = np.roll(state, axis=0, shift=1)
                state[0, :, :] = data[i][0]
                dim = np.roll(dim, shift=3)
                dim[:3] = data[i][1]
                action = data[i][2][:2]
                rotation = data[i][3]
                replay_buffer.store([state, dim, action, rotation])

            if len(replay_buffer.buffer) >= self.batch_size:
                states_batch, dims_batch, actions_batch, rotations_batch = replay_buffer.sample_batch(self.batch_size)
                states_batch = torch.FloatTensor(states_batch) / self.height
                dims_batch = torch.FloatTensor(dims_batch) / self.height
                actions_batch = torch.from_numpy(actions_batch)
                rotations_batch = torch.from_numpy(rotations_batch)

                if torch.cuda.is_available():
                    states_batch = states_batch.to(device)
                    dims_batch = dims_batch.to(device)
                    actions_batch = actions_batch.to(device)
                    rotations_batch = rotations_batch.to(device)

                action_pred, mean, std, rotation_pred = self.policy_model.sample(states_batch.float(), dims_batch.float())
                x, y, temp_rotation = self.shift(action_pred, rotation_pred)

                pred = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
                rotation_pred = torch.Tensor(temp_rotation)

                optimizer.zero_grad()
                loss_action = F.mse_loss(pred, actions_batch.float())
                loss_rotation = F.mse_loss(rotation_pred.squeeze(1), rotations_batch.float())
                total_loss = loss_action + loss_rotation
                total_loss.backward()

                self.writer.add_scalar('Loss', total_loss.item(), episode + start_episode)
                optimizer.step()

            if episode % 5000 == 0 and episode != 0:
                model_path = os.path.join(self.save_path, self.model_name)
                os.makedirs(model_path, exist_ok=True)
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.policy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, os.path.join(model_path, f"{episode}.pt"))
        
        self.writer.close()

if __name__ == "__main__":
    if not os.path.exists('./Models'):
        os.makedirs('./Models')

    trainer = PolicyTrainer()
    trainer.train()


