'''
CNN with history of 4 boxes as states
'''
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
from model import CNN_task2
from data_generator import get_inverse_rotation,Box
import matplotlib.pyplot as plt

# Set device to CUDA or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_x = 1
MAX_y = 1
    
class PolicyTrainer():
    def __init__(self):
        self.length = 100
        self.width = 100
        self.height  = 100
        self.num_actions = 3
        self.data_maker = Box(self.height,self.width,self.length)
        self.policy = CNN_task2().to(device)
        self.search_range = 6
        self.search = np.arange(0,self.search_range,1)
        self.neg_search = -np.arange(1,self.search_range,1)
        self.search_arr = np.append(self.search,self.neg_search)
        self.name = "Task2"
            
        self.writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def shift(self,action,rotation): #
        x = action[:,0]
        y = action[:,1]
        r = rotation
        x = x*self.length/2 +self.length/2
        y = y*self.width/2 +self.width/2
        return x,y,r
    
    def visualization(self,a):
        plt.imshow(a,cmap='hot',vmin=0,vmax=self.height)
        plt.savefig('Box_data/evaluation.jpg')    
        

    def calculate_stability(self, i, j, ldc, dimn, ldc_x=0, ldc_y=0, rotation=0):  # 
        ldc = ldc.T
        temp_i, temp_j = j, i  # Switch
        i, j = temp_i, temp_j
    
        rotated_shape = get_inverse_rotation(np.array(dimn), rotation)
        height = rotated_shape[2]
    
        is_feasible = False
        found_flat, found_non_flat = 0, 0
        if (j >= self.length * ldc_x) and (j + rotated_shape[0] <= self.length * (ldc_x + 1)) and \
           (i >= self.width * ldc_y) and (i + rotated_shape[1] <= self.width * (ldc_y + 1)):
        
            level = ldc[i, j]
            if level + height <= self.height:
                is_feasible = True
                # Check
                if len(np.unique(ldc[i:i + rotated_shape[1], j:j + rotated_shape[0]])) == 1:
                    stab_score = 1
                    found_flat = 1

                if not found_flat:
                    corners = [ldc[i, j], ldc[i + rotated_shape[1] - 1, j], 
                               ldc[i, j + rotated_shape[0] - 1], ldc[i + rotated_shape[1] - 1, j + rotated_shape[0] - 1]]
                    if np.max(corners) == np.min(corners) and np.max(corners) == np.max(ldc[i:i + rotated_shape[1], j:j + rotated_shape[0]]):
                        stab_score = -np.sum(np.max(corners) - ldc[i:i + rotated_shape[1], j:j + rotated_shape[0]]) / (rotated_shape[0] * rotated_shape[1] * self.height)
                        found_non_flat = 1

        if found_flat or found_non_flat:
            min_j = np.max((self.length * ldc_x, j - 1))
            max_j = np.min((self.length * (ldc_x + 1), j + rotated_shape[0]))
            min_i = np.max((self.width * ldc_y, i - 1))
            max_i = np.min((self.width * (ldc_y + 1), i + rotated_shape[1]))

            # Upper bound
            if i == ldc_y * self.width:
                upper_border = (self.height - 1 + np.ones_like(ldc[min_i, j:(j + int(rotated_shape[0]))])).tolist()
            else:
                upper_border = ldc[min_i, j:(j + int(rotated_shape[0]))].tolist()

            unique_ht = np.unique(upper_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if unique_ht[0] == level:
                    stab_score -= 2
                elif unique_ht[0] == self.height:
                    stab_score += 1.5
                else:
                    score = 1. - abs(unique_ht[0] - (level + height)) / self.height
                    if unique_ht[0] > level:
                        stab_score += 1.5 * score
                    else:
                        stab_score += 0.75 * score
            else:
                stab_score += 0.25 * (1. - len(unique_ht) / height)
                stab_score += 0.25 * (1. - sum(abs(ht - (level + height)) for ht in unique_ht) / (height * len(unique_ht)))
                stab_score += 0.50 * sum(ht != level for ht in unique_ht) / len(unique_ht)

            del upper_border

            # Left bound
            if j == ldc_x * self.length:
                left_border = (self.height - 1 + np.ones_like(ldc[i:(i + int(rotated_shape[1])), min_j])).tolist()
            else:
                left_border = ldc[i:(i + int(rotated_shape[1])), min_j].tolist()

            unique_ht = np.unique(left_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if unique_ht[0] == level:
                    stab_score -= 2
                elif unique_ht[0] == self.height:
                    stab_score += 1.5
                else:
                    score = 1. - abs(unique_ht[0] - (level + height)) / self.height
                    if unique_ht[0] > level:
                        stab_score += 1.5 * score
                    else:
                        stab_score += 0.75 * score
            else:
                stab_score += 0.25 * (1. - len(unique_ht) / height)
                stab_score += 0.25 * (1. - sum(abs(ht - (level + height)) for ht in unique_ht) / (height * len(unique_ht)))
                stab_score += 0.50 * sum(ht != level for ht in unique_ht) / len(unique_ht)

            del left_border

            # lower bound
            if (i + rotated_shape[1] < self.width * (ldc_y + 1)):
                lower_border = ldc[max_i, j:(j + int(rotated_shape[0]))].tolist()
            else:
                lower_border = (self.height - 1 + np.ones_like(ldc[max_i - 1, j:(j + int(rotated_shape[0]))])).tolist()

            unique_ht = np.unique(lower_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if lower_border[0] == level:
                    stab_score -= 2
                elif lower_border[0] == self.height:
                    stab_score += 1.5
                else:
                    score = 1. - abs(unique_ht[0] - (level + height)) / self.height
                    if unique_ht[0] > level:
                        stab_score += 1.5 * score
                    else:
                        stab_score += 0.75 * score
            else:
                stab_score += 0.25 * (1. - len(unique_ht) / height)
                stab_score += 0.25 * (1. - sum(abs(ht - (level + height)) for ht in unique_ht) / (height * len(unique_ht)))
                stab_score += 0.50 * sum(ht != level for ht in unique_ht) / len(unique_ht)

            del lower_border

            # right bound
            if (j + rotated_shape[0] < (ldc_x + 1) * self.length):
                right_border = ldc[i:(i + int(rotated_shape[1])), max_j].tolist()
            else:
                right_border = (self.height - 1 + np.ones_like(ldc[i:(i + int(rotated_shape[1])), max_j - 1])).tolist()

            unique_ht = np.unique(right_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if right_border[0] == level:
                    stab_score -= 2
                elif right_border[0] == self.height:
                    stab_score += 1.5
                else:
                    score = 1. - abs(unique_ht[0] - (level + height)) / self.height
                    if unique_ht[0] > level:
                        stab_score += 1.5 * score
                    else:
                        stab_score += 0.75 * score
            else:
                stab_score += 0.25 * (1. - len(unique_ht) / height)
                stab_score += 0.25 * (1. - sum(abs(ht - (level + height)) for ht in unique_ht) / (height * len(unique_ht)))
                stab_score += 0.50 * sum(ht != level for ht in unique_ht) / len(unique_ht)

            del right_border

            # Continuity
            if i == ldc_y * self.width:
                stab_score += 0.02
            else:
                if (j == ldc_x * self.length):
                    stab_score += 0.01
                if (j + rotated_shape[0]) == (ldc_x + 1) * self.length:
                    stab_score += 0.01

            if i + rotated_shape[1] == self.width * (ldc_y + 1):
                stab_score += 0.02
            else:
                if (j == ldc_x * self.length):
                    stab_score += 0.01
                if (j + rotated_shape[0]) == (ldc_x + 1) * self.length:
                    stab_score += 0.01

            if j == ldc_x * self.length:
                stab_score += 0.02
            else:
                if (i == ldc_y * self.width):
                    stab_score += 0.01
                if (i + rotated_shape[1] == self.width * (ldc_y + 1)):
                    stab_score += 0.01

            if j + rotated_shape[0] == (ldc_x + 1) * self.length:
                stab_score += 0.02
            else:
                if (i == ldc_y * self.width):
                    stab_score += 0.01
                if (i + rotated_shape[1] == self.width * (ldc_y + 1)):
                    stab_score += 0.01

            stab_score -= ldc_x / MAX_x + ldc_y / MAX_y
            stab_score -= 0.05 * (i / ((ldc_y + 1) * self.width) + j / ((ldc_x + 1) * self.length))

        else:
            stab_score = -10

        return stab_score


                    
    def get_position(self,state,dims):   #        
        normalized_state = torch.FloatTensor(state)/self.height
        normalized_state = normalized_state.unsqueeze(0)

        normalized_dim = torch.FloatTensor(dims)/self.height
        normalized_dim = normalized_dim.unsqueeze(0)  # Normalized input
        
        if torch.cuda.is_available():  #GPU
            normalized_state = normalized_state.to(device)
            normalized_dim = normalized_dim.to(device)

        current_dim = np.array(dims[:3], dtype=np.uint16)
        action, mean, stddev, rotation = self.policy.sample(normalized_state.float(),normalized_dim.float())  # Policies
        offset_x, offset_y,temp_rot = self.shift(action,rotation)

        temp_rot = int(temp_rot[0])
        offset_x = int(offset_x.cpu()[0])
        offset_y = int(offset_y.cpu()[0])

        stability_score = self.calculate_stability(offset_x, offset_y, state[0,:,:], dimn = current_dim, ldc_x = 0, ldc_y = 0,rotation = temp_rot)
        
        final_rotation=temp_rot
        if stability_score <= 0:
            base_x, base_y = offset_x, offset_y

            # Search
            for delta_x in self.search_arr:
                for delta_y in self.search_arr:
                    for rotation_candidate in range(6):
                        search_stability = self.calculate_stability(base_x + delta_x, base_y + delta_y, state[0,:,:], dimn = current_dim, ldc_x = 0, ldc_y = 0,rotation = rotation_candidate)
                        if search_stability > stability_score:
                            stability_score = search_stability
                            offset_x = base_x + delta_x
                            offset_y = base_y + delta_y
                            final_rotation = rotation_candidate
                            
        return offset_x, offset_y, stability_score, final_rotation
    
    
    def perform_step(self, state, action, dims, rotation):   #
        inverse_dims = get_inverse_rotation(dims,rotation)
        length, width, height = inverse_dims

        pos_x, pos_y = action

        temp_state = np.copy(state[0,:,:])
        state = np.roll(state,shift=1,axis=0)

        state[0,:,:] = temp_state
        state[0, pos_x:pos_x+length, pos_y:pos_y+width] += height # Update
        return state    
    
    
    def evaluate(self):   #
        current_dim = np.zeros((12))
        data = self.data_maker.get_data_dict(flatten=False)
        dims_list = []  # Init

        for i in range(len(data)):
            current_dim = np.roll(current_dim, shift=3)
            current_dim[:3] = data[i][1]
            dims_list.append(current_dim)  # Update dims

        total_volume = 0
        state = np.zeros((4, self.length, self.width))
        wall_volume = 0
        wall_score = 0
        packman_volume = 0
        search_positions = []

        for i in range(len(dims_list)):
            dim = np.array(dims_list[i][:3], dtype=np.uint16)
            length, breadth, height = dim
            x_pos, y_pos, stability_score, rotation = self.get_position(state, dims_list[i])

            if stability_score != -10:
                state = self.perform_step(state, [x_pos, y_pos], dim, rotation)
                search_positions.extend([[x_pos, y_pos, rotation], [x_pos + length, y_pos, rotation], [x_pos, y_pos + breadth, rotation]])
                total_volume += dim[0] * dim[1] * dim[2]
                packman_volume += dim[0] * dim[1] * dim[2]
        
            if stability_score == -10:
                max_stability = -10
                best_x, best_y, best_r = 0, 0, 0

                # Search
                for base_x, base_y, base_r in search_positions:
                    wall_score = self.calculate_stability(base_x, base_y, state[0, :, :], dimn=dim, ldc_x = 0, ldc_y = 0, rotation = base_r)
                    if wall_score > max_stability:
                        max_stability = wall_score
                        best_x, best_y, best_r = base_x, base_y, base_r
  
                if max_stability != -10:
                    x_pos = best_x
                    y_pos = best_y
                    rotation = best_r
                    state = self.perform_step(state, [x_pos, y_pos], dim, rotation)
                    search_positions.extend([[x_pos, y_pos, rotation], [x_pos + length, y_pos, rotation], [x_pos, y_pos + breadth, rotation]])
                    total_volume += dim[0] * dim[1] * dim[2]
                    wall_volume += dim[0] * dim[1] * dim[2]

        print(
            f"Total volume: {total_volume / (self.height * self.length * self.width) * 100:.2f}%",
            f"Packman volume: {packman_volume / (self.height * self.length * self.width) * 100:.2f}%",
            f"Wall volume: {wall_volume / (self.height * self.length * self.width) * 100:.2f}"
        )
        self.visualization(state[0, :, :])
        


if __name__ == "__main__":
    if not os.path.exists('./Models'):
        os.makedirs('./Models')

    Trainer = PolicyTrainer()
    Trainer.evaluate()
