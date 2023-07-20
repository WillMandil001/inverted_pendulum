# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import re
import copy
import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

model_path      = "/home/willow/Robotics/SPOTS/models/saved_models/ACTP/model_04_07_2023_15_18/ACTP_model"  # .zip
data_save_path  = "/home/willow/Robotics/SPOTS/models/saved_models/ACTP/model_04_07_2023_15_18/"
# model_path      = "/home/willow/Robotics/SPOTS/models/saved_models/ACTP/model_04_07_2023_15_35/ACTP_model"  # .zip
# data_save_path  = "/home/willow/Robotics/SPOTS/models/saved_models/ACTP/model_04_07_2023_15_35/"
test_data_dir   = "/home/willow/Robotics/datasets/PRI/MarkedHeavyBox/Dataset_2c_15p/test_examples_formatted/"
scaler_dir      = "/home/willow/Robotics/datasets/PRI/MarkedHeavyBox/scalers/"

seed = 42
epochs = 200
batch_size = 256
learning_rate = 1e-3
context_frames = 2
sequence_length = 17

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(test_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map):
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(test_data_dir + value[0])
        tactile_data = np.load(test_data_dir + value[1])
        experiment_number = np.load(test_data_dir + value[4])
        time_steps = np.load(test_data_dir + value[5])
        meta = test_data_dir + value[5]
        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps, meta]

class ACTP(nn.Module):
    def __init__(self):
        super(ACTP, self).__init__()
        self.lstm1 = nn.LSTM(48, 200).to(device)  # tactile
        self.lstm2 = nn.LSTM(200 + 48, 200).to(device)  # tactile
        self.fc1 = nn.Linear(200 + 48, 200).to(device)  # tactile + pos
        self.fc2 = nn.Linear(200, 48).to(device)  # tactile + pos
        self.tan_activation = nn.Tanh().to(device)
        self.relu_activation = nn.ReLU().to(device)

    def forward(self, tactiles, actions):
        state = actions[0]
        state.to(device)
        batch_size__ = tactiles.shape[1]
        outputs = []
        hidden1 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
        hidden2 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))

        for index, (sample_tactile, sample_action,) in enumerate(zip(tactiles.squeeze()[:-1], actions.squeeze()[1:])):
            # 2. Run through lstm:
            if index > context_frames-1:
                out4 = out4.squeeze()
                out1, hidden1 = self.lstm1(out4.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), out4), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                outputs.append(out4.squeeze())
            else:
                out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), sample_tactile), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                last_output = out4

        outputs = [last_output] + outputs
        return torch.stack(outputs)


class ModelTester:
    def __init__(self):
        self.test_full_loader = BG.load_full_data()

        # load model:
        self.full_model = ACTP()
        self.full_model = torch.load(model_path)
        self.full_model.eval()

        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

    def test_full_model(self):
        self.objects = []
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp = 0

        for index, batch_features in enumerate(self.test_full_loader):
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)
            tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.

            experiment_number = [int(re.sub('\D', '', i)[3:]) for i in batch_features[4]]
            time_steps = batch_features[3][:, context_frames:]

            self.prediction_data.append([tactile_predictions.cpu().detach(), tactile[context_frames:].cpu().detach(), experiment_number, time_steps.cpu().detach()])
            print ("currently testing trial number: ", str(self.current_exp))
            self.calc_trial_performance()
            self.save_predictions(self.current_exp)
            self.prediction_data = []
            self.current_exp += 1

        print ("Hello :D ")

        self.calc_test_performance()

    def save_predictions(self, experiment_to_test):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''

        plot_save_dir = data_save_path + "SCALED_test_plots_" + str (experiment_to_test)
        try:
            os.mkdir (plot_save_dir)
        except:
            "directory already exists"

        np.save (plot_save_dir + '/trial_groundtruth_data', np.array (self.prediction_data[-1][1]))
        np.save (plot_save_dir + '/trial_predictions_data', np.array (self.prediction_data[-1][0]))

    def calc_trial_performance(self):
        mae_loss = 0.0
        mae_losses = [0.0 for i in range(15)]

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1
                mae_loss += self.criterion (batch_set[0], batch_set[1]).item()
                for pred_step in range(0, 15):
                    mae_losses[pred_step] += self.criterion(batch_set[0][pred_step], batch_set[1][pred_step]).item()

        self.performance_data.append([mae_loss/index, [i/index for i in mae_losses]])

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(["test loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1][1 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 2: ", (sum([i[1][2 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 3: ", (sum([i[1][3 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 4: ", (sum([i[1][4 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ", (sum([i[1][5 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 6: ", (sum([i[1][6 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 7: ", (sum([i[1][7 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 8: ", (sum([i[1][8 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 9: ", (sum([i[1][9 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ", (sum([i[1][10 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 11: ", (sum([i[1][11 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 12: ", (sum([i[1][12 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 13: ", (sum([i[1][13 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 14: ", (sum([i[1][14 - 1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 15: ", (sum([i[1][15 - 1] for i in self.performance_data]) / len (self.performance_data))])

        [print(i) for i in performance_data_full]
        np.save(data_save_path + 'model_performance_loss_data', np.asarray (performance_data_full))

        # save performance data as txt
        with open(data_save_path + 'model_performance_loss_data.txt', 'w') as f:
            for item in performance_data_full:
                f.write("%s\n" % item)


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTester()
    MT.test_full_model()