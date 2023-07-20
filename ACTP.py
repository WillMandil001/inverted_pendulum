# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn
import torch.optim as optim
import universal_networks.utils as utility_prog

class ACTP(nn.Module):
    def __init__(self, features):
        super(ACTP, self).__init__()
        self.features = features

        self.device = self.features["device"]

        if self.features["optimizer"] == "adam" or self.features["optimizer"] == "Adam":
            self.optimizer = optim.Adam

        if self.features["criterion"] == "L1":
            self.criterion = nn.L1Loss().to(self.device)
        if self.features["criterion"] == "L2":
            self.criterion = nn.MSELoss().to(self.device)

    def load_model(self, full_model):
        self.lstm1 = full_model["lstm1"].to(self.device)
        self.lstm2 = full_model["lstm2"].to(self.device)
        self.fc1 = full_model["fc1"].to(self.device)
        self.fc2 = full_model["fc2"].to(self.device)
        self.tan_activation = nn.Tanh().to(self.device)
        self.relu_activation = nn.ReLU().to(self.device)

    def initialise_model(self):
        self.lstm1 = nn.LSTM(self.features["tactile_size"], self.features["hidden_size"]).to(self.device)
        self.lstm2 = nn.LSTM(self.features["hidden_size"] + self.features["state_action_size"], self.features["hidden_size"]).to(self.device)
        self.fc1 = nn.Linear(self.features["hidden_size"] + self.features["tactile_size"], self.features["hidden_size"]).to(self.device)
        self.fc2 = nn.Linear(self.features["hidden_size"], self.features["tactile_size"]).to(self.device)
        self.tan_activation = nn.Tanh().to(self.device)
        self.relu_activation = nn.ReLU().to(self.device)

    def save_model(self):
        torch.save({'lstm1': self.lstm1, 'lstm2': self.lstm2, 'fc1': self.fc1, 'fc2': self.fc2, 'features': self.features}, 
                    self.features["model_dir"] + self.features["model_name"] + "_model" + self.features["model_name_save_appendix"])

    def set_train(self):
        self.lstm1.train()
        self.lstm2.train()
        self.fc1.train()
        self.fc2.train()

    def set_test(self):
        self.lstm1.eval()
        self.lstm2.eval()
        self.fc1.eval()
        self.fc2.eval()

    def forward(self, tactiles, actions, test=False):
        mae = 0

        state = actions[0]
        state.to(self.device)
        outputs = []
        self.hidden1 = (torch.zeros(1, self.features["batch_size"], self.features["hidden_size"], device=torch.device(self.device)), torch.zeros(1, self.features["batch_size"], self.features["hidden_size"], device=torch.device(self.device)))
        self.hidden2 = (torch.zeros(1, self.features["batch_size"], self.features["hidden_size"], device=torch.device(self.device)), torch.zeros(1, self.features["batch_size"], self.features["hidden_size"], device=torch.device(self.device)))

        for index, (sample_tactile, sample_action,) in enumerate(zip(tactiles.squeeze()[:-1], actions.squeeze()[1:])):
            # 2. Run through lstm:
            if index > self.features["context_frames"] - 1:
                out4 = out4.squeeze()
                out1, hidden1 = self.lstm1(out4.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), out4), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                outputs.append(out4.squeeze())

                mae += self.criterion(out4, tactiles[index + 1])  # prediction model

            else:
                out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), sample_tactile), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                last_output = out4

                mae += self.criterion(out4, tactiles[index + 1])  # prediction model

        outputs = [last_output] + outputs

        if test is False:
            loss = mae
            loss.backward()

            self.lstm1.step()
            self.lstm2.step()
            self.fc1.step()
            self.fc2.step()

        return mae.data.cpu().numpy() / (self.features["n_past"] + self.features["n_future"]), torch.stack(outputs)
