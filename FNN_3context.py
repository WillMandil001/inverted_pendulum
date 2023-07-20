# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn
import torch.optim as optim

class FNN_context(nn.Module):
    def __init__(self, features):
        super(FNN_context, self).__init__()
        self.features = features

        self.device = self.features["device"]

        if self.features["optimizer"] == "adam" or self.features["optimizer"] == "Adam":
            self.optimizer = optim.Adam

        if self.features["criterion"] == "L1":
            self.criterion = nn.L1Loss().to(self.device)
        if self.features["criterion"] == "L2":
            self.criterion = nn.MSELoss().to(self.device)

    def load_model(self, full_model):
        self.fc1 = full_model["fc1"].to(self.device)
        self.fc2 = full_model["fc2"].to(self.device)

    def initialise_model(self):
        self.fc1 = nn.Linear(self.features["tactile_size"], self.features["hidden_size"]).to(self.device)
        self.fc2 = nn.Linear(self.features["hidden_size"], self.features["output_size"]).to(self.device)

    def save_model(self):
        torch.save({'fc1': self.fc1, 'fc2': self.fc2, 'features': self.features}, 
                    self.features["model_dir"] + self.features["model_name"] + "_model" + self.features["model_name_save_appendix"])

    def set_train(self):
        self.fc1.train()
        self.fc2.train()

    def set_test(self):
        self.fc1.eval()
        self.fc2.eval()

    def forward(self, tactiles, actions, target, test=False):
        # concat tactiles and actions]
        x = torch.cat((tactiles, actions), 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        mae = self.criterion(x, target)

        if test is False:
            loss = mae
            loss.backward()

            self.fc1.step()
            self.fc2.step()

        return mae.data.cpu().numpy(), x