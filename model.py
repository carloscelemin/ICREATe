import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyEnsemble(nn.Module):
    def __init__(self, config):
        super(PolicyEnsemble, self).__init__()
        # First layer is shared
        self.fc0 = nn.Linear(config['GENERAL'].getint('input_size'), 16)
        # Ensemble
        self.fc1_0 = nn.Linear(16, 16)
        self.fc1_1 = nn.Linear(16, 32)
        self.fc1_2 = nn.Linear(16, 64)
        self.fc1_3 = nn.Linear(16, 128)
        self.fc1_4 = nn.Linear(16, 256)

        self.fc2_0 = nn.Linear(16, 8)
        self.fc2_1 = nn.Linear(32, 16)
        self.fc2_2 = nn.Linear(64, 32)
        self.fc2_3 = nn.Linear(128, 64)
        self.fc2_4 = nn.Linear(256, 128)

        self.fc3_0 = nn.Linear(8, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_0.bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc3_0.weight, mean=0.0, std=0.01)
        self.fc3_1 = nn.Linear(16, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_1.bias, mean=0.25, std=0.01)
        torch.nn.init.normal_(self.fc3_1.weight, mean=0.0, std=0.01)
        self.fc3_2 = nn.Linear(32, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_2.bias, mean=-0.5, std=0.01)
        torch.nn.init.normal_(self.fc3_2.weight, mean=0.0, std=0.01)
        self.fc3_3 = nn.Linear(64, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_3.bias, mean=01.75, std=0.01)
        torch.nn.init.normal_(self.fc3_3.weight, mean=0.0, std=0.01)
        self.fc3_4 = nn.Linear(128, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_4.bias, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.fc3_4.weight, mean=0.0, std=0.01)
        # Head for predicting ambiguities
        self.fc1_a = nn.Linear(16,64)
        self.fc2_a = nn.Linear(64,32)
        self.fc3_a = nn.Linear(32,1)
        torch.nn.init.normal_(self.fc3_a.bias, mean=-4.0, std=0.01)
        torch.nn.init.normal_(self.fc3_a.weight, mean=0.0, std=0.01)

        return

    def forward(self,x):
        l0 = self.fc0(x)
        l0 = F.relu(l0)
        # head 0
        h0 = self.fc1_0(l0)
        h0 = F.relu(h0)
        h0 = self.fc2_0(h0)
        h0 = F.relu(h0)
        h0 = self.fc3_0(h0)
        #h0 = F.sigmoid(h0)
        # head 1
        h1 = self.fc1_1(l0)
        h1 = F.relu(h1)
        h1 = self.fc2_1(h1)
        h1 = F.relu(h1)
        h1 = self.fc3_1(h1)
        #h1 = F.sigmoid(h1)
        # head 2
        h2 = self.fc1_2(l0)
        h2 = F.relu(h2)
        h2 = self.fc2_2(h2)
        h2 = F.relu(h2)
        h2 = self.fc3_2(h2)
        #h2 = F.sigmoid(h2)
        # head 3
        h3 = self.fc1_3(l0)
        h3 = F.relu(h3)
        h3 = self.fc2_3(h3)
        h3 = F.relu(h3)
        h3 = self.fc3_3(h3)
        #h3 = F.sigmoid(h3)
        # head 4
        h4 = self.fc1_4(l0)
        h4 = F.relu(h4)
        h4 = self.fc2_4(h4)
        h4 = F.relu(h4)
        h4 = self.fc3_4(h4)
        #h4 = F.sigmoid(h4)
        # head a
        ha = self.fc1_a(l0)
        ha = F.relu(ha)
        ha = self.fc2_a(ha)
        ha = F.relu(ha)
        ha = self.fc3_a(ha)
        ha = torch.sigmoid(ha)

        return h0, h1, h2, h3, h4, ha

    def predict(self,x):
        heads = self.forward(x)
        sig = lambda h: torch.softmax(h,1)
        somx_list = list(map(sig,heads[:-1])) #converting all the heads predictions to probabilities
        ensemble = torch.stack(somx_list)
        action_pred = torch.argmax(torch.sum(ensemble,0),1) # action chosen for each input

        ensemble_log = torch.stack(heads[:-1])
        std_action = torch.sqrt(torch.mean(torch.var(ensemble_log, 0),1))

        return action_pred, std_action, heads[-1]


class PolicyRAMEnsemble(nn.Module):
    def __init__(self, config):
        super(PolicyRAMEnsemble, self).__init__()
        # First layer is shared
        self.fc0 = nn.Linear(config['GENERAL'].getint('input_size'), 128)
        # Ensemble
        self.fc1_0 = nn.Linear(128, 128)
        self.fc1_1 = nn.Linear(128, 128)
        self.fc1_2 = nn.Linear(128, 128)
        self.fc1_3 = nn.Linear(128, 128)
        self.fc1_4 = nn.Linear(128, 128)

        self.fc2_0 = nn.Linear(128, 32)
        self.fc2_1 = nn.Linear(128, 32)
        self.fc2_2 = nn.Linear(128, 64)
        self.fc2_3 = nn.Linear(128, 128)
        self.fc2_4 = nn.Linear(128, 128)

        self.fc3_0 = nn.Linear(32, 16)
        self.fc3_1 = nn.Linear(32, 16)
        self.fc3_2 = nn.Linear(64, 32)
        self.fc3_3 = nn.Linear(128, 64)
        self.fc3_4 = nn.Linear(128, 64)

        self.fc4_0 = nn.Linear(16, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_0.bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc3_0.weight, mean=0.0, std=0.01)
        self.fc4_1 = nn.Linear(16, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_1.bias, mean=0.25, std=0.01)
        torch.nn.init.normal_(self.fc3_1.weight, mean=0.0, std=0.01)
        self.fc4_2 = nn.Linear(32, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_2.bias, mean=-0.5, std=0.01)
        torch.nn.init.normal_(self.fc3_2.weight, mean=0.0, std=0.01)
        self.fc4_3 = nn.Linear(64, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_3.bias, mean=01.75, std=0.01)
        torch.nn.init.normal_(self.fc3_3.weight, mean=0.0, std=0.01)
        self.fc4_4 = nn.Linear(64, config['GENERAL'].getint('output_size'))
        torch.nn.init.normal_(self.fc3_4.bias, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.fc3_4.weight, mean=0.0, std=0.01)
        # Head for predicting ambiguities
        self.fc1_a = nn.Linear(128, 128)
        self.fc2_a = nn.Linear(128,64)
        self.fc3_a = nn.Linear(64,32)
        self.fc4_a = nn.Linear(32,1)
        torch.nn.init.normal_(self.fc4_a.bias, mean=-4.0, std=0.01)
        torch.nn.init.normal_(self.fc4_a.weight, mean=0.0, std=0.01)

        return

    def forward(self,x):
        l0 = self.fc0(x)
        l0 = F.relu(l0)
        # head 0
        h0 = self.fc1_0(l0)
        h0 = F.relu(h0)
        h0 = self.fc2_0(h0)
        h0 = F.relu(h0)
        h0 = self.fc3_0(h0)
        h0 = F.relu(h0)
        h0 = self.fc4_0(h0)
        #h0 = F.sigmoid(h0)
        # head 1
        h1 = self.fc1_1(l0)
        h1 = F.relu(h1)
        h1 = self.fc2_1(h1)
        h1 = F.relu(h1)
        h1 = self.fc3_1(h1)
        h1 = F.relu(h1)
        h1 = self.fc4_1(h1)
        #h1 = F.sigmoid(h1)
        # head 2
        h2 = self.fc1_2(l0)
        h2 = F.relu(h2)
        h2 = self.fc2_2(h2)
        h2 = F.relu(h2)
        h2 = self.fc3_2(h2)
        h2 = F.relu(h2)
        h2 = self.fc4_2(h2)
        #h2 = F.sigmoid(h2)
        # head 3
        h3 = self.fc1_3(l0)
        h3 = F.relu(h3)
        h3 = self.fc2_3(h3)
        h3 = F.relu(h3)
        h3 = self.fc3_3(h3)
        h3 = F.relu(h3)
        h3 = self.fc4_3(h3)
        #h3 = F.sigmoid(h3)
        # head 4
        h4 = self.fc1_4(l0)
        h4 = F.relu(h4)
        h4 = self.fc2_4(h4)
        h4 = F.relu(h4)
        h4 = self.fc3_4(h4)
        h4 = F.relu(h4)
        h4 = self.fc4_4(h4)
        #h4 = F.sigmoid(h4)
        # head a
        ha = self.fc1_a(x)
        ha = F.sigmoid(ha)
        ha = self.fc2_a(ha)
        ha = F.sigmoid(ha)
        ha = self.fc3_a(ha)
        ha = F.sigmoid(ha)
        ha = self.fc4_a(ha)
        ha = torch.sigmoid(ha)

        return h0, h1, h2, h3, h4, ha

    def predict(self,x):
        heads = self.forward(x)
        sig = lambda h: torch.softmax(h,1)
        somx_list = list(map(sig,heads[:-1])) #converting all the heads predictions to probabilities
        ensemble = torch.stack(somx_list)
        action_pred = torch.argmax(torch.sum(ensemble,0),1) # action chosen for each input

        ensemble_log = torch.stack(heads[:-1])
        std_action = torch.sqrt(torch.mean(torch.var(ensemble_log, 0),1))

        return action_pred, std_action, heads[-1]