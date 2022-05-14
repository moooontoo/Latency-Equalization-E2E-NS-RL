import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 60)
        self.fc1_p = nn.Linear(60, 20)
        self.fc1_c = nn.Linear(60, 20)
        self.fc2_p = nn.Linear(60, 20)
        self.fc2_c = nn.Linear(60, 20)
        self.fc3_p = nn.Linear(60, 20)
        self.fc3_c = nn.Linear(60, 20)
        self.fc4_p = nn.Linear(60, 20)
        self.fc4_c = nn.Linear(60, 20)
        self.fc5_p = nn.Linear(60, 20)
        self.fc5_c = nn.Linear(60, 20)
        self.fc6_p = nn.Linear(60, 20)
        self.fc6_c = nn.Linear(60, 20)
        self.fc7_p = nn.Linear(60, 20)
        self.fc7_c = nn.Linear(60, 20)
        self.fc8_p = nn.Linear(60, 20)
        self.fc8_c = nn.Linear(60, 20)
        self.fc9_p = nn.Linear(60, 20)
        self.fc9_c = nn.Linear(60, 20)
        self.fc10_p = nn.Linear(60, 20)
        self.fc10_c = nn.Linear(60, 20)
        self.fc11_p = nn.Linear(60, 20)
        self.fc11_c = nn.Linear(60, 20)
        self.fc12_p = nn.Linear(60, 20)
        self.fc12_c = nn.Linear(60, 20)
        self.fc13_p = nn.Linear(60, 20)
        self.fc13_c = nn.Linear(60, 20)
        self.fc14_p = nn.Linear(60, 20)
        self.fc14_c = nn.Linear(60, 20)
        self.fc15_p = nn.Linear(60, 20)
        self.fc15_c = nn.Linear(60, 20)
        self.fc16_p = nn.Linear(60, 20)
        self.fc16_c = nn.Linear(60, 20)
        self.fc17_p = nn.Linear(60, 20)
        self.fc17_c = nn.Linear(60, 20)
        self.fc18_p = nn.Linear(60, 20)
        self.fc18_c = nn.Linear(60, 20)
        self.fc19_p = nn.Linear(60, 20)
        self.fc19_c = nn.Linear(60, 20)
        self.fc20_p = nn.Linear(60, 20)
        self.fc20_c = nn.Linear(60, 20)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 1, 20) # 1*1*3
        x = F.relu(self.fc1(x)) # 1*1*180

        x1_p = self.dropout(F.relu(self.fc1_p(x))).view(-1,1,20) # 20
        x1_c = self.dropout(F.relu(self.fc1_c(x))).view(-1,1,20) # 6
        x2_p = self.dropout(F.relu(self.fc2_p(x))).view(-1,1,20) # 20
        x2_c = self.dropout(F.relu(self.fc2_c(x))).view(-1,1,20) # 6
        x3_p = self.dropout(F.relu(self.fc3_p(x))).view(-1,1,20) # 20
        x3_c = self.dropout(F.relu(self.fc3_c(x))).view(-1,1,20) # 6
        x4_p = self.dropout(F.relu(self.fc4_p(x))).view(-1,1,20) # 20
        x4_c = self.dropout(F.relu(self.fc4_c(x))).view(-1,1,20) # 6
        x5_p = self.dropout(F.relu(self.fc5_p(x))).view(-1,1,20) # 20
        x5_c = self.dropout(F.relu(self.fc5_c(x))).view(-1,1,20) # 6
        x6_p = self.dropout(F.relu(self.fc6_p(x))).view(-1,1,20) # 20
        x6_c = self.dropout(F.relu(self.fc6_c(x))).view(-1,1,20) # 6
        x7_p = self.dropout(F.relu(self.fc7_p(x))).view(-1,1,20) # 20
        x7_c = self.dropout(F.relu(self.fc7_c(x))).view(-1,1,20) # 6
        x8_p = self.dropout(F.relu(self.fc8_p(x))).view(-1,1,20) # 20
        x8_c = self.dropout(F.relu(self.fc8_c(x))).view(-1,1,20) # 6
        x9_p = self.dropout(F.relu(self.fc9_p(x))).view(-1,1,20) # 20
        x9_c = self.dropout(F.relu(self.fc9_c(x))).view(-1,1,20) # 6
        x10_p = self.dropout(F.relu(self.fc10_p(x))).view(-1,1,20) # 20
        x10_c = self.dropout(F.relu(self.fc10_c(x))).view(-1,1,20) # 6
        x11_p = self.dropout(F.relu(self.fc11_p(x))).view(-1,1,20) # 20
        x11_c = self.dropout(F.relu(self.fc11_c(x))).view(-1,1,20) # 6
        x12_p = self.dropout(F.relu(self.fc12_p(x))).view(-1,1,20) # 20
        x12_c = self.dropout(F.relu(self.fc12_c(x))).view(-1,1,20) # 6
        x13_p = self.dropout(F.relu(self.fc13_p(x))).view(-1,1,20) # 20
        x13_c = self.dropout(F.relu(self.fc13_c(x))).view(-1,1,20) # 6
        x14_p = self.dropout(F.relu(self.fc14_p(x))).view(-1,1,20) # 20
        x14_c = self.dropout(F.relu(self.fc14_c(x))).view(-1,1,20) # 6
        x15_p = self.dropout(F.relu(self.fc15_p(x))).view(-1,1,20) # 20
        x15_c = self.dropout(F.relu(self.fc15_c(x))).view(-1,1,20) # 6
        x16_p = self.dropout(F.relu(self.fc16_p(x))).view(-1,1,20) # 20
        x16_c = self.dropout(F.relu(self.fc16_c(x))).view(-1,1,20) # 6
        x17_p = self.dropout(F.relu(self.fc17_p(x))).view(-1,1,20) # 20
        x17_c = self.dropout(F.relu(self.fc17_c(x))).view(-1,1,20) # 6
        x18_p = self.dropout(F.relu(self.fc18_p(x))).view(-1,1,20) # 20
        x18_c = self.dropout(F.relu(self.fc18_c(x))).view(-1,1,20) # 6
        x19_p = self.dropout(F.relu(self.fc19_p(x))).view(-1,1,20) # 20
        x19_c = self.dropout(F.relu(self.fc19_c(x))).view(-1,1,20) # 6
        x20_p = self.dropout(F.relu(self.fc20_p(x))).view(-1,1,20) # 20
        x20_c = self.dropout(F.relu(self.fc20_c(x))).view(-1,1,20) # 6

        return x1_p,x1_c,x2_p,x2_c,x3_p,x3_c,x4_p,x4_c,x5_p,x5_c,x6_p,x6_c,x7_p,x7_c,x8_p,x8_c,x9_p,x9_c,x10_p,x10_c,\
               x11_p,x11_c,x12_p,x12_c,x13_p,x13_c,x14_p,x14_c,x15_p,x15_c,x16_p,x16_c,x17_p,x17_c,x18_p,x18_c,x19_p,x19_c,x20_p,x20_c
        # x_c:u_c 用户信道分配
        # x_p:u_p 用户功率分配



