import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
num = 30

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num, 60)
        self.fc1_p = nn.Linear(60, num)
        self.fc1_c = nn.Linear(60, num)
        self.fc2_p = nn.Linear(60, num)
        self.fc2_c = nn.Linear(60, num)
        self.fc3_p = nn.Linear(60, num)
        self.fc3_c = nn.Linear(60, num)
        self.fc4_p = nn.Linear(60, num)
        self.fc4_c = nn.Linear(60, num)
        self.fc5_p = nn.Linear(60, num)
        self.fc5_c = nn.Linear(60, num)
        self.fc6_p = nn.Linear(60, num)
        self.fc6_c = nn.Linear(60, num)
        self.fc7_p = nn.Linear(60, num)
        self.fc7_c = nn.Linear(60, num)
        self.fc8_p = nn.Linear(60, num)
        self.fc8_c = nn.Linear(60, num)
        self.fc9_p = nn.Linear(60, num)
        self.fc9_c = nn.Linear(60, num)
        self.fc10_p = nn.Linear(60, num)
        self.fc10_c = nn.Linear(60, num)
        self.fc11_p = nn.Linear(60, num)
        self.fc11_c = nn.Linear(60, num)
        self.fc12_p = nn.Linear(60, num)
        self.fc12_c = nn.Linear(60, num)
        self.fc13_p = nn.Linear(60, num)
        self.fc13_c = nn.Linear(60, num)
        self.fc14_p = nn.Linear(60, num)
        self.fc14_c = nn.Linear(60, num)
        self.fc15_p = nn.Linear(60, num)
        self.fc15_c = nn.Linear(60, num)
        self.fc16_p = nn.Linear(60, num)
        self.fc16_c = nn.Linear(60, num)
        self.fc17_p = nn.Linear(60, num)
        self.fc17_c = nn.Linear(60, num)
        self.fc18_p = nn.Linear(60, num)
        self.fc18_c = nn.Linear(60, num)
        self.fc19_p = nn.Linear(60, num)
        self.fc19_c = nn.Linear(60, num)
        self.fc20_p = nn.Linear(60, num)
        self.fc20_c = nn.Linear(60, num)
        self.fc21_p = nn.Linear(60, num)
        self.fc21_c = nn.Linear(60, num)
        self.fc22_p = nn.Linear(60, num)
        self.fc22_c = nn.Linear(60, num)
        self.fc23_p = nn.Linear(60, num)
        self.fc23_c = nn.Linear(60, num)
        self.fc24_p = nn.Linear(60, num)
        self.fc24_c = nn.Linear(60, num)
        self.fc25_p = nn.Linear(60, num)
        self.fc25_c = nn.Linear(60, num)
        self.fc26_p = nn.Linear(60, num)
        self.fc26_c = nn.Linear(60, num)
        self.fc27_p = nn.Linear(60, num)
        self.fc27_c = nn.Linear(60, num)
        self.fc28_p = nn.Linear(60, num)
        self.fc28_c = nn.Linear(60, num)
        self.fc29_p = nn.Linear(60, num)
        self.fc29_c = nn.Linear(60, num)
        self.fc30_p = nn.Linear(60, num)
        self.fc30_c = nn.Linear(60, num)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 1, num) # 1*1*3
        x = F.relu(self.fc1(x)) # 1*1*180

        x1_p = self.dropout(F.relu(self.fc1_p(x))).view(-1,1,num) # 20
        x1_c = self.dropout(F.relu(self.fc1_c(x))).view(-1,1,num) # 6
        x2_p = self.dropout(F.relu(self.fc2_p(x))).view(-1,1,num) # 20
        x2_c = self.dropout(F.relu(self.fc2_c(x))).view(-1,1,num) # 6
        x3_p = self.dropout(F.relu(self.fc3_p(x))).view(-1,1,num) # 20
        x3_c = self.dropout(F.relu(self.fc3_c(x))).view(-1,1,num) # 6
        x4_p = self.dropout(F.relu(self.fc4_p(x))).view(-1,1,num) # 20
        x4_c = self.dropout(F.relu(self.fc4_c(x))).view(-1,1,num) # 6
        x5_p = self.dropout(F.relu(self.fc5_p(x))).view(-1,1,num) # 20
        x5_c = self.dropout(F.relu(self.fc5_c(x))).view(-1,1,num) # 6
        x6_p = self.dropout(F.relu(self.fc6_p(x))).view(-1,1,num) # 20
        x6_c = self.dropout(F.relu(self.fc6_c(x))).view(-1,1,num) # 6
        x7_p = self.dropout(F.relu(self.fc7_p(x))).view(-1,1,num) # 20
        x7_c = self.dropout(F.relu(self.fc7_c(x))).view(-1,1,num) # 6
        x8_p = self.dropout(F.relu(self.fc8_p(x))).view(-1,1,num) # 20
        x8_c = self.dropout(F.relu(self.fc8_c(x))).view(-1,1,num) # 6
        x9_p = self.dropout(F.relu(self.fc9_p(x))).view(-1,1,num) # 20
        x9_c = self.dropout(F.relu(self.fc9_c(x))).view(-1,1,num) # 6
        x10_p = self.dropout(F.relu(self.fc10_p(x))).view(-1,1,num) # 20
        x10_c = self.dropout(F.relu(self.fc10_c(x))).view(-1,1,num) # 6
        x11_p = self.dropout(F.relu(self.fc11_p(x))).view(-1,1,num) # 20
        x11_c = self.dropout(F.relu(self.fc11_c(x))).view(-1,1,num) # 6
        x12_p = self.dropout(F.relu(self.fc12_p(x))).view(-1,1,num) # 20
        x12_c = self.dropout(F.relu(self.fc12_c(x))).view(-1,1,num) # 6
        x13_p = self.dropout(F.relu(self.fc13_p(x))).view(-1,1,num) # 20
        x13_c = self.dropout(F.relu(self.fc13_c(x))).view(-1,1,num) # 6
        x14_p = self.dropout(F.relu(self.fc14_p(x))).view(-1,1,num) # 20
        x14_c = self.dropout(F.relu(self.fc14_c(x))).view(-1,1,num) # 6
        x15_p = self.dropout(F.relu(self.fc15_p(x))).view(-1,1,num) # 20
        x15_c = self.dropout(F.relu(self.fc15_c(x))).view(-1,1,num) # 6
        x16_p = self.dropout(F.relu(self.fc16_p(x))).view(-1,1,num) # 20
        x16_c = self.dropout(F.relu(self.fc16_c(x))).view(-1,1,num) # 6
        x17_p = self.dropout(F.relu(self.fc17_p(x))).view(-1,1,num) # 20
        x17_c = self.dropout(F.relu(self.fc17_c(x))).view(-1,1,num) # 6
        x18_p = self.dropout(F.relu(self.fc18_p(x))).view(-1,1,num) # 20
        x18_c = self.dropout(F.relu(self.fc18_c(x))).view(-1,1,num) # 6
        x19_p = self.dropout(F.relu(self.fc19_p(x))).view(-1,1,num) # 20
        x19_c = self.dropout(F.relu(self.fc19_c(x))).view(-1,1,num) # 6
        x20_p = self.dropout(F.relu(self.fc20_p(x))).view(-1,1,num) # 20
        x20_c = self.dropout(F.relu(self.fc20_c(x))).view(-1,1,num) # 6
        x21_p = self.dropout(F.relu(self.fc21_p(x))).view(-1, 1, num)  # 20
        x21_c = self.dropout(F.relu(self.fc21_c(x))).view(-1, 1, num)  # 6
        x22_p = self.dropout(F.relu(self.fc22_p(x))).view(-1, 1, num)  # 20
        x22_c = self.dropout(F.relu(self.fc22_c(x))).view(-1, 1, num)  # 6
        x23_p = self.dropout(F.relu(self.fc23_p(x))).view(-1, 1, num)  # 20
        x23_c = self.dropout(F.relu(self.fc23_c(x))).view(-1, 1, num)  # 6
        x24_p = self.dropout(F.relu(self.fc24_p(x))).view(-1, 1, num)  # 20
        x24_c = self.dropout(F.relu(self.fc24_c(x))).view(-1, 1, num)  # 6
        x25_p = self.dropout(F.relu(self.fc25_p(x))).view(-1, 1, num)  # 20
        x25_c = self.dropout(F.relu(self.fc25_c(x))).view(-1, 1, num)  # 6
        x26_p = self.dropout(F.relu(self.fc26_p(x))).view(-1, 1, num)  # 20
        x26_c = self.dropout(F.relu(self.fc26_c(x))).view(-1, 1, num)  # 6
        x27_p = self.dropout(F.relu(self.fc27_p(x))).view(-1, 1, num)  # 20
        x27_c = self.dropout(F.relu(self.fc27_c(x))).view(-1, 1, num)  # 6
        x28_p = self.dropout(F.relu(self.fc28_p(x))).view(-1, 1, num)  # 20
        x28_c = self.dropout(F.relu(self.fc28_c(x))).view(-1, 1, num)  # 6
        x29_p = self.dropout(F.relu(self.fc29_p(x))).view(-1, 1, num)  # 20
        x29_c = self.dropout(F.relu(self.fc29_c(x))).view(-1, 1, num)  # 6
        x30_p = self.dropout(F.relu(self.fc30_p(x))).view(-1, 1, num)  # 20
        x30_c = self.dropout(F.relu(self.fc30_c(x))).view(-1, 1, num)  # 6

        return x1_p,x1_c,x2_p,x2_c,x3_p,x3_c,x4_p,x4_c,x5_p,x5_c,x6_p,x6_c,x7_p,x7_c,x8_p,x8_c,x9_p,x9_c,x10_p,x10_c,\
               x11_p,x11_c,x12_p,x12_c,x13_p,x13_c,x14_p,x14_c,x15_p,x15_c,x16_p,x16_c,x17_p,x17_c,x18_p,x18_c,x19_p,x19_c,x20_p,x20_c,\
               x21_p, x21_c, x22_p, x22_c, x23_p, x23_c, x24_p, x24_c, x25_p, x25_c, x26_p, x26_c, x27_p, x27_c, x28_p, x28_c, x29_p, x29_c,\
    x30_p, x30_c
        # x_c:su_c 次用户信道分配
        # x_p:su_p 次用户功率分配



