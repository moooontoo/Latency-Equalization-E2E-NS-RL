# coding=utf-8
import shelve
from utils import *
import numpy as np
import os
import time as t

from config import *

class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.width = config.width
        self.heigth = config.heigth
        self.length = config.length
        self.batch_size = config.batch_size  # 32
        # lin1307
        # 这是一个有限度的存储
        if not (self.is_empty()):  # is_empty()是空的，返回true
            self.actions, self.screens, self.rewards, self.count, self.current = self.load()
        else:
            self.actions = np.empty((self.memory_size, config.action_rows, config.action_cols),
                                    dtype=np.float32)  # .shape[8000, 3, 4]
            self.rewards = np.empty((self.memory_size, config.reward_size),
                                    dtype=np.float32)  # .shape[8000, 1]
            self.screens = np.empty((self.memory_size, self.heigth, self.width),
                                    dtype=np.float32)  # [8000, 1, 4]
        # print(self.screens.shape)
        # screens.shape = (8000, 1, 50, 1)
        # kkw
        self.count = 0  # 这个count用来跟current比较，判断是否存储器存储满了
        self.current = 0  # 在存储器中的顺序，有点像指针，只能是1—8000

        self.current_states = np.empty((self.batch_size, self.heigth, self.width),
                                       dtype=np.float64)  # (32,1,4)
        self.next_states = np.empty((self.batch_size, self.heigth, self.width),
                                    dtype=np.float64)  # (32,1,4)

    def add(self, screen, reward, action):  # self.memory.add(s_t, reward, power)
        self.actions[self.current, ...] = action
        # print(action)
        # kkw
        self.screens[self.current, ...] = screen  # s_t
        self.rewards[self.current, ...] = reward
        self.count = max(self.count, self.current + 1)
        # print(self.count)
        self.current = (self.current + 1) % self.memory_size
        # 保存的都是actions,screens,rewards

    def sample(self):  # 从存储器里随机抽取 self.current_states, actions, reward, self.next_states
        indexes = []
        # print(len(indexes))
        # jjlin
        while len(indexes) < self.batch_size:
            # print(self.count)
            # jjlin
            if self.count > self.current + 1:  # 说明memory_size已经满了，current重新计数了
                index = np.random.randint(1, self.memory_size)  # 在整个满的memory_size随机抽取
            else:
                index = np.random.randint(1, self.current)  # 在已存储的记忆里（记忆没满）随机抽取

            self.current_states[len(indexes), ...] = self.screens[index - 1, ...]
            self.next_states[len(indexes), ...] = self.screens[index, ...]
            indexes.append(index)
        # 以上的意思是随机抽取batch_size个记忆，并赋给current_states和next_states

        actions = self.actions[indexes, :]
        reward = self.rewards[indexes]
        return self.current_states, actions, reward, self.next_states

    # 将数据保存到硬盘
    def save(self):
        print('\n [*]Data saving...')
        tim = str(int(t.time()))
        # kkw
        filenames = os.listdir(Tools.memory_data)
        if len(filenames) != 0:
            for data in filenames:
                os.remove(Tools.memory_data + '/' + data)
        try:
            datas = shelve.open(Tools.memory_data + '/' + tim + '_.db', writeback=True, flag='c')
            # kkw
            datas['actions'] = self.actions
            datas['screens'] = self.screens
            datas['rewards'] = self.rewards
            datas['count'] = self.count
            datas['current'] = self.current
            print('[>^<]Data save SUCCESS')
        except KeyError:
            print('[!]Data save FAILED!!!')
        finally:
            datas.close()
            # kkw

    def is_empty(self):
        # jjlin
        filenames1 = os.listdir(Tools.memory_data)  # Tools.memory_data为地址
        # kkw
        # jjlin
        # print(len(filenames1))
        # print(filenames1)
        # kkw
        if len(filenames1) == 0:
            print('\n[!]There no data!!!')
            # kkw
            return True
        # kkw
        print('\n[>^<]There have data')
        return False

    def load(self):  # 从已存储的数据中恢复出来
        try:
            datas = os.listdir(Tools.memory_data)  # 返回指定文件列表
            tim = []

            for i in range(len(datas)):  # i代表文件的个数
                tim.append(int(datas[i].split('_')[0]))
            '''加载最新的数据'''
            datas = shelve.open(Tools.memory_data + '/' + str(np.max(tim)) + '_.db', writeback=True,
                                flag='c')  # writeback=true可以随时修改存储
            actions = datas['actions']
            screens = datas['screens']
            rewards = datas['rewards']
            count = datas['count']
            current = datas['current']
            print('\n[>^<]Data load success')
            return actions, screens, rewards, count, current
        except KeyError:
            print('\n[!]Data load FAILED!!!')
        finally:
            datas.close()

