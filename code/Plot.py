import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
def plot(data, x_number, name,start, end):
       xlabel = np.linspace(start, end, x_number)
       plt.figure(figsize=(10, 5))
       plt.grid(linestyle="--")
       plt.ylabel(name, fontsize=15, fontweight='bold')
       plt.xlim(0, x_number)
       plt.ylim((0, 20))
       plt.plot(xlabel, data)
       plt.savefig(name + '.jpg')


if __name__ == '__main__':
       # qoe = [0.7384999999999999, 0.7430000000000001, 0.7380000000000001, 0.7309999999999998, 0.7315,
       #        0.7099999999999999, 0.7445, 0.7260000000000001,
       #        0.7594999999999998, 0.7069999999999999, 0.7425, 0.7284999999999999, 0.7259999999999998,
       #        0.7234999999999999, 0.7395, 0.7439999999999999,
       #        0.7354999999999999, 0.7414999999999999, 0.7809999999999999, 0.7959999999999998, 0.8105,
       #        0.7655000000000001, 0.7944999999999999,
       #        0.7955, 0.7839999999999998, 0.7955000000000001, 0.7739999999999999, 0.7840000000000003,
       #        0.7864999999999999, 0.7799999999999998,
       #        0.7925, 0.8015000000000001, 0.7939999999999999, 0.8064999999999999, 0.7695000000000002,
       #        0.8135000000000001, 0.8135000000000001,
       #        0.8045, 0.7855000000000001, 0.813]
       # plot(qoe, len(qoe), 'qoe changes during ran rl', 1, len(qoe))

       #csv文件 绘图
       # f = open('ep_rewards.csv')
       # read = csv.reader(f)
       # data = list(read)
       # len = len(data)
       # x, y =[], []
       # for i in range(96, int(len)):
       #        if i%2 == 0:
       #               x.append(data[i])
       #        else:
       #               y.append(data[i])
       # for i in range(48):
       #        a = y[i]
       #        b = a[0]
       #        y[i] = float(b)
       # xlabel = np.linspace(200, 5000, 49)
       # y.insert(0, 0)
       # y = np.array(y)
       # plt.figure(figsize=(10, 5))
       # plt.grid(linestyle="--")
       # plt.xlabel('episodes', fontsize=15, fontweight='bold')
       # plt.ylabel('rewards', fontsize=15, fontweight='bold')
       # plt.plot(xlabel, y)
       # plt.axhline(y = y[3], color='r', linestyle='-')
       # plt.xlim(0, 5000)
       # plt.ylim((0, 1))
       # plt.savefig('ran_rewards.jpg')
       # plt.show()

       s1 = [0.65, 0.65, 0.65, 0.7, 0.75, 0.55, 0.6, 0.7, 0.65, 0.6, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
       xlabel = np.linspace(0,20,20)
       plt.figure(figsize=(10, 5))
       plt.grid(linestyle="--")
       plt.xlabel('slices', fontsize=15, fontweight='bold')
       plt.ylabel('ratio', fontsize=15, fontweight='bold')
       plt.axhline(y=0.5, color='r', linestyle='-')
       plt.ylim((0,1))
       plt.plot(xlabel, s1)
       plt.show()