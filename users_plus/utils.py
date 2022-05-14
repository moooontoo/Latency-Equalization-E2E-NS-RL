# coding=utf-8
import os
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import inspect
import pprint


pp = pprint.PrettyPrinter().pprint


class Tools:
    hetnet_data = 'HetNet_data'
    checkpoints = hetnet_data + '/checkpoints'
    summary = hetnet_data + '/summary'
    train_data = hetnet_data + '/train_data'
    image = hetnet_data + '/image'
    memory_data = hetnet_data + '/memory_data'
    processed_data = hetnet_data + '/processed_data'
    test_data = hetnet_data + '/test_data'

    @classmethod
    def create_dirs(cls):
        if not (os.path.exists(cls.hetnet_data)):
            os.mkdir(cls.hetnet_data)
            # jjlin
        if not (os.path.exists(cls.checkpoints)):
            os.mkdir(cls.checkpoints)
        if not (os.path.exists(cls.summary)):
            os.mkdir(cls.summary)
        if not (os.path.exists(cls.train_data)):
            os.mkdir(cls.train_data)
        if not (os.path.exists(cls.image)):
            os.mkdir(cls.image)
        if not (os.path.exists(cls.memory_data)):
            os.mkdir(cls.memory_data)
        if not (os.path.exists(cls.processed_data)):
            os.mkdir(cls.processed_data)
        if not (os.path.exists(cls.test_data)):
            os.mkdir(cls.test_data)

def get_epoch_data(net): # network #epoch_data include gain and adjacent_matrix
    gain = net.gain #gain.shape=(400,9)
    gain = np.reshape(gain, [60, 60])
    gain_mean = np.mean(gain)
    adjacent_matrix = net.adjacent_matrix
    adjacent_matrix = np.reshape(adjacent_matrix, [60, 60]) * gain_mean #adjacent不是0和1而是具体的数值
    epoch_data = np.zeros((1, 2, 60, 60))
    epoch_data[0, 0] = gain
    epoch_data[0, 1] = adjacent_matrix
    epoch_data = np.transpose(epoch_data, (0, 2, 3, 1)) #(1,60,60,2)这是一个state的值
    return epoch_data


def is_gpu_available():
    is_gpu = tf.test.is_gpu_available(True)
    return is_gpu


def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    gpu_name = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpu_name)  # 返回gpu的个数


def set_gpu():
    if is_gpu_available():
        gpus = get_available_gpus()
        print('There are %d GPUS'%gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(['{i}'.format(i=a) for a in range(gpus)]) # 对所有gpu可见
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 设置GPU使用量，但是需求比较大时会自动多分配
        # config.gpu_options.allow_growth=True # 不全部占满显存，按需分配
        session = tf.InteractiveSession(config=config)
        return session
    else:
        print('There are no gpu! ')
        return tf.InteractiveSession()


def class_vars(obj): #m,obj config
    return {k: v for k, v in inspect.getmembers(obj) #--return all the members in a list of object
            if not k.startswith('__') and not callable(k)} #String.StartsWith 用以检测字符串是否以指定的前缀开始


def str_map_float(str_array):
    nums = []
    for strs in str_array:
        nums.append(float(strs))
    return nums


def save_data(filepath, data):
    np.savetxt(filepath, data, delimiter=',')

# def load_data(filepath):
#     np.loadtxt(filepath, delimiter=',')


# def get_random_action(net):
#     number = []
#     mbs_number = np.random.randint(20, 40, size=1)
#     pbs_number = np.random.randint(10, 20, size=4)
#     fbs_number = np.random.randint(5, 10, size=4)
#     number += (list(mbs_number))
#     number += (list(pbs_number))
#     number += (list(fbs_number))
#     number = np.floor(number / np.sum(number) * net.UE_num)
#     if np.sum(number) == net.UE_num:
#         return number
#     number[0] += int(net.UE_num - np.sum(number))
#     assert np.sum(number) == net.UE_num
#     number = [int(num) for num in number]
#     return np.array(number)


# def number2adjacent(net, number):
#     if np.sum(number) != net.UE_num:
#         raise Exception('用户没有完全覆盖或者用户数目不对')
#     adjacent_matrix = np.zeros_like(net.BS_UE_distance)
#     index = np.argsort(net.BS_UE_distance, axis=0)  # 考虑距离进行分配是否妥当？
#     '''先考虑FBS的选择，后考虑PBS，再考虑MBS， 后期可以考虑添加约束条件'''
#     bs_user = []
#     for bs in range(net.expectBS-1, -1, -1):
#         num = 0
#         while np.sum(adjacent_matrix[:, bs]) < number[bs]:
#             bu_temp = index[:number[bs] + num, bs]
#             adjacent_matrix[list(set(bu_temp).difference(set(bs_user))), bs] = 1
#             num += 1
#         bs_user = np.concatenate((bs_user, bu_temp))
#     return adjacent_matrix

# Tools().create_dirs()