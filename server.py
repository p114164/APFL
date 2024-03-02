import warnings  # 屏蔽warning
import os
warnings.filterwarnings("ignore", message="WARNING: The input does not fit in a single ciphertext*")
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, cifar10_cnn
from clients_ckks import ClientsGroup, client
from model.WideResNet import WideResNet
import time
from copy import deepcopy
# import copy  # 往列表里append
from numpy import *
# 记录csv列表和画图
import random
import pandas as pd
from datetime import datetime
import tenseal as ts

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=4, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
# batchsize大小，默认为10
parser.add_argument('-B', '--batchsize', type=int, default=16, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='cifar10_cnn', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.0002, help="learning rate, \
                    use value from origin paper as default")
# 数据集
parser.add_argument('-dataset', "--dataset", type=str, default="cifar10", help="需要训练的数据集")
# parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=100, help='global model save frequency(of communication)')
# num_comm 表示通信次数(训练轮次)
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')

parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
parser.add_argument('-Algorithm', '--algorithm', type=str, default='APFL', help='FLAvg or APFL')
parser.add_argument('-poisoning ratio', '--pr', type=float, default=0, help='poisoning ratio')
parser.add_argument('-poisoning attack type', '--attack', type=str, default="NA", help='label flipping attack:LFA'
                                                                                       'model poisoning attacks:MPA'
                                                                                       'no attack:NA')


def tes_mkdir(path):  # 将该程序页的test_mkdir都改成tes_mkdir
    if not os.path.isdir(path):
        os.mkdir(path)


def l2(parameters):  # 求l2范数的函数
    l2 = 0
    for var in parameters:
        l2 += torch.norm(parameters[var], p=2)
    return l2


def parameters_cosine(parameters1, parameters2):  # 定义求余弦相似度函数
    cosine = []
    for var in parameters2:
        cos = torch.mean(torch.cosine_similarity(parameters1[var], parameters2[var], dim=-1))
        cos = cos.cpu().numpy()
        cosine.append(deepcopy(cos))
        cos1 = sum(cosine)
    return cos1


def enc(parameters, context):  # 对模型参数加密的函数
    for var in parameters:
        parameters[var] = parameters[var].flatten().to('cpu')
        parameters[var] = ts.ckks_vector(context, parameters[var])
    return parameters


def dec(parameters, secret_key):  # 解密模型参数
    parameters1 = dict.fromkeys(parameters, 0)  # 重新定一个字典存储模型参数，不要改变原parameters的值，不然后面的客户端的parameters是解密之后的值
    for var in parameters:
        parameters1[var] = parameters[var].decrypt(secret_key)
    return parameters1


def cosine_medain_sum(list):  # 定义求余弦相似度函数
    cosine_sum = 0
    for i in range(len(list)):
        if list[i] >= np.median(list):
            cosine_sum += list[i]
    return cosine_sum


def w_sum(list):
    list = np.array(list)
    suoyin = np.argsort(-cosine2)
    list2 = np.sort(list)
    return suoyin, list2


def MPA(parameter):
    new_parameter = dict.fromkeys(parameter, 0)  # 加权聚合后的全局模型
    for var in parameter:
        new_parameter[var] = parameter[var].reshape(-1).to('cpu')
        new_parameter[var] = np.array(new_parameter[var])
        np.random.shuffle(new_parameter[var])
        new_parameter[var] = new_parameter[var].reshape(parameter[var].shape)
        new_parameter[var] = torch.from_numpy(new_parameter[var])
        new_parameter[var] = new_parameter[var].to('cuda')
    return new_parameter

# def parameters_cosine(parameters1, parameters2):  # 定义求余弦相似度函数
#     cosine = []
#     for var in parameters2:
#         cos = torch.cosine_similarity(parameters1[var].reshape(1, -1), parameters2[var].reshape(1, -1))
#         # cos = torch.cosine_similarity(parameters1[var], parameters2[var], dim=-1)
#         cos = cos.cpu().numpy()
#         cosine.append(deepcopy(cos))
#     return cosine


# ckks设置
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = pow(2, 40)
secret_key = context.secret_key()

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__
    # -----------------------文件保存-----------------------#
    # 创建结果文件夹
    # tes_mkdir("./result")
    # path = os.getcwd()
    # 结果存放test_accuracy中
    # test_txt = open("test_accuracy.txt", mode="a")
    # loss_txt = open("loss.txt", mode="a")
    # global_parameters_txt = open("global_parameters.txt",mode="a",encoding="utf-8")
    # ----------------------------------------------------#
    tes_mkdir(args['save_path'])
    # 记录程序运行时间
    print(time.strftime("开始运行时间:" + "时间:%Y-%m-%d-%H:%M:%S", time.localtime()))  # 输出当前时间
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = None
    # 初始化模型
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet网络
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)
    elif args['model_name'] == 'cifar10_cnn':
        net = cifar10_cnn()
    # 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)

    # 定义损失函数
    loss_func = F.cross_entropy
    if args['dataset'] == 'cifar10':
        opti = optim.Adam(net.parameters(), lr=args['learning_rate'])
        print("优化算法：", opti)
    elif args['dataset'] == 'mnist':
        opti = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9)
        print("优化算法：", opti)
    # 创建Clients群
    myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 选择客户端比例
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
    # 全局模型参数
    global_parameters = {}
    # net.state_dict()  # 获取模型参数以共享
    # 得到每一层中全连接层中的名称fc1.weight
    # 以及权重weights(tenor)
    # 得到网络每一层上
    for key, var in net.state_dict().items():
        # print("key:"+str(key)+",var:"+str(var))
        print("张量的维度:" + str(var.shape))
        print("张量的Size:" + str(var.size()))
        global_parameters[key] = var.clone()
    for var in global_parameters:
        print("全局模型参数类型", global_parameters[var].shape, global_parameters[var].type())

    for var in global_parameters:  # 加密第一次迭代的全局模型参数
        global_parameters[var] = global_parameters[var].to('cpu')
        global_parameters[var] = global_parameters[var].flatten()
        global_parameters[var] = ts.ckks_vector(context, global_parameters[var])
    # num_comm 表示通信次数
    accuracy_list = []  # 记录精度
    loss_list = []  # 记录损失值
    for i in range(args['num_comm']):
        global_parameters_Gaussian = dict.fromkeys(net.state_dict(), 0)  # 定义加噪后的全局模型参数
        print("communicate round {}".format(i + 1))
        # 选取全部的用户进行训练
        if args['algorithm'] == 'FLAvg':
            order = random.sample(range(0, int(args['num_of_clients'])), args['num_of_clients'])
        elif args['algorithm'] == 'APFL':
            order = random.sample(range(0, int(args['num_of_clients'])), args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order]
        print("随机挑选的客户端:", clients_in_comm)
        # print("clients_in_comm的type:", type(clients_in_comm))  # <class 'list'>
        sum_parameters = None
        localParameters = []  # 定义一个新的列表用以存放客户端的局部模型参数
        localParameters_Gaussian = []
        # 每个Client基于当前模型参数和自己的数据训练并更新模型

        for client in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数，这一行代码表示Client端的训练函数，我们详细展开：
            # local_parameters 得到客户端的局部变量
            local_parameters, local_parameters_Gaussian = myClients.clients_set[client].localUpdate(args['epoch'],
                                                                                                    args['batchsize'],
                                                                                                    net,
                                                                                                    loss_func, opti,
                                                                                                    global_parameters,
                                                                                                    context, secret_key)
            # test_img(client, local_parameters_Gaussian, testDataLoader)
            # 如果是MPA投毒用户，则参数置反
            if clients_in_comm.index(client) < 10 and args['algorithm'] == 'FLAvg' and args['attack'] == 'MPA':
                for key in local_parameters:
                    local_parameters[key] = -local_parameters[key]
            elif clients_in_comm.index(client) < 10 and args['algorithm'] == 'APFL' and args['attack'] == 'MPA':
                for key in local_parameters:
                    local_parameters[key] = -local_parameters[key]
                for key in local_parameters_Gaussian:
                    local_parameters_Gaussian[key] = -local_parameters_Gaussian[key]
            # 用localParameters(list类型)记录各个客户端的本地参数
            localParameters.append(
                deepcopy(local_parameters))  # 往列表里append
            # 用localParameters_Gaussian(list类型)记录各个客户端的加噪本地参数
            localParameters_Gaussian.append(
                deepcopy(local_parameters_Gaussian))  # 加噪的本地模型参数

        if args['algorithm'] == 'FLAvg':
            # 取平均值，得到本次通信中Server得到的更新后的模型参数
            for var in global_parameters:
                global_parameters[var] = (sum_parameters[var] / num_in_comm)
            print("全局模型范数", l2(global_parameters))
        elif args['algorithm'] == 'APFL':
            # 求加入高斯噪声后的平均模型参数：
            for i in range(len(localParameters_Gaussian)):  # 求当前轮次加噪后的本地模型参数之和
                for var in localParameters_Gaussian[i]:
                    global_parameters_Gaussian[var] += localParameters_Gaussian[i][var]
            sum_cos = 0  # 记录余弦相似度之和
            cosine2 = [0] * num_in_comm  # 初始化每个用户的余弦相似度的list
            for i1 in range(len(localParameters_Gaussian)):
                clientcos = 0
                locals()["client{}_cos".format(clients_in_comm[i1])] = []  # 记录局部模型的余弦相似度
                for i in range(num_in_comm):
                    cos = parameters_cosine(localParameters_Gaussian[i1], localParameters_Gaussian[i])
                    locals()["client{}_cos".format(clients_in_comm[i1])].append(deepcopy(cos))
                # 取大于等于中位数的余弦值当做聚合余弦值
                locals()["client{}_cos".format(clients_in_comm[i1])] = np.array(
                    locals()["client{}_cos".format(clients_in_comm[i1])])
                cosine2[i1] = cosine_medain_sum(locals()["client{}_cos".format(clients_in_comm[i1])])
                locals()["{}_cos".format(clients_in_comm[i1])] = cosine2[i1]
            cosine2 = np.array(cosine2)
            # print("索引", np.argsort(-cosine2))  # [5 3 4 8 7 2 9 0 1 6]
            index = np.argsort(-cosine2)  # 相关性从大到小排序的索引
            for i in range(num_in_comm):
                if i < int(num_in_comm / 2):  # 参与训练的一半的客户端余弦相似度之和
                    locals()["{}_cos".format(clients_in_comm[index[i]])] = cosine2[index[i]]
                    # print("{}相似度".format(clients_in_comm[index[i]]), locals()["{}_cos".format(clients_in_comm[index[i]])])
                    sum_cos += cosine2[index[i]]
            # 给每个客户端赋予权重
            for i in range(num_in_comm):
                if i < int(num_in_comm / 2):  # 参与训练的一半的客户端的聚合权重
                    locals()["{}_cos".format(clients_in_comm[index[i]])] = locals()["{}_cos".format(
                        clients_in_comm[index[i]])] / sum_cos  # 10个权重初始值
                    print("{}权重:".format(clients_in_comm[index[i]]),
                          locals()["{}_cos".format(clients_in_comm[index[i]])])
            # 根据权重聚合局部模型
            global_parameters = dict.fromkeys(localParameters_Gaussian[0], 0)
            for i in range(num_in_comm):
                if i < int(num_in_comm / 2):
                    for var in global_parameters:
                        global_parameters[var] += localParameters[index[i]][var] * locals()[
                            "{}_cos".format(clients_in_comm[index[i]])]

        #  解密全局模型并测试模型精度
        global_parameters_dec = dec(global_parameters, secret_key)
        for var in global_parameters_dec:
            global_parameters_dec[var] = np.array(global_parameters_dec[var]).reshape(net.state_dict()[var].shape)
            global_parameters_dec[var] = torch.tensor(global_parameters_dec[var], dtype=torch.float32,
                                                      device=torch.device('cuda'))  # 在GPU上
        print("全局模型解密完成,开始测试并记录全局模型准确率")

        sum_accu = 0
        num = 0
        net.load_state_dict(global_parameters_dec, strict=True)
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            label = label.long()  # cifar10数据集的时候要添这句
            loss = loss_func(preds, label)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n" + 'loss: {}'.format(loss))
        print("\n" + 'accuracy: {}'.format(sum_accu / num))
        # 记录损失值和精度list
        loss_list.append(deepcopy(loss.detach().cpu().numpy()))
        accuracy_list.append(deepcopy((sum_accu / num).item()))
        # print("loss:", loss.detach().numpy())
        # loss_txt.write('loss:' + str(loss) + "\n")
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
# 建立csv文件存储loss和acc
df = pd.DataFrame(columns=['time', 'round', 'loss', 'accuracy'])  # 列名
df.to_csv("./log/list.csv".format(args['algorithm'], args['dataset'], args['attack'], args['pr']),
          index=False)  # 路径可以根据需要更改
for i in range(args['num_comm']):
    time = "%s" % datetime.now()
    round = "round[%d]" % i
    loss = "%f" % loss_list[i]
    accuracy = "%g" % accuracy_list[i]
    # 将数据保存在一维列表
    list = [time, round, loss, accuracy]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv("./log/list.csv".format(args['algorithm'], args['dataset'], args['attack'], args['pr']),
                mode='a', header=False, index=False)  # mode设为a,就可以
