import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import tenseal as ts
from Models import Mnist_2NN, Mnist_CNN
import torch.nn.functional as F
from torch import optim

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, context, secret_key):
        '''
            param: localEpoch 迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前全局模型参数
            return: 返回本地模型参数
        '''
        # 现将global_parameters解密
        global_parameters_dec = dec(global_parameters, secret_key)
        for var in global_parameters_dec:
            global_parameters_dec[var] = np.array(global_parameters_dec[var]).reshape(Net.state_dict()[var].shape)
            global_parameters_dec[var] = torch.tensor(global_parameters_dec[var], dtype=torch.float32, device=torch.device('cuda'))  # 在GPU上
        print("解密完成,开始本地训练")
        # 加载当前通信中最新全局参数
        # 传入网络模型，并加载global_parameters参数的
        Net.load_state_dict(global_parameters_dec, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        # 设置迭代次数
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                label = label.long()  # 跑cifar10数据集时需要这句
                loss = lossFun(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # loss_txt.write('loss: ' + str(loss) + "\n")

        def l2(parameters):  # 求l2范数的函数
            l2 = 0
            for var in parameters:
                l2 += torch.norm(parameters[var], p=2)
            return l2

        # 对局部模型参数添加高斯噪声
        def addnoice(C, epsilon, delta, local):
            c = np.sqrt(2 * 100 * np.log(1/delta))
            s = 2*C/2380  # cifar10敏感度
            # s = 2*C/2857  # mnist敏感度
            l2 = 0
            for var in local:
                l2 += torch.norm(local[var], p=2)
            for var in local:
                local[var] = local[var] / max(1, l2/C)  # 裁剪，将l2范数限制在C以内
                local[var] = local[var]+torch.normal(0, ((c**2) * (s**2))/(epsilon * epsilon), local[var].shape).to(dev)
            return local
        # Net.state_dict_Gaussian = addnoice(l2(global_parameters), 0.01, 40, Net.state_dict())  # 设置高斯噪声的参数并加噪
        Net.state_dict_Gaussian = addnoice(40, 60, 0.01,  Net.state_dict())  # 设置高斯噪声的参数并加噪
        state_dict= {}
        for var in Net.state_dict():  # 裁剪阈值为40
            Net.state_dict()[var] = Net.state_dict()[var] / max(1, l2(Net.state_dict())/40)
        state_dict = enc(Net.state_dict(), context)
        print("加密完成")
        # Net.state_dict_Gaussian = dict.fromkeys(Net.state_dict(), 0)  # 初始化加噪后的局部模型参数
        return state_dict, Net.state_dict_Gaussian  # 返回本地训练的模型参数Net.state_dict()和高斯加噪版Net.state_dict_Gaussian

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)
        test_data = torch.tensor(mnistDataSet.test_data)
        # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        if self.data_set_name == 'mnist':
            test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        elif self.data_set_name == 'cifar10':
            test_label = torch.tensor(mnistDataSet.test_label)
        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2  # shard_size:每个数据切片的大小
        # print("shard_size:"+str(shard_size))
        print("mnistDataSet.train_data_size // shard_size", mnistDataSet.train_data_size // shard_size)
        # np.random.permutation 将序列进行随机排序
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)  # 随机打乱切片数据

        print("*" * 100)
        print("shards_id:", shards_id)  # 输出随机打乱后的切片id
        print("shards_id.shape:", shards_id.shape)
        print("*" * 100)
        for i in range(self.num_of_clients):  # 对每一个客户端进行操作
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            # 将数据以及的标签分配给该客户端
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            if self.data_set_name == 'mnist':
                local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_label = np.argmax(local_label, axis=1)
            elif self.data_set_name == 'cifar10':
                local_data, local_label = np.vstack((data_shards1, data_shards2)), np.hstack(
                    (label_shards1, label_shards2))
            # # LFA投毒攻击.
            # if i > int(21-(21*0)):
            #     for pr in range(len(local_label)):  # 翻转所有标签
            #         local_label[pr] = 10-local_label[pr]-1
            # 创建一个客户端
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            # 为每一个clients 设置一个名字
            self.clients_set['client{}'.format(i)] = someone

if __name__ == "__main__":
    MyClients = ClientsGroup('cifar10', False, 21, 1)  # def __init__(self, dataSetName, isIID, numOfClients, dev)
    print("client:", client)  # 输出:client: <class '__main__.client'>
    print('用户部分样本', MyClients.clients_set['client10'].train_ds[0:10])
    train_ids = MyClients.clients_set['client10'].train_ds[0:10]
    print("train_ids", train_ids, type(train_ids))  # 元组
    print("train_ds type", type(MyClients.clients_set['client10'].train_ds))  # tensordataset
    print("MyClients.clients_set['client10'].train_ds.label:", MyClients.clients_set['client10'].train_ds[200][1])
    print("MyClients.clients_set['client10'].train_ds.label,type:", type(MyClients.clients_set['client10'].train_ds[200][1]))
    print("MyClients.clients_set['client10'].train_ds.label:", MyClients.clients_set['client10'].train_ds[200][1])
    # 其中train_ds就是训练数据，是tensordataset类型，每个用户里面有600个样本数据，一个样本数据有28*28个像素点，然后1个标签
    i = 0
    for x_train in train_ids[0]:
        print("client10 数据:" + str(i))
        print("x_train:", x_train)
        i = i + 1
    for i in range(21):
        print("查看标签{}".format(i), MyClients.clients_set['client{}'.format(i)].train_ds[0:10][1])
