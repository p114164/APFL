# 可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.pyplot import MultipleLocator
from matplotlib.font_manager import FontProperties
fonten = FontProperties(fname=r"C:\Users\y'y'p\AppData\Local\Microsoft\Windows\Fonts\euclid.ttf", size=8)
fontcn = {'family': 'SimSun', 'size': 8}#设置宋体，字号
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
str = 'cifar10_NA'
if str == 'mnist_NA':
    Baseline = pd.read_csv("log/mnist/FLAvg_acc_mnist_NA.csv")
    FLDH_data = pd.read_csv("log/mnist/FLDH_acc_mnist_NA.csv")
    krum_data = pd.read_csv("log/mnist/krum_acc_mnist_NA.csv")
    Baseline_acc= Baseline[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    FLAvg_acc_NA = np.array(Baseline_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax.plot(round, Baseline_acc, label='Baseline')
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    axins = ax.inset_axes((0.75, 0.78, 0.2, 0.125))
    axins.plot(round[90:102:1], Baseline_acc[90:102:1], color='dodgerblue')
    axins.plot(round[90:102:1], FLDH_acc[90:102:1], color='red')
    axins.plot(round[90:102:1], krum_acc[90:102:1], color='green')
    axins.set_ylim((0.96, 0.986))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')  # 插入对齐标线
    x_major_locator=MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator=MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()   # 显示标签
    plt.show()
if str == 'cifar10_NA':
    Baseline = pd.read_csv("log/FLAvg_acc_cifar10_NA.csv")
    FLAvg_data = pd.read_csv("log/cifar10/krum_acc_cifar10_NA.csv")
    FLDH_data = pd.read_csv("log/cifar10/FLDH_acc_cifar10_NA.csv")
    krum_data = pd.read_csv("log/cifar10/krum_acc_cifar10_NA.csv")
    Baseline_acc = Baseline[['accuracy']]
    FLAvg_acc = FLAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    FLAvg_acc_NA = np.array(Baseline_acc)[0:100:1]
    FLAvg_acc = np.array(FLAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline', lw=0.5)
    # ax.plot(round, FLAvg_acc, label="FLAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours", lw=0.5)
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$", lw=0.5)
    # 刻度线设置
    my_x_ticks = np.arange(0, 120, 20)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0.1, 0.8, 0.1)
    plt.yticks(my_y_ticks)
    # 子图位置和数据
    axins = ax.inset_axes((0.6, 0.6, 0.2, 0.125))
    axins.plot(round[90:103:1], Baseline_acc[90:103:1], color='dodgerblue', lw=0.5)
    # axins.plot(round[90:103:1], FLAvg_acc[90:103:1], color='orange', lw=0.5)
    axins.plot(round[90:103:1], FLDH_acc[90:103:1], color='red', lw=0.5)
    # axins.plot(round[90:103:1], krum_acc[90:103:1], color='green', lw=0.5)
    # 设置子图坐标轴范围和格式
    axins.set_xlim((90, 102))
    axins.set_ylim((0.705, 0.755))
    for tickx in axins.xaxis.get_major_ticks():
        tickx.label.set_fontproperties(fonten)
    for ticky in axins.yaxis.get_major_ticks():
        ticky.label.set_fontproperties(fonten)
    # 插入小图对齐标线
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.5, linestyle='--')
    # # 背景虚线
    # x_major_locator = MultipleLocator(10)
    # ax.xaxis.set_major_locator(x_major_locator)
    # y_major_locator = MultipleLocator(0.1)
    # ax.yaxis.set_major_locator(y_major_locator)
    # ax.grid(linestyle='-.')
    plt.xlabel('迭代次数')
    plt.xlabel(u'迭代次数', fontproperties=fontcn)
    plt.ylabel(u'精度', fontproperties=fontcn)
    plt.xticks(fontproperties=fonten)
    plt.yticks(fontproperties=fonten)
    plt.legend(loc='best', frameon=False, prop=fonten)#  标签无方框,euclid格式
    plt.show()
    fig.savefig('log/images/cifar10/ceshi1.svg', format="svg", dpi=600, bbox_inches='tight', pad_inches=0.02)
if str == 'mnist_LFA_0.3':
    Baseline = pd.read_csv("log/mnist/FLAvg_acc_mnist_NA.csv")
    FedAvg_data = pd.read_csv("log/mnist/LFA/FLAvg_acc_mnist_LFA_0.3.csv")
    FLDH_data = pd.read_csv("log/mnist/LFA/FLDH_acc_mnist_LFA_0.3.csv")
    krum_data = pd.read_csv("log/mnist/LFA/krum_acc_mnist_LFA_0.3.csv")
    Baseline_acc = Baseline[['accuracy']]
    FedAvg_acc = FedAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    Baseline_acc = np.array(Baseline_acc)[0:100:1]
    FedAvg_acc = np.array(FedAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')  # 主图
    ax.plot(round, FedAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    axins = ax.inset_axes((0.65, 0.75, 0.2, 0.125))  # 子图
    axins.plot(round[90:100:1], Baseline_acc[90:100:1], color='dodgerblue')
    axins.plot(round[90:100:1], FLDH_acc[90:100:1], color='red')
    axins.plot(round[90:100:1], krum_acc[90:100:1], color='green')
    axins.set_ylim((0.96, 0.99))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')
    x_major_locator = MultipleLocator(10)  # 背景
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'mnist_LFA_0.5':
    Baseline = pd.read_csv("log/mnist/FLAvg_acc_mnist_NA.csv")
    FedAvg_data = pd.read_csv("log/mnist/LFA/FLAvg_acc_mnist_LFA_0.5.csv")
    FLDH_data = pd.read_csv("log/mnist/LFA/FLDH_acc_mnist_LFA_0.5.csv")
    krum_data = pd.read_csv("log/mnist/LFA/krum_acc_mnist_LFA_0.5.csv")
    Baseline_acc = Baseline[['accuracy']]
    FedAvg_acc = FedAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    Baseline_acc = np.array(Baseline_acc)[0:100:1]
    FedAvg_acc = np.array(FedAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')  # 主图
    ax.plot(round, FedAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    axins = ax.inset_axes((0.65, 0.75, 0.2, 0.125))  # 子图
    axins.plot(round[90:100:1], Baseline_acc[90:100:1], color='dodgerblue')
    axins.plot(round[90:100:1], FLDH_acc[90:100:1], color='red')
    axins.plot(round[90:100:1], krum_acc[90:100:1], color='green')
    axins.set_ylim((0.96, 0.99))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')
    x_major_locator = MultipleLocator(10)  # 背景
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'mnist_MPA_0.3':
    Baseline = pd.read_csv("log/mnist/FLAvg_acc_mnist_NA.csv")
    FedAvg_data = pd.read_csv("log/mnist/MPA/FLAvg_acc_mnist_MPA_0.3.csv")
    FLDH_data = pd.read_csv("log/mnist/MPA/FLDH_acc_mnist_MPA_0.3.csv")
    krum_data = pd.read_csv("log/mnist/MPA/krum_acc_mnist_MPA_0.3.csv")
    Baseline_acc = Baseline[['accuracy']]
    FedAvg_acc = FedAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    Baseline_acc = np.array(Baseline_acc)[0:100:1]
    FedAvg_acc = np.array(FedAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')  # 主图
    ax.plot(round, FedAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    axins = ax.inset_axes((0.65, 0.75, 0.2, 0.125))  # 子图
    axins.plot(round[90:100:1], Baseline_acc[90:100:1], color='dodgerblue')
    axins.plot(round[90:100:1], FLDH_acc[90:100:1], color='red')
    axins.plot(round[90:100:1], krum_acc[90:100:1], color='green')
    axins.set_ylim((0.96, 0.99))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')
    x_major_locator = MultipleLocator(10)  # 背景
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'mnist_MPA_0.5':
    Baseline = pd.read_csv("log/mnist/FLAvg_acc_mnist_NA.csv")
    FedAvg_data = pd.read_csv("log/mnist/MPA/FLAvg_acc_mnist_MPA_0.5.csv")
    FLDH_data = pd.read_csv("log/mnist/MPA/FLDH_acc_mnist_MPA_0.5.csv")
    krum_data = pd.read_csv("log/mnist/MPA/krum_acc_mnist_MPA_0.5.csv")
    Baseline_acc = Baseline[['accuracy']]
    FedAvg_acc = FedAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    Baseline_acc = np.array(Baseline_acc)[0:100:1]
    FedAvg_acc = np.array(FedAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')  # 主图
    ax.plot(round, FedAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    axins = ax.inset_axes((0.65, 0.75, 0.2, 0.125))  # 子图
    axins.plot(round[90:100:1], Baseline_acc[90:100:1], color='dodgerblue')
    axins.plot(round[90:100:1], FLDH_acc[90:100:1], color='red')
    axins.plot(round[90:100:1], krum_acc[90:100:1], color='green')
    axins.set_ylim((0.96, 0.99))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')
    x_major_locator = MultipleLocator(10)  # 背景
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'cifar10_LFA_0.3':
    Baseline = pd.read_csv("log/FLAvg_acc_cifar10_NA.csv")
    FLAvg_data = pd.read_csv("log/cifar10/LFA/FLAvg_acc_cifar10_LFA_0.3.csv")
    FLDH_data = pd.read_csv("log/cifar10/LFA/FLDH_acc_cifar10_LFA_0.3.csv")
    krum_data = pd.read_csv("log/cifar10/LFA/krum_acc_cifar10_LFA_0.3.csv")
    Baseline_acc = Baseline[['accuracy']]
    FLAvg_acc = FLAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    FLAvg_acc_NA = np.array(Baseline_acc)[0:100:1]
    FLAvg_acc = np.array(FLAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')
    ax.plot(round, FLAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'cifar10_LFA_0.5':
    Baseline = pd.read_csv("log/FLAvg_acc_cifar10_NA.csv")
    FLAvg_data = pd.read_csv("log/cifar10/LFA/FLAvg_acc_cifar10_LFA_0.5.csv")
    FLDH_data = pd.read_csv("log/cifar10/LFA/FLDH_acc_cifar10_LFA_0.5.csv")
    krum_data = pd.read_csv("log/cifar10/LFA/krum_acc_cifar10_LFA_0.5.csv")
    Baseline_acc = Baseline[['accuracy']]
    FLAvg_acc = FLAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    FLAvg_acc_NA = np.array(Baseline_acc)[0:100:1]
    FLAvg_acc = np.array(FLAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')
    ax.plot(round, FLAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'cifar10_MPA_0.3':
    Baseline = pd.read_csv("log/FLAvg_acc_cifar10_NA.csv")
    FLAvg_data = pd.read_csv("log/cifar10/MPA/FLAvg_acc_cifar10_MPA_0.3.csv")
    FLDH_data = pd.read_csv("log/cifar10/MPA/FLDH_acc_cifar10_MPA_0.3.csv")
    krum_data = pd.read_csv("log/cifar10/MPA/krum_acc_cifar10_MPA_0.3.csv")
    Baseline_acc = Baseline[['accuracy']]
    FLAvg_acc = FLAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    FLAvg_acc_NA = np.array(Baseline_acc)[0:100:1]
    FLAvg_acc = np.array(FLAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label='Baseline')
    ax.plot(round, FLAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")

    axins = ax.inset_axes((0.53, 0.55, 0.2, 0.125))
    axins.plot(round[90:103:1], Baseline_acc[90:103:1], color='dodgerblue')
    axins.plot(round[90:103:1], FLAvg_acc[90:103:1], color='orange')
    axins.plot(round[90:103:1], FLDH_acc[90:103:1], color='red')
    axins.plot(round[90:103:1], krum_acc[90:103:1], color='green')
    axins.set_ylim((0.705, 0.755))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示标签
    plt.show()
if str == 'cifar10_MPA_0.5':
    Baseline = pd.read_csv("log/FLAvg_acc_cifar10_NA.csv")
    FLAvg_data = pd.read_csv("log/cifar10/MPA/FLAvg_acc_cifar10_MPA_0.5.csv")
    FLDH_data = pd.read_csv("log/cifar10/MPA/FLDH_acc_cifar10_MPA_0.5.csv")
    krum_data = pd.read_csv("log/cifar10/MPA/krum_acc_cifar10_MPA_0.5.csv")
    Baseline_acc = Baseline[['accuracy']]
    FLAvg_acc = FLAvg_data[['accuracy']]
    FLDH_acc = FLDH_data[['accuracy']]
    krum_acc = krum_data[['accuracy']]
    round = np.arange(0, 100, 1)
    FLAvg_acc_NA = np.array(Baseline_acc)[0:100:1]
    FLAvg_acc = np.array(FLAvg_acc)[0:100:1]
    FLDH_acc = np.array(FLDH_acc)[0:100:1]
    krum_acc = np.array(krum_acc)[0:100:1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(round, Baseline_acc, label=(r'$\mathrm{Baseline}$'))
    ax.plot(round, FLAvg_acc, label="FedAvg")
    ax.plot(round, FLDH_acc, color='red', label="Ours")
    ax.plot(round, krum_acc, color='green', label="Krum$^{[11]}$")

    axins = ax.inset_axes((0.53, 0.55, 0.2, 0.125))
    axins.plot(round[90:103:1], Baseline_acc[90:103:1], color='dodgerblue')
    axins.plot(round[90:103:1], FLAvg_acc[90:103:1], color='orange')
    axins.plot(round[90:103:1], FLDH_acc[90:103:1], color='red')
    axins.plot(round[90:103:1], krum_acc[90:103:1], color='green')
    axins.set_ylim((0.705, 0.755))
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=0.75, linestyle='--')

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(linestyle='-.')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('迭代次数', fontsize=8)
    plt.ylabel('模型精度', fontsize=8)
    plt.legend()  # 显示标签
    plt.show()
    plt.savefig('图片.svg', dpi=600, format='svg')
