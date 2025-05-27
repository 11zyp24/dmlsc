import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader,Dataset,random_split,TensorDataset
import torchvision.transforms as transforms
import torch
import os


"""
初始化数据集的生成和不平衡处理类
"""
class Imbalance_NSdata(Dataset):
    cls_num = 10

    def __init__(self, N=600, dim=1024, img_size=32, imb_type='exp', imb_factor=0.05,
                 rand_number=0, train=True, reverse=False, transform=None):
        super(Imbalance_NSdata, self).__init__()
        self.N = N
        self.dim = dim
        self.img_size = img_size
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.reverse = reverse
        self.transform = transform
        self.train = train

        np.random.seed(rand_number)
        # 初始化完整数据集
        #self.data_train_tensor, self.target_train_tensor, self.data_val_tensor, self.target_val_tensor,self.data_test_tensor, self.target_test_tensor = self.NSdata_tensor()
        # 获取不平衡的样本数量并生成不平衡数据
        #img_num_list = self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, self.reverse)
        #self.data_train_tensor, self.target_train_tensor = self.gen_imbalanced_data(
            #img_num_list, self.data_train_tensor, self.target_train_tensor)

    def NSdata_tensor(self):
        #生成张量形式的模拟数据集，每个类别服从不同的多元正态分布
        means = [np.random.rand(self.dim) * (j + 1) for j in range(self.cls_num)]
        covariances = [np.eye(self.dim) * (j + 1) for j in range(self.cls_num)]
        data, target = [], []

        for j in range(self.cls_num): #每个类别服从不同的多元正态分布
            data_j = np.random.multivariate_normal(means[j], covariances[j], self.N)
            data.append(data_j)

            # 计算样本在各类别 m 下的概率密度值
            p_ij = np.array([
                multivariate_normal.pdf(data_j, mean=means[m], cov=covariances[m])
                for m in range(self.cls_num)]).T

            # 归一化 p_ij 矩阵为概率值
            p_ij = np.clip(p_ij, 1e-8, None)  # 避免数值过小
            p_ij /= p_ij.sum(axis=1, keepdims=True)  # 确保每行的总和为 1

            # 检查是否正确归一化
            if not np.allclose(p_ij.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Probability normalization failed, check p_ij.")

            # 根据多项式分布从类别概率中抽样得到类别标签
            target_j = np.array([np.random.choice(self.cls_num, p=p) for p in p_ij])
            target.extend(target_j)

        # 将数据从列表转换为 NumPy 数组形式
        data = np.vstack(data)
        target = np.array(target)

        # 划分数据集（按比例划分）
        from sklearn.model_selection import train_test_split
        # 先划分出训练集 + 验证集 (80%) 和测试集 (20%)
        data_train_val, data_test, target_train_val, target_test = train_test_split(
            data, target, test_size=0.2, stratify=target)

        # 再划分训练集 和验证集
        data_train, data_val, target_train, target_val = train_test_split(
            data_train_val, target_train_val, test_size=0.1, stratify=target_train_val)

        # 转换为张量形式
        data_train_tensor = torch.tensor(data_train, dtype=torch.float32).view(-1, 1, self.img_size, self.img_size)
        data_train_tensor = data_train_tensor.repeat(1, 3, 1, 1)  # 转为 RGB

        data_val_tensor = torch.tensor(data_val, dtype=torch.float32).view(-1, 1, self.img_size, self.img_size)
        data_val_tensor = data_val_tensor.repeat(1, 3, 1, 1)  # 转为 RGB

        data_test_tensor = torch.tensor(data_test, dtype=torch.float32).view(-1, 1, self.img_size, self.img_size)
        data_test_tensor = data_test_tensor.repeat(1, 3, 1, 1)  # 转为 RGB

        target_train_tensor = torch.tensor(target_train, dtype=torch.long)
        target_val_tensor = torch.tensor(target_val, dtype=torch.long)
        target_test_tensor = torch.tensor(target_test, dtype=torch.long)

        return data_train_tensor, target_train_tensor, data_val_tensor, target_val_tensor, data_test_tensor, target_test_tensor

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        #根据指定不平衡类型获取每个类别的样本数量
        img_max = len(data_train_tensor) / self.cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2, cls_num):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls = [int(img_max)] * cls_num

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls,data, target):
        print("gen fixd imbalanced data")
        #根据类别不平衡比率生成不平衡数据集
        new_data = []
        new_targets = []
        targets_np = target.numpy()
        classes = torch.unique(target).numpy()  # 获取类别列表
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            selec_idx = idx[:the_img_num]
            # 添加到新的数据和标签中
            new_data.append(data[selec_idx])
            new_targets.append(target[selec_idx])
        # 拼接数据和标签
        new_data = torch.cat(new_data, dim=0)  # 按第一个维度拼接
        new_targets = torch.cat(new_targets, dim=0)  # 同样拼接标签

        return new_data, new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__ == "__main__":
    # 数据生成参数
    cls_num = 10          # 类别数目
    dim = 1024           # 特征维度
    img_size = 32        # 图像尺寸
    N = 600           # 每类样本数量
    imb_type = 'exp'     # 不平衡类型（指数分布）
    imb_factor = 0.05    # 不平衡因子

    # 实例化生成器，生成数据集并进行不平衡处理
    generator = Imbalance_NSdata(N=N, dim=dim, img_size=img_size, imb_type=imb_type, imb_factor=imb_factor, reverse=False)

    # 生成训练集、验证集和测试集（张量形式）
    data_train_tensor, target_train_tensor, data_val_tensor, target_val_tensor, data_test_tensor, target_test_tensor = generator.NSdata_tensor()
    #train_class_counts1 = torch.bincount(target_train_tensor)
    #print("平衡训练集每类样本数:", train_class_counts1.tolist())
    # 应用不平衡处理，仅对训练集进行处理
    img_num_list = generator.get_img_num_per_cls(generator.cls_num, imb_type, imb_factor, reverse=False)
    data_train_tensor, target_train_tensor = generator.gen_imbalanced_data(img_num_list, data_train_tensor, target_train_tensor)

    # 检查数据分布
    train_class_counts = torch.bincount(target_train_tensor)
    val_class_counts = torch.bincount(target_val_tensor)
    test_class_counts = torch.bincount(target_test_tensor)

    print("训练集每类样本数:", train_class_counts.tolist())
    print("验证集每类样本数:", val_class_counts.tolist())
    print("测试集每类样本数:", test_class_counts.tolist())
    print("训练集数据维度:", data_train_tensor.shape)
    print("验证集数据维度:", data_val_tensor.shape)
    print("测试集数据维度:", data_test_tensor.shape)

    # 将特征和标签合并为字典
    train_data = {'features': data_train_tensor, 'labels': target_train_tensor}
    val_data = {'features': data_val_tensor, 'labels': target_val_tensor}
    test_data = {'features': data_test_tensor, 'labels': target_test_tensor}

    # 定义保存路径
    save_dir = r'C:\Users\hp\LSC\label-shift-correction\LSC\data_txt\NSdata_LT'
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录（如果不存在）

    # 保存训练集和测试集字典
    train_save_path = os.path.join(save_dir, 'NSdata_LT_train.pt')
    val_save_path = os.path.join(save_dir, 'NSdata_LT_val.pt')
    test_save_path = os.path.join(save_dir, 'NSdata_LT_test.pt')
    torch.save(train_data, train_save_path)
    torch.save(val_data, val_save_path)
    torch.save(test_data, test_save_path)

    print(f"训练集和测试集已保存至：{save_dir}")


    '''
    # 数据标准化：以不平衡训练集计算均值和标准差
    mean = data_train_imb.mean(dim=(0, 2, 3), keepdim=True)
    std = data_train_imb.std(dim=(0, 2, 3), keepdim=True)
    std[std == 0] = 1e-8  # 防止标准差为零
    
    # 定义标准化函数
    def normalize(tensor, mean, std):
        return (tensor - mean) / std

    # 标准化训练集和测试集
    data_train_imb_norm = normalize(data_train_imb, mean, std)
    data_test_norm = normalize(data_test_tensor, mean, std)

    # 包装数据集为 TensorDataset
    train_dataset = TensorDataset(data_train_imb_norm, target_train_imb)
    test_dataset = TensorDataset(data_test_norm, target_test_tensor)

    # 定义 DataLoader
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 检查生成的不平衡训练集类别分布
    train_class_counts = torch.bincount(target_train_imb)
    print("训练集每类样本数:", train_class_counts.tolist())
    print("训练集样本总数:", len(target_train_imb))
    print("测试集样本总数:", len(target_test_tensor))

    # 测试 DataLoader 输出
    for batch_data, batch_labels in trainloader:
        print("批数据维度:", batch_data.shape)
        print("批标签维度:", batch_labels.shape)
        break
    '''



