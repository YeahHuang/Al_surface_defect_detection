# -*- coding: utf-8 -*-
'''
Edit from training code by @hubert-chen
Serve as the test.py for 12 classes. 
'''
import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import model_v4
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # 随机种子
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)

    # 获取当前文件名，用于创建模型及结果文件的目录
    file_name = os.path.basename(__file__).split('.')[0]
    # 创建保存模型和结果的文件夹
    if not os.path.exists('./model/%s' % file_name):
        os.makedirs('./model/%s' % file_name)
    if not os.path.exists('./result/%s' % file_name):
        os.makedirs('./result/%s' % file_name)
    # 创建日志文件
    if not os.path.exists('./result/%s.txt' % file_name):
        with open('./result/%s.txt' % file_name, 'w') as acc_file:
            pass
    with open('./result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

    # 默认使用PIL读图
    def default_loader(path):
        # return Image.open(path)
        return Image.open(path).convert('RGB')

    # 测试集图片读取
    class TestDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, filename

        def __len__(self):
            return len(self.imgs)

    # 数据增强：在给定角度中随机进行旋转
    class FixedRotation(object):
        def __init__(self, angles):
            self.angles = angles

        def __call__(self, img):
            return fixed_rotate(img, self.angles)

    def fixed_rotate(img, angles):
        angles = list(angles)
        angles_num = len(angles)
        index = random.randint(0, angles_num - 1)
        return img.rotate(angles[index])
        
    # 测试函数
    def test(test_loader, model):
        csv_map = OrderedDict({'filename': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            # bs, ncrops, c, h, w = images.size()
            filepath = [i.split('/')[-1] for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

            with torch.no_grad():
                y_pred = model(image_var)

                # get the index of the max log-probability
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
            csv_map['filename'].extend(filepath)
            for output in smax_out:
                prob = ';'.join([str(i) for i in output.data.tolist()])
                csv_map['probability'].append(prob)

        result = pd.DataFrame(csv_map)
        result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

        # 转换成提交样例中的格式
        sub_filename, sub_label = [], []
        for index, row in result.iterrows():
            sub_filename.append(row['filename'])
            pred_label = np.argmax(row['probability'])
            if row['probability'][pred_label]<valid_prob_threshold or pred_label>11:
                print("file_name=%s, ori_pred=%d, prob=%f"%(row['filename'],pred_label,row['probability'][pred_label]))
                pred_label = 11
            if pred_label<11:
                summ[pred_label] += row['probability'][pred_label]
                num[pred_label] += 1
            if pred_label == 0:
                sub_label.append('norm')
            else:
                sub_label.append('defect%d' % pred_label)
        result.to_csv('./result/%s/%s_pred_result.csv'%(file_name, submission_name))
        # 生成结果文件，保存在result文件夹中
        submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
        submission.to_csv('./result/%s/%s.csv' % (file_name,submission_name), header=None, index=False)
        return

    # 保存最新模型以及最优模型
    def save_checkpoint(state, is_best, is_lowest_loss, filename='./model/%s/checkpoint.pth.tar' % file_name):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, './model/%s/model_best.pth.tar' % file_name)
        if is_lowest_loss:
            shutil.copyfile(filename, './model/%s/lowest_loss.pth.tar' % file_name)

    # 用于计算精度和时间的变化
    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # 学习率衰减：lr = lr / lr_decay
    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    # 计算top K准确率
    def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0) ???
        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT
    
    # 程序主体

    # GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    # 小数据集上，batch size不易过大
    batch_size = 12
    # 进程数量，最好不要超过电脑最大进程数，尽量能被batch size整除，如果出现ran out of input错误可以设置workers为0
    workers = 12
    valid_prob_threshold=0.4
    # epoch数量，分stage进行，跑完一个stage后降低学习率进入下一个stage
    stage_epochs = [6, 3, 3]  
    # 初始学习率
    lr = 1e-4
    # 学习率衰减系数 (new_lr = lr / lr_decay)
    lr_decay = 5
    # 正则化系数
    weight_decay = 1e-4

    #submission_name='submission_vda_664_t0_lowest_loss.csv'
    submission_name = 'submission_ljh_new_t0.4.csv'
    # 参数初始化
    stage = 0
    start_epoch = 0
    total_epochs = sum(stage_epochs)
    best_precision = 0
    lowest_loss = 100

    # 训练及验证时的打印频率，用于观察loss和acc的实时变化
    print_freq = 1
    # 验证集比例
    val_ratio = 0.1
    # 是否只验证，不训练
    evaluate = True
    # 是否从断点继续跑
    resume = False
    # 创建inception_v4模型
    model = model_v4.v4(num_classes=12)
    model = torch.nn.DataParallel(model).cuda()
    
    #data_augmentation
    double_label = [3,6,7,8]
    triple_label = [1,9,12]


    # 读取训练图片列表
    all_data = pd.read_csv('/data/huangx/tianchi_competition/data/label12.csv')
    # 分离训练集和测试集，stratify参数用于分层抽样
    train_data_list, val_data_list = train_test_split(all_data, test_size=val_ratio, random_state=666, stratify=all_data['label'])
    # 读取测试图片列表
    test_data_list = pd.read_csv('/data/huangx/tianchi_competition/data/test_1009.csv')



    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 测试集图片变换
    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((400, 400)),
                                transforms.CenterCrop(384),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    # 生成图片迭代器

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=workers)
    
    # 读取最佳模型，预测测试集
    #model_name = 'vda_664'
    model_name = 'ljh_main_inception2'
    best_model = torch.load('./model/%s/lowest_loss.pth.tar' % model_name)
    model.load_state_dict(best_model['state_dict'])
    summ = [0.0 for i in range(11)]
    num = [0 for i in range(11)]
    test(test_loader=test_loader, model=model)

    for i in range(11):
        if num[i]==0:
            print("defect%d num=0"%i)
        else:
            print("%f / %d = %f"%(summ[i],num[i], summ[i]/num[i]))
    # 释放GPU缓存
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()



