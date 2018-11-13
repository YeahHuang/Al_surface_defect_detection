# -*-coding:utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import os.path as osp

label_dict = {'norm': 0,
              'defect1': 1,
              'defect2': 2,
              'defect3': 3,
              'defect4': 4,
              'defect5': 5,
              'defect6': 6,
              'defect7': 7,
              'defect8': 8,
              'defect9': 9,
              'defect10': 10,
              'defect11': 11
              }
chinese_dict = ['正常','不导电','擦花','横条压凹','桔皮','漏底','碰伤' ,'起坑' ,'凸粉' ,'涂层开裂','脏点','其他', '碰凹']

data_path = '/data/huangx/tianchi_competition/chusai/result_all'
print_chinese_flag = True # if true, print chinese label; else print defect 1,defect2 etc
out_csv_name = 'sub_all.csv' 

file_list = []
for first_path in os.listdir(data_path):
    first_path = osp.join(data_path, first_path)
    if '.txt' not in first_path:
        for second_path in os.listdir(first_path):
            if '.csv' in second_path and 'pred' not in second_path:
                file_list.append(osp.join(first_path, second_path))

img_num = 1000
file_num = len(file_list)
submit_all = [[-1 for j in range(file_num)] for i in range(img_num)]
print("You have %d submission files in total:"%file_num)

ans = ['norm' for i in range(img_num)]
for j in range(file_num):
    submit_data = pd.read_csv(file_list[j],header=None)
    print(file_list[j])
    for i in range(img_num):
       img_id = int(submit_data[0][i].replace('.jpg','')) 
       img_label = submit_data[1][i]
       submit_all[img_id][j] = img_label


flag_all_right = [1 for i in range(img_num)]    #judge if all model has the same output label

label_cnt = [0 for i in range(12)]              #count the num of each label
for i in range(img_num):
    for j in range(1,file_num):
        if submit_all[i][j]!=submit_all[i][j-1]:
            flag_all_right[i] = 0
            break

for i in range(img_num):
    label_cnt =[0 for i in range(12)]
    for j in range(file_num):
        qpp = submit_all[i][j]
        label_cnt[label_dict[qpp]]+=1
    max1 = -1
    max_label = -1
    for j in range(12):
        if label_cnt[j]>max1:
            max1 = label_cnt[j]
            max_label = j
    for j in range(12):
        if label_cnt[j]==max1 and j!=max_label:
            print(str(i)+': defect'+str(j))
    if max_label==0:
        ans[i] = 'norm'
    else:
        ans[i] = 'defect'+str(max_label)
file_name = [str(i)+'.jpg' for i in range(1000)]
k = pd.DataFrame({'file_name':file_name,'label':ans})
k.to_csv(out_csv_name, header=None, index=False)


for i in range(img_num):
    if flag_all_right[i]==0:
        print(i, end=":")
        for j in range(file_num):
            if print_chinese_flag:
              print(chinese_dict[label_dict[submit_all[i][j]]],end=',')
            else:
              print(submit_all[i][j],end=',')            
        print()

