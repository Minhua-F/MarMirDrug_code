import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import math

trans_hit_arr = []
trans_meanrank_arr = []
trans_train_arr = []
trans_valid_arr = []
K_cv = 5

CVType_Class_Name = 'MultiClass' + '_000'
method_list = ["transE_pytorch", "transH_pytorch",
               "transR_pytorch", "transD_pytorch"]

def slice_list(lst, start, end, step):
    return lst[start:end:step]

def plot_bar(objList, objTittle):
    for iter in range(0, 4):
        # 设置条形图的宽度
        width = 0.2
        # 计算条形图的位置
        index = np.arange(5)
        # 绘制没有使用LMKG的精确度条形图
        #   plt.bar(index, last_hit_arr, width, label='Accuracy without LMKG')
        # 绘制使用了LMKG的精确度条形图，位置稍微偏移
        objList_my = slice_list(objList, iter, 21, 4)
        integer_list = [float(item) for item in objList_my]
        if iter == 0:
            plt.bar(index - width, integer_list, width, label='transE', color='b')
            print("E", integer_list)
        elif iter == 1:
            plt.bar(index, integer_list, width, label='transH', color='orange')
            print("H", integer_list)
        # elif iter == 2:
        #     plt.bar(index + width * 1, integer_list, width, label='transR', color='g')
        elif iter == 3:
            plt.bar(index + width * 1, integer_list, width, label='transD', color='r')
            print("D",integer_list)

    integer_list = [float(item) for item in objList]
    # 设置x轴的标签
    method_list = ["Group I", "Group II", "Group III", "Group IV", "Group V"]
    # plt.xlabel('Models')
    plt.xticks(index + width / 2, method_list)
    # 设置y轴的标签
    plt.ylabel('Score')
    # 设置图例
    plt.legend(loc='upper right')
    # 添加标题
    plt.title(objTittle)
    # 显示网格
    plt.grid(True)

    plt.xlim(-0.5 - 0.1, 4.7 + 0.1)  # 假设有5个分类，适当调整x轴的范围
    plt.ylim(0, 1.3 * max(integer_list))  # 根据数据自动调整y轴的上限
    # plt.tight_layout()
    # # 显示图形
    # plt.savefig(objTittle + '_my_histogram.png', dpi=300)
    # plt.show()
def read_and_plot_two_columns(filename, x_column, y_column,tittle_label,mycolor):

    with open(filename, 'r') as file:
        x_list = []
        y_list = []
        for line in file:
            # parts = re.findall(r'[-+]?\d*\.\d+|\d+', line)
            parts = line.strip().split('\t')
            x_list.append(float(parts[x_column]))
            y_list.append(float(parts[y_column]))
                # 绘制折线图
    plt.plot(x_list, y_list, marker='.', label=tittle_label[0:6], color=mycolor)
    # plt.ylim(0, int(max(y_list)))
    # print(int(len(y_list)))
    # plt.xlim(0, int(max(x_list)))
    # print(int(len(x_list)))
    # # plt.axis("square")
    # plt.xlim(0, int(max(x_list)))
    # plt.ylim(0,1000)

def read_file_last_row(filename, cols):
    # 初始化一个变量来存储最后一行的非空值
    last_line_non_empty_value = None

    # 打开文件
    with open(filename, 'r') as file:
        # 遍历文件的每一行
        for line in file:
            # 去除行尾的换行符，并按逗号或空格分割
            values = line.strip().split()

            # 检查指定列的值是否为空
            if cols < len(values) and values[cols]:
                # 如果非空，更新last_line_non_empty_value变量
                last_line_non_empty_value = values[cols]

                # 返回最后一行的非空值，如果找不到则返回None
    return last_line_non_empty_value

def read_best_row(filename, cols):
    # 初始化一个变量来存储最后一行的非空值
    last_line_non_empty_value = None

    list_row = []
    # 打开文件
    with open(filename, 'r') as file:
        # 遍历文件的每一行
        for line in file:
            # 去除行尾的换行符，并按逗号或空格分割
            values = line.strip().split()

            # 检查指定列的值是否为空
            if cols < len(values) and values[cols]:
                # 如果非空，更新last_line_non_empty_value变量
                list_row.append(values[cols])
    if cols == 1:
        last_line_non_empty_value = max(list_row)
    else:
        last_line_non_empty_value = min(list_row)
                # 返回最后一行的非空值，如果找不到则返回None
    return last_line_non_empty_value


def my_plot_line(file_name,my_obj):
    line_plot_list = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    if 'hit' not in my_obj:
        plt.figure(figsize=(6.4 * 2 * 2, 2.55 * 2 * 2 ))
    else:
        plt.figure(figsize=(6.4 * 3, 2.55 * 3))
    for i_cv in range(0, 5):
        data_name = "./KG2ID_iRefIndex/" + "D_DrugCV_" + CVType_Class_Name + "_CV_" + \
                    '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
        print(data_name)
        for iter in range(0, 2):
            if 'hit' not in my_obj:
                plt.subplot(2, 5, iter * 5 + i_cv + 1)
            else:
                plt.subplot(1, 5, i_cv + 1)

            for i_method in [3, 2, 1, 0]:  # 调整顺序为 TransD、TransR、TransH、TransE
                dir_cur_name = './model/' + data_name[:] + '_' + method_list[i_method] + '/'
                file_hit_meanrank_cur_name = dir_cur_name + method_list[i_method] + '_' + file_name

                x_column = 0
                y_column = 1
                if 'meanrank' in my_obj:
                    y_column = 2

                if method_list[i_method] == 'transE_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, 'TransE','b')  # TransE 为蓝色
                elif method_list[i_method] == 'transH_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, 'TransH','orange')  # TransH 为橙色
                elif method_list[i_method] == 'transR_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, 'TransR','g')  # TransR 为绿色
                elif method_list[i_method] == 'transD_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, 'TransD','r')  # TransD 为红色

             # 设置 x 轴范围
            if 'hit@10' in my_obj.lower() or 'meanrank' in my_obj.lower():
                plt.xlim(0, 25)  # hit@10 和 meanrank 的 x 轴范围为 0-25
            else:
                plt.xlim(0, 100)  # Total training loss 和 Total validing loss 的 x 轴范围为 0-100

            # 设置 x 轴和 y 轴标签
            plt.xlabel('Epoch')
            if 'hit@10' in my_obj.lower():
                plt.ylabel('Hit@10')  # hit@10 的 Y 轴名称
            elif 'meanrank' in my_obj.lower():
                plt.ylabel('Mean Rank')  # meanrank 的 Y 轴名称
            else:
                plt.ylabel('Loss')  # Total training loss 和 Total validing loss 的 Y 轴名称

            # 设置子图标题
            if iter == 0:
                if 'hit@10' in my_obj.lower():
                    plt.title('Hit@10 (' + line_plot_list[i_cv] + ')')  # hit@10 的标题
                elif 'meanrank' in my_obj.lower():
                    plt.title('Mean Rank (' + line_plot_list[i_cv] + ')')  # meanrank 的标题
                elif 'train' in my_obj.lower():
                    plt.title('Training Total Loss (' + line_plot_list[i_cv] + ')')  # Total training loss 的标题
                elif 'valid' in my_obj.lower():
                    plt.title('Validing Total Loss (' + line_plot_list[i_cv] + ')')  # Total validing loss 的标题
            else:
                # 只保留下半部分（Localized enlargement）
                if 'hit@10' not in my_obj.lower():
                    plt.title('Localized enlargement of ' + line_plot_list[i_cv])

            plt.grid(True)
            if 'hit@10' in my_obj.lower():
                plt.legend(loc='lower right')
            else:
                plt.legend(loc='upper right')

            if 'hit' in my_obj:
                break

    plt.tight_layout()
    plt.savefig(my_obj + '_my_line.png', dpi=300)
    plt.show()

# print(trans_hit_arr)
# print()
# print(trans_meanrank_arr)

for i_cv in range(0, 5):
    data_name = "./KG2ID_iRefIndex/" + "D_DrugCV_" + CVType_Class_Name + "_CV_" + \
                '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
    #   print(data_name)
    for i_method in range(len(method_list)):
        dir_cur_name = './model/' + data_name[:] + '_' + method_list[i_method] + '/'
        file_hit_meanrank_cur_name = dir_cur_name + method_list[i_method] + '_hit10_meanrank.txt'
        file_valid_cur_name = dir_cur_name + method_list[i_method] + '_Loss_valid.txt'
        file_train_cur_name = dir_cur_name + method_list[i_method] + '_Loss_train.txt'
        if not os.path.exists(dir_cur_name):
            print(dir_cur_name + ' does not exist')
        # trans_train_arr.append(read_file_last_row(file_train_cur_name,1))
        # trans_valid_arr.append(read_file_last_row(file_valid_cur_name,1))
        trans_hit_arr.append(read_best_row(file_hit_meanrank_cur_name, 1))
        trans_meanrank_arr.append(read_best_row(file_hit_meanrank_cur_name, 2))
        # trans_hit_arr.append(read_file_last_row(file_hit_meanrank_cur_name, 1))
        # trans_meanrank_arr.append(read_file_last_row(file_hit_meanrank_cur_name, 2))

plt.figure(figsize=(6.4 * 2.5, 2.55 * 2.5))
plt.subplot(1,2,1)
plot_bar(trans_hit_arr,'hit@10')
plt.subplot(1,2,2)
plot_bar(trans_meanrank_arr,'meanrank')
plt.tight_layout()
# 显示图形
plt.savefig('my_histogram.png', dpi=300)
plt.show()
my_plot_line('Loss_train.txt','Total training loss')
my_plot_line('Loss_valid.txt','Total validing loss')
my_plot_line('hit10_meanrank.txt','hit@10')
my_plot_line('hit10_meanrank.txt','meanrank')


def average_every_n_starting_at_k(lst, n, k):
    """Compute the average of every n-th value in the list starting at position k."""
    # Select every n-th value starting at position k
    selected_values = lst[k::n]

    # Filter out non-numeric values
    numeric_values = [v for v in selected_values if isinstance(v, (int, float))]

    # Compute the average of the selected values
    return sum(numeric_values) / len(numeric_values) if numeric_values else None


numeric_list = [float(item) for item in trans_hit_arr]
average_value1 = average_every_n_starting_at_k(numeric_list, 4, 0)
print(average_value1)

average_value2 = average_every_n_starting_at_k(numeric_list, 4, 3)
print(average_value2)

print((average_value2 - average_value1)/average_value1)

numeric_list = [float(item) for item in trans_meanrank_arr]
average_value3 = average_every_n_starting_at_k(numeric_list, 4, 0)
print(average_value3)

average_value4 = average_every_n_starting_at_k(numeric_list, 4, 3)
print(average_value4)

print((average_value4 - average_value3)/average_value3)
