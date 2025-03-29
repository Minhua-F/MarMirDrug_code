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
        elif iter == 2:
            plt.bar(index + width * 1, integer_list, width, label='transR', color='g')
        elif iter == 3:
            plt.bar(index + width * 2, integer_list, width, label='transD', color='r')
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

def read_best_row(filename, cols,row):
    # 初始化一个变量来存储最后一行的非空值
    last_line_non_empty_value = None
    count = 0
    max_num_ind = 0
    if cols == 1:
        max_num = 0
    else:
        max_num = 10000
    list_row = []
    # 打开文件
    with open(filename, 'r') as file:
        # 遍历文件的每一行
        for line in file:
            # 去除行尾的换行符，并按逗号或空格分割
            values = line.strip().split()
            count += 1
            # 检查指定列的值是否为空
            if cols < len(values) and values[cols]:
                # 如果非空，更新last_line_non_empty_value变量
                list_row.append(values[cols])
            if cols == 1:
                if max_num < float(values[cols]):
                    max_num = float(values[cols])
                    max_num_ind = count
            else:
                if max_num > float(values[cols]):
                    max_num = float(values[cols])
                    max_num_ind = count
    if cols == 1:
        last_line_non_empty_value = max(list_row)
    else:
        last_line_non_empty_value = min(list_row)
                # 返回最后一行的非空值，如果找不到则返回None
    print(max_num,last_line_non_empty_value)
    if max_num == last_line_non_empty_value:
        print('yes')

    # if cols ==  1:
    #     return float(list_row[row - 1]),max_num_ind
    # return last_line_non_empty_value
    # return max_num,max_num_ind
    return max_num, max_num_ind

def find_max_in_column(file_name, column_index):
    """
    找到名为file_name文件（txt格式）特定列的最大值，并返回最大值和它所在的第几行。
    分隔符是空格和制表符。

    参数:
    file_name (str): 文件名（包括路径和扩展名）
    column_index (int): 特定列的索引（从0开始）

    返回:
    tuple: (最大值, 行号)
    """
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        max_value = None
        max_value_line_num = -1

        for line_num, line in enumerate(lines, start=1):
            # 按空格和制表符分割行
            columns = line.split()  # split() 默认按空白字符（包括空格、制表符等）分割

            if len(columns) > column_index:
                column_value = columns[column_index]
                try:
                    # 尝试将列值转换为浮点数以进行比较
                    column_value_float = float(column_value)

                    if max_value is None or column_value_float > max_value:
                        max_value = column_value_float
                        max_value_line_num = line_num
                except ValueError:
                    # 如果列值不能转换为浮点数，则忽略该行
                    continue

        if max_value is None:
            raise ValueError(f"文件 {file_name} 中没有足够的数据或列索引 {column_index} 超出范围。")

        return max_value, max_value_line_num

    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_name} 未找到。")
    except Exception as e:
        raise e


def read_specific_cell(file_path, n, m, delimiter=' '):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if n > len(lines) or n < 1:
            return None  # 行号超出范围

        target_line = lines[n - 1].strip()  # 获取目标行
        columns = target_line.split(delimiter)

        if m > len(columns) or m < 1:
            return None  # 列号超出范围

        return columns[m - 1]

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def my_plot_line(file_name,my_obj):
    line_plot_list = ["Group I", "Group II", "Group III", "Group IV", "Group V"]
    if 'hit' not in my_obj:
        plt.figure(figsize=(6.4 * 2 * 2, 2.55 * 2 * 2 ))
    else:
        plt.figure(figsize=(6.4 * 3, 2.55 * 3))
    for i_cv in range(0, 5):
        data_name = "./KG2ID_iRefIndex/" + "D_DrugCV_" + CVType_Class_Name + "_CV_" + \
                    '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
        for iter in range(0, 2):
            if 'hit' not in my_obj:
                plt.subplot(2, 5, iter * 5 + i_cv + 1)
            else:
                plt.subplot(1, 5, i_cv + 1)

            for i_method in range(len(method_list)):
                dir_cur_name = './model/' + data_name[:] + '_' + '/'
                file_hit_meanrank_cur_name = dir_cur_name + method_list[i_method] + '_' + file_name

                x_column = 0
                y_column = 1
                if 'meanrank' in my_obj:
                    y_column = 2

                if  method_list[i_method] == 'transE_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, method_list[i_method] ,'b')
                elif method_list[i_method] == 'transH_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, method_list[i_method] ,'orange')
                elif method_list[i_method] == 'transR_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, method_list[i_method],'g')
                elif method_list[i_method] == 'transD_pytorch':
                    read_and_plot_two_columns(file_hit_meanrank_cur_name, x_column, y_column, method_list[i_method] ,'r')

            if iter == 1 and 'train' in my_obj:
                plt.ylim(2000,10000)
            elif iter == 1 and 'mean' in my_obj:
                plt.ylim(20, 100)
            elif iter == 1 and 'test' in my_obj:
                plt.ylim(0,200)

            # plt.axis("square")
            plt.xlabel('epoch')
            plt.ylabel('score')
            if iter == 0:
                if 'valid' in my_obj:
                    my_obj = 'Total testing loss'
                plt.title(line_plot_list[i_cv] + ':' + my_obj)
            else:
                # plt.title(line_plot_list[i_cv] + ':' + my_obj + '(zoom in locally)')
                plt.title('Localized enlargement of the ' + line_plot_list[i_cv] )
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
        row = 1
        best_mean , row = read_best_row(file_hit_meanrank_cur_name, 2,row)
        trans_meanrank_arr.append(best_mean)

        best_hit, row = read_best_row(file_hit_meanrank_cur_name, 1, row)
        trans_hit_arr.append(best_hit)

        # trans_meanrank_arr.append(read_best_row(file_hit_meanrank_cur_name, 2))
        # trans_hit_arr.append(read_best_row(file_hit_meanrank_cur_name, 1))

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
