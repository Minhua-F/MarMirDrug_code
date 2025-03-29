import numpy as np
import re

def roc(y_true, y_score, pos_label):
    """
    y_true：真实标签
    y_score：模型预测分数
    pos_label：正样本标签，如“1”
    """
    # 统计正样本和负样本的个数
    num_positive_examples = (y_true == pos_label).sum()
    num_negtive_examples = len(y_true) - num_positive_examples

    print(num_positive_examples)
    print(num_negtive_examples)

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1

    # 根据排序后的预测分数分别计算fpr和tpr
    for i in np.flip(np.argsort(y_score)):
        # 处理样本预测分数相同的情况
        if y_score[i] != score:
            fpr.append(fp / num_negtive_examples)
            tpr.append(tp / num_positive_examples)
            thresholds.append(score)
            score = y_score[i]

        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1


    fpr.append(fp / num_negtive_examples)
    tpr.append(tp / num_positive_examples)
    thresholds.append(score)

    return fpr, tpr, thresholds


# # 这个函数计算每个id_num的AUC值，并返回前n个AUC值最高的id_num列表
# def get_top_n_auc_id_nums(data_by_id_np, n):
#     # 初始化一个字典来存储每个id_num的AUC值
#     auc_scores = {}
#
#     # 遍历data_by_id_np字典，计算每个id_num的AUC值
#     for id_num, np_array in data_by_id_np.items():
#         np_array_for_id = data_by_id_np[id_num]  # 从字典中获取特定编号对应的NumPy数组
#         _, _, auc = roc(np_array_for_id[:, 2], np_array_for_id[:, 3], 1)
#         auc_scores[id_num] = sum(auc)
#
#         # 根据AUC值对id_num进行排序（降序）
#     sorted_id_nums = sorted(auc_scores, key=auc_scores.get, reverse=True)
#
#     # 提取前n个id_num
#     top_n_id_nums = sorted_id_nums[:n]
#
#     return top_n_id_nums

import matplotlib.pyplot as plt
#
# def read_and_plot_two_columns(filename, x_column, y_column,delimiter='\t'):
#
#     with open(filename, 'r') as file:
#         # 跳过文件头（如果有的话）
#         # next(file)  # 假设第一行是列名，如果不是，请注释掉此行
#         # next(file)  # 假设第一行是列名，如果不是，请注释掉此行
#         x_list = []
#         y_list = []
#         for line in file:
#             parts = re.findall(r'[-+]?\d*\.\d+|\d+', line)
#          #   parts = line.strip().split(delimiter)
#             if len(parts) > max(x_column, y_column):
#                 x_list.append(float(parts[x_column]))
#                 y_list.append(float(parts[y_column]))
#
#     return x_list, y_list
#
# fpr_t, tpr_t = read_and_plot_two_columns('transR_top10_tail_examples.txt',2,3)
# y_true = np.array(fpr_t)
# y_score = np.array(tpr_t)
# print(y_true)
# print("***********")
# print(y_score)

# fpr, tpr, thresholds = roc(y_true, y_score, pos_label=11)
#
#
# plt.plot(fpr, tpr)
# plt.axis("square")
# plt.xlabel("False positive rate")
# plt.ylabel("True positive rate")
# plt.title("ROC curve")
# plt.show()
# import numpy as np
#
# # y_true = np.array(
# #     [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
# # )
# y_true = np.array(
#     [0, 0, 1, 1, 1, 0, 1, 1,0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
# )
# y_score = np.array([
#     .9, .8, .7, .6, .55, .54, .53, .52, .51, .505,
#     .4, .39, .38, .37, .36, .35, .34, .33, .3, .1
# ])
#
# fpr, tpr, thresholds = roc(y_true, y_score, pos_label=1)
#
# import matplotlib.pyplot as plt
#
# plt.plot(fpr, tpr)
# plt.axis("square")
# plt.xlabel("False positive rate")
# plt.ylabel("True positive rate")
# plt.title("ROC curve")
# plt.show()

