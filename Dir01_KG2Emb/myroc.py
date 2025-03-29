import numpy as np
from sklearn.metrics import roc_auc_score  # 假设您使用sklearn的roc_auc_score来计算AUC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def read_third_column_data(filename, fix_str):
    drug_list = []
    drug_name = []
    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if fix_str in stripped_line:  #如果fix_str在stripped_line
                parts = stripped_line.split('\t')
                # print(len(parts))
                fourth_column_value = parts[1]
                string_with_prefix = parts[0]
                string_without_prefix = string_with_prefix.lstrip(fix_str)
                drug_list.append(fourth_column_value)
                drug_name.append(string_without_prefix)
    return drug_name,drug_list


def input_data(lstrip_str):
    # 使用函数
    output_filename = 'cell_line_entity.txt'
    for iteration in range(1, 6):
        tstr = str(iteration)
        filename = '../DataKG/KG2ID_iRefIndex/D_DrugCV_MultiClass_000_CV_05_0' + tstr + '/entity2id.txt'
        drug_name_list,drug_list = read_third_column_data(filename, lstrip_str)
        if iteration == 1:
            with open(output_filename, 'w') as outfile:
                for name,data in zip(drug_name_list,drug_list):
                    combined_string = name + '\t' + data + '\n'
                    outfile.write(combined_string)  # 每个数据项后添加换行符
        else:
            with open(output_filename, 'a') as outfile:
                for name,data in zip(drug_name_list,drug_list):
                    combined_string = name + '\t' + data + '\n'
                    outfile.write(combined_string)  # 每个数据项后添加换行符


#在一个文件中找到字符串
def check_string_and_return_name(filename,search_string):

    with open(filename, 'r') as file:
        # 逐行读取文件
        for line in file:
            # 去除行尾的空白字符，并按空格或制表符分割成列
            columns = line.strip().split()

            # 检查是否有足够的列，并且第二列是否包含搜索的字符串
            if len(columns) > 1 and columns[1] == search_string:
                string_with_prefix = columns[0]
                # string_without_prefix = string_with_prefix.lstrip(lstrip_str)
                return  string_with_prefix
    return search_string


#读取finish_trans
def read_file_after_non_numeric(filename, count_threshold):
    # 初始化变量
    lines = []
    fourth_column_values = []
    non_numeric_count = 0
    started_reading = False
    max_fourth_column = None

    with open(filename, 'r') as file:
        for line in file:
            # 去除行首尾的空白字符
            stripped_line = line.strip()

            # 如果行包含特定字符串，则增加计数器
            if './model/' in stripped_line:
                non_numeric_count += 1

                # 如果达到阈值，开始读取数据
                if non_numeric_count >= count_threshold and not started_reading:
                    started_reading = True
                    continue
                    # 如果已经开始读取数据，遇到下一个特定行则停止
                elif started_reading:
                    started_reading = False
                    break

                    # 如果已经开始读取数据，则处理行数据
            if started_reading:
                parts = stripped_line.split('\t')
                if len(parts) >= 4:
                    # 尝试将第四列的值转换为整数
                    try:
                        fourth_column_value = int(parts[3])
                        fourth_column_values.append(fourth_column_value)
                        # 更新第四列的最大值
                        if max_fourth_column is None or fourth_column_value > max_fourth_column:
                            max_fourth_column = fourth_column_value
                    except ValueError:
                        # 忽略不能转换为整数的行
                        pass
                        # 添加原始行到lines列表
                lines.append(line)

                # 返回所有行的数据和第四列的最大值
    return lines, max_fourth_column


# 假设ROC.roc函数返回AUC值，这里我们使用sklearn的roc_auc_score作为示例
def calculate_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

# 这个函数计算每个id_num的AUC值，并返回前n个AUC值最高的id_num（整数）列表
def get_top_n_auc_id_nums(data_by_id_np, n):
    # 初始化一个字典来存储每个id_num的AUC值
    auc_scores = {}

    # 遍历data_by_id_np字典，计算每个id_num的AUC值
    for id_num, np_array in data_by_id_np.items():
        np_array_for_id = data_by_id_np[id_num]  # 从字典中获取特定编号对应的NumPy数组
        # if len(np.unique(np_array_for_id[:, 2])) > 1 and check_string_in_second_column(id_num,'drug_entity.txt'):
        if len(np.unique(np_array_for_id[:, 2])) > 1 :
            # auc = calculate_auc(np_array_for_id[:, 2], np_array_for_id[:, 3])
        # 确保id_num是整数类型并存储AUC值
            auc_scores[id_num] = auc
        else:
            auc_scores[id_num] = 0

        # 根据AUC值对id_num进行排序（降序）
    sorted_id_nums = sorted(auc_scores, key=auc_scores.get, reverse=True)

    # 提取前n个id_num，并确保它们是整数（尽管它们已经是，但为了清晰性）
    top_n_id_nums = [id_num for id_num in sorted_id_nums[:n]]

    return top_n_id_nums


def myroc(i_cv,spical_id,page):
    # method_list = [
    #     'transE',
    #     'transH',
    #     'transR',
    #     'transD'
    # ]
    method_list = [
        'finish_transE',
        'finish_transH',
        'finish_transR',
        'finish_transD'
    ]

    auc_list = []
    for method in method_list:
        lines , max_value = read_file_after_non_numeric(method + '_top10_tail_examples.txt',i_cv + 1)
        # 处理第四列数据，大值变小值，小值变大值
        processed_lines = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                # 假设第四列是整数，转换为整数进行比较和交换
                fourth_col = int(parts[3])
                # 找到第四列中的另一个值，这里我们假设它是0和1（基于你提供的示例）
                other_value = max_value + 1 - fourth_col
                # 更新第四列的值
                parts[3] = str(other_value)
                processed_lines.append('\t'.join(parts) + '\n')

            # 将处理后的数据写回文件
        f_data = method + '_finish_data.txt'
        # 如果需要覆盖原文件，将 'new_file.txt' 改为 'original_file.txt'
        with open(f_data, 'w') as file:
            file.writelines(processed_lines)

        print("生成",f_data)

        char_i_cv = str(i_cv+1)
        # for iteration in range(1, 6):
        #     tstr = str(iteration)
        filename = '../DataKG/KG2ID_iRefIndex/D_DrugCV_MultiClass_000_CV_05_0' + char_i_cv + '/valid2id.txt'
        print(char_i_cv)

        file_name = 'fast_finis_com_'+method+char_i_cv+'.txt'

        # 读取 test_test.txt 的内容
        with open(f_data, 'r') as test_file:
            test_lines = test_file.readlines()

        # 读取 test_valid.txt 的内容
        with open(filename, 'r') as valid_file:
            valid_lines = valid_file.readlines()
        # 用于存储结果的字典，key为前三列的组合，value为对应的行
        max_values = {}

        # 遍历 test_test.txt 的每一行
        for test_line in test_lines:
            parts = test_line.strip().split('\t')
            if len(parts) >= 4:
                key = '\t'.join(parts[:3])  # 前三列作为键
                value = parts[3]  # 第四列的值

                # 如果该键不存在或者当前值更大，则更新
                if key not in max_values or value > max_values[key][1]:
                    max_values[key] = (test_line.strip(), value)

        # 遍历 test_valid.txt 的每一行
        results = []
        for valid_line in valid_lines:
            clean_valid_line = valid_line.strip()  # 去除首尾空白
            if not clean_valid_line:
                continue  # 跳过空行

            # 遍历 test_test.txt 的每一行，检查是否包含 valid_line
            for key, (test_line, _) in max_values.items():
                if clean_valid_line in test_line:
                    # 如果包含，记录对应的 test_test.txt 的行
                    if test_line not in results:
                        results.append(test_line)

        # 将结果写入 test_result.txt
        with open(file_name, 'w') as result_file:
            for result in results:
                result_file.write(result + '\n')

        print("处理完成，结果已保存到",file_name)

        # 初始化一个字典来存储每个编号对应的所有行内容
        data_by_id = {}

        # 读取txt文件并处理数据
        with open(file_name, 'r') as file:
            for line in file:
                # 分割每一行数据
                parts = line.strip().split('\t')

                # id_num, _, third_col, _ = parts
                _,id_num, third_col, _ = parts

                # 将第三列的'12'转换为1，'11'转换为0，并转换为整数
                processed_third_col = int(1 if third_col == '12' else 0)
                # processed_third_col = int(1 if third_col == '11' else 0)

                # print(id_num, processed_third_col)

                # 将行数据转换为整数列表，并添加到对应编号的列表中
                # row_data = [int(id_num), int(parts[1]), processed_third_col, int(parts[3])]
                row_data = [ int(parts[0]),int(id_num), processed_third_col, int(parts[3])]

                # 如果编号在字典中，追加行数据；否则初始化一个列表
                if id_num in data_by_id:
                    data_by_id[id_num].append(row_data)
                else:
                    data_by_id[id_num] = [row_data]
                # print(len(data_by_id[id_num]))

                # 将每个编号对应的行数据列表转换为NumPy数组，指定dtype为整数
        data_by_id_np = {k: np.array(v, dtype=int) for k, v in data_by_id.items()}


        # 假设data_by_id_np是之前创建并填充的字典
        # n是想要的前n个AUC值最高的id_num的数量
        n = 10 # 前10个
        top_n_auc_id_nums = get_top_n_auc_id_nums(data_by_id_np, n)
        if method == 'finish_transE' and i_cv == 0:
            with open('finish_trans_top_index.txt', 'w') as file:  # 使用'a'模式打开文件以追加内容
                file.write(' '.join(map(str, top_n_auc_id_nums)) + '\n')
        else:
            with open('finish_trans_top_index.txt', 'a') as file:  # 使用'a'模式打开文件以追加内容
                file.write(' '.join(map(str, top_n_auc_id_nums)) + '\n')
        # if i_cv == 0 and method == 'finish_transE':
        #     with open('trans_top_index.txt', 'w') as file:
        #         file.writelines(processed_lines)
        # print("前{}个AUC值最高的id_num列表:".format(n))
        # print(top_n_auc_id_nums)
        # for id_num in top_n_auc_id_nums:
        #     np_array_for_id = data_by_id_np[id_num]
        #
        #     # print(np_array_for_id)
        #     # fpr, tpr, thersholds = roc_curve(np_array_for_id[:,2], np_array_for_id[:,3], pos_label=1)
        #     auc = calculate_auc(np_array_for_id[:, 2], np_array_for_id[:, 3])
        #     print(auc)
        #     print()
        if method == 'finish_transE':
            if spical_id == '0':
                # np_array_for_id = data_by_id_np[top_n_auc_id_nums[0]]
                nofirst = top_n_auc_id_nums[0]
            else:
                nofirst = spical_id
        # else:
        #     np_array_for_id = data_by_id_np[nofirst]
#这里开始是断开的
        # np_array_for_id = data_by_id_np[nofirst]
        # print(nofirst)
        # fpr, tpr, thersholds = roc_curve(np_array_for_id[:,2], np_array_for_id[:,3], pos_label=1)
        # auc = calculate_auc(np_array_for_id[:, 2], np_array_for_id[:, 3])
        #
        #
        # print(auc)
        # mytitle = check_string_and_return_name('cell_line_entity.txt',nofirst)
        # if page > 1:
        #     plt.subplot(3 , 5, 5 * (page - 2) + i_cv+1)
        # else:
        #     plt.subplot(1, 5, 5 * (page - 1) + i_cv + 1)
        # tmethod = method[7:]
        # plt.plot(fpr, tpr,label = tmethod + f'(AUC={auc:.2f})')
        # auc_list.append(auc)
        # plt.ylim(0-0.02, 1+0.02)
        # plt.xlim(0-0.02, 1+0.02)
        # plt.axis("square")
        # plt.ylabel("sensitivities")
        # plt.xlabel("1-Specificity")
        # plt.title(line_plot_list[i_cv] + "ROC curve on " + mytitle)
        # plt.grid(True)
        # plt.legend()
    return auc_list

input_data('DrugName:')

line_plot_list = ["(I):", "(II):", "(III):", "(IV):", "(V):"]

auc_h_list = []

plt.figure(figsize=(6.4 * 3, 2.55 * 3))
# nofirst = str(find_common_value_with_min_columns_in_file('finish_trans_top_index.txt'))
auc_h_list.append(myroc(0,'14699',2))
auc_h_list.append(myroc(1,'14699',2))
auc_h_list.append(myroc(2,'14699',2))
auc_h_list.append(myroc(3,'14699',2))
auc_h_list.append(myroc(4,'14699',2))

# 调整子图之间的间距和边距
plt.tight_layout()
plt.savefig(str(0) + '_my_roc.png',dpi=300)
plt.show()
print(auc_h_list)


#
# plt.figure(figsize=(6.4 * 2 * 2, 2.55 * 3 * 2))
#
# auc_h_list.append(myroc(0,'13020',2))
# auc_h_list.append(myroc(1,'13020',2))
# auc_h_list.append(myroc(2,'13020',2))
# auc_h_list.append(myroc(3,'13020',2))
# auc_h_list.append(myroc(4,'13020',2))
# #
# # 调整子图之间的间距和边距
# # plt.tight_layout()
# # plt.savefig(str(1) + '_my_roc.png',dpi=300)
# # plt.show()
#
# # plt.figure(figsize=(6.4 * 3, 2.4 * 3))
# myroc(0,'12137',3)
# myroc(1,'12137',3)
# myroc(2,'12137',3)
# myroc(3,'12137',3)
# myroc(4,'12137',3)
# #
# # 调整子图之间的间距和边距
# # plt.tight_layout()
# # plt.savefig(str(2) + '_my_roc.png',dpi=300)
# # plt.show()
#
# # plt.figure(figsize=(6.4 * 3, 2.4 * 3))
# myroc(0,'12680',4)
# myroc(1,'12680',4)
# myroc(2,'12680',4)
# myroc(3,'12680',4)
# myroc(4,'12680',4)
# #
# # 调整子图之间的间距和边距
# plt.tight_layout()
# plt.savefig('simple' + '_my_roc.png',dpi=300)
# plt.show()