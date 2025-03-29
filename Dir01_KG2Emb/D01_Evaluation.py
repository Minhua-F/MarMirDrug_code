import numpy as np
import pandas as pd
import shutil
import os
import time

# ------- part 01: CV_index to train/test lists -------
# merge different classes from CV_index
def func_merge_classes_CV_index(CV_index, N_class, N_pos_neg, K_cv):
    CV_index_no_class = []
    class_list = []
    # initialization CV_index[i_pos_neg][i_cv]
    for i_pos_neg in range(N_pos_neg):
        ind_t1 = []
        ind_t2 = []
        for i_cv in range(K_cv):
            ind_t1.append([])
            ind_t2.append([])
        # ind_t = [ [], [], [], [], [] ]
        CV_index_no_class.append(ind_t1)
        class_list.append(ind_t2)
    # CV_index_new, class_list = [[[], [], [], [], []], [[], [], [], [], []]]

    # merging classes
    for i_class in range(N_class):
        for i_pos_neg in range(N_pos_neg):
            for i_cv in range(K_cv):
                # index list
                CV_index_no_class[i_pos_neg][i_cv].extend(CV_index[i_class][i_pos_neg][i_cv])
                # class list
                len_cur = len(CV_index[i_class][i_pos_neg][i_cv])
                class_list[i_pos_neg][i_cv].extend(list(i_class * np.ones(len_cur, dtype=int)))
    # classes merged
    return CV_index_no_class, class_list


# merge ind_cv of folds from CV_index_no_class
def func_merge_ind_cv_CV_index(CV_index_no_class, ind_cv, N_pos_neg):
    index_pos_neg = []
    # initialization CV_index[i_pos_neg][i_cv]
    for i_pos_neg in range(N_pos_neg):
        index_pos_neg.append([])
    # index_pos_neg = [ [], [] ]
    #
    # merging ind_cv
    for i_cv in ind_cv:
        for i_pos_neg in range(N_pos_neg):
            index_pos_neg[i_pos_neg].extend(CV_index_no_class[i_pos_neg][i_cv])
    # ind_cv merged
    return index_pos_neg


# assign labels to index_pos_neg
def func_index_label_list(index_pos_neg, class_pos_neg, N_pos_neg):
    index_list = []
    class_list = []
    label_list = []
    for i_pos_neg in range(N_pos_neg):
        index_list.extend(index_pos_neg[i_pos_neg])
        class_list.extend(class_pos_neg[i_pos_neg])

        len_cur = len(index_pos_neg[i_pos_neg])
        len_cur2 = len(class_pos_neg[i_pos_neg])
        if len_cur != len_cur2:  # error
            print('index and class mismatch!')
            return 0, 0, 0
        label_list.extend(list(i_pos_neg * np.ones(len_cur, dtype=int)))
    return index_list, class_list, label_list


def func_get_list_for_index_label(CV_index_no_class, class_list, K_cv, N_pos_neg):
    index_train_CV = []  # list for train
    index_test_CV = []  # list for test
    class_train_CV = []  # class for train
    class_test_CV = []  # class for test
    label_train_CV = []  # label for train
    label_test_CV = []  # label for test
    for i_cv in range(K_cv):
        test_i_cv = set([i_cv])
        # train and test folds in CV
        all_i_cv = set(range(K_cv))
        train_i_cv = set.difference(all_i_cv, test_i_cv)

        # cv merge
        # index for train and test
        index_pos_neg_train = func_merge_ind_cv_CV_index(CV_index_no_class, list(train_i_cv), N_pos_neg)
        index_pos_neg_test = func_merge_ind_cv_CV_index(CV_index_no_class, list(test_i_cv), N_pos_neg)

        # index for train and test
        class_pos_neg_train = func_merge_ind_cv_CV_index(class_list, list(train_i_cv), N_pos_neg)
        class_pos_neg_test = func_merge_ind_cv_CV_index(class_list, list(test_i_cv), N_pos_neg)

        # list for train and test
        index_list_train, class_list_train, label_list_train = \
            func_index_label_list(index_pos_neg_train, class_pos_neg_train, N_pos_neg)
        index_list_test, class_list_test, label_list_test = \
            func_index_label_list(index_pos_neg_test, class_pos_neg_test, N_pos_neg)

        # append i_cv
        index_train_CV.append(index_list_train)
        index_test_CV.append(index_list_test)
        class_train_CV.append(class_list_train)
        class_test_CV.append(class_list_test)
        label_train_CV.append(label_list_train)
        label_test_CV.append(label_list_test)
    # end of for loog

    # index_train_CV, index_test_CV, label_train_CV, label_test_CV = [[1], [2], .. [k]]
    return index_train_CV, index_test_CV, class_train_CV, class_test_CV, label_train_CV, label_test_CV

# ------- part 01: end (CV_index to train/test lists) -------














