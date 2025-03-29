import os

rootpath = './'
dir1_list = ['KG2ID_STRING/', 'KG2ID_iRefIndex/']
CVType_Class_Name = 'MultiClass' + '_000'
K_cv = 5
data_rootpath = '/data1/xjn/XuChen/Dir01_KG2Emb/DataKG/'
txt_list = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt', 'entity2id.txt', 'relation2id.txt']
method_list = ["transE", "transD", "transH", "transR"]

for dir1 in dir1_list:
    if not os.path.exists(rootpath + dir1):
        os.mkdir(rootpath + dir1)
    for i_cv in range(K_cv):
        dir2 = "D_DrugCV_" + CVType_Class_Name + "_CV_" + '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
        if not os.path.exists(rootpath + dir1 + dir2):
            os.mkdir(rootpath + dir1 + dir2)
        for file_txt in txt_list:
            os.system('cp ' + data_rootpath + dir1 + dir2 + '/' + file_txt + ' ' + rootpath + dir1 + dir2)
        for method in method_list:
            dir3 = dir2 + '_' + method + '/'
            if not os.path.exists(rootpath + dir1 + dir2 + '/' + dir3):
                os.mkdir(rootpath + dir1 + dir2 + '/' + dir3)
            if not os.path.exists(rootpath + dir1 + dir2 + '/' + dir3 + 'vocab'):
                os.mkdir(rootpath + dir1 + dir2 + '/' + dir3 + 'vocab')


