import os

rootpath = './'
data_rootpath = '/data1/xjn/XuChen/Dir01_KG2Emb/knowledge_representation_pytorch-master/model/'
dir1_list = ['KG2ID_STRING/', 'KG2ID_iRefIndex/']
CVType_Class_Name = 'MultiClass' + '_000'
K_cv = 5
method_list = ["transE", "transD", "transH", "transR"]

for dir1 in dir1_list:
    for i_cv in range(K_cv):
        dir2 = "D_DrugCV_" + CVType_Class_Name + "_CV_" + '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
        for method in method_list:
            dir3 = dir2 + '_' + method + '/'
            path = rootpath + dir1 + dir2 + '/' + dir3
            data_dir3 = dir2 + '_' + method + '_pytorch' + '/'
            dataPath = data_rootpath + dir1 + data_dir3
            data_file = dataPath + method + '_pytorch' + '_%s_%s_best.pkl' % (dir2, str(100))
            os.system('cp ' + data_file + ' ' + path)
