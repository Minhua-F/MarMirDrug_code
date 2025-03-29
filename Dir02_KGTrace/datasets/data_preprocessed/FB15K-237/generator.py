import pickle
# import torch
import numpy as np
# import os
from preprocessing import K_means_clustering

def trans_pkl_to_bern(input_file,dataPath):
    with open(input_file, 'rb') as file:
        data = pickle.load(file)
        ent_embeddings = data.ent_embeddings.weight.data.cpu().numpy()
        rel_embeddings = data.rel_embeddings.weight.data.cpu().numpy()
        np.savetxt(dataPath + '/entity2vec.bern', ent_embeddings, fmt='%f', delimiter='\t')
        np.savetxt(dataPath + '/relation2vec.bern', rel_embeddings, fmt='%f', delimiter='\t')

    # str_input_file = ['./transD_pytorch_D_DrugCV_MultiClass_000_CV_05_01_100_best.pkl',
    #                   './transE_pytorch_D_DrugCV_MultiClass_000_CV_05_01_100_best.pkl',
    #                   './transH_pytorch_D_DrugCV_MultiClass_000_CV_05_01_100_best.pkl',
    #                   './transR_pytorch_D_DrugCV_MultiClass_000_CV_05_01_100_best.pkl']

    # str_path = './D_DrugCV_MultiClass_000_CV_05_01_trans'

    # for input_file in str_input_file:
    #     file = open(input_file, 'rb')
    #     data = pickle.load(file)
    #     ent_embeddings = data.ent_embeddings.weight.data.cpu().numpy()
    #     rel_embeddings = data.rel_embeddings.weight.data.cpu().numpy()
    #
    #     if input_file[2:8] == 'transD':
    #         if not os.path.exists(str_path + 'D'):
    #             os.mkdir(str_path+'D')
    #         np.savetxt(str_path + 'D' + '/entity2vec.bern', ent_embeddings, fmt='%f', delimiter='\t')
    #         np.savetxt(str_path + 'D' + '/relation2vec.bern', rel_embeddings, fmt='%f', delimiter='\t')
    #         print('transD done')
    #
    #     elif input_file[2:8] == 'transE':
    #         if not os.path.exists(str_path + 'E'):
    #             os.mkdir(str_path+'E')
    #         np.savetxt(str_path + 'E' + '/entity2vec.bern', ent_embeddings, fmt='%f', delimiter='\t')
    #         np.savetxt(str_path + 'E' + '/relation2vec.bern', rel_embeddings, fmt='%f', delimiter='\t')
    #         print('transE done')
    #
    #     elif input_file[2:8] == 'transH':
    #         if not os.path.exists(str_path+'H'):
    #             os.mkdir(str_path+'H')
    #         np.savetxt(str_path + 'H' + '/entity2vec.bern', ent_embeddings, fmt='%f', delimiter='\t')
    #         np.savetxt(str_path + 'H' + '/relation2vec.bern', rel_embeddings, fmt='%f', delimiter='\t')
    #         print('transH done')
    #
    #     elif input_file[2:8] == 'transR':
    #         if not os.path.exists(str_path+'R'):
    #             os.mkdir(str_path+'R')
    #         np.savetxt(str_path + 'R' + '/entity2vec.bern', ent_embeddings, fmt='%f', delimiter='\t')
    #         np.savetxt(str_path + 'R' + '/relation2vec.bern', rel_embeddings, fmt='%f', delimiter='\t')
    #         print('transR done')
    #
    #     file.close()

# file = open(str_input_file, 'rb')
# data = pickle.load(file)
# print(data)

# ent_embeddings = data.ent_embeddings.weight.data.cpu().numpy()
# rel_embeddings = data.rel_embeddings.weight.data.cpu().numpy()
# ent_proj_embeddings = data.ent_proj_embeddings.weight.data.cpu().numpy()
# rel_proj_embeddings = data.rel_proj_embeddings.weight.data.cpu().numpy()

# print(ent_embeddings)
# print(rel_embeddings)
# print(ent_proj_embeddings)
# print(rel_proj_embeddings)

# np.savetxt('./' + str_sss[i] + 'ent_embeddings.bern', ent_embeddings, fmt='%f', delimiter='\t')
# np.savetxt('./ent_embeddings.bern', ent_embeddings, delimiter='\t')

# file.close()

if __name__ == '__main__':
    rootpath = './'
    dir1_list = ['KG2ID_STRING/', 'KG2ID_iRefIndex/']
    CVType_Class_Name = 'MultiClass' + '_000'
    K_cv = 5
    method_list = ["transE", "transD", "transH", "transR"]
    k = 50

    for dir1 in dir1_list:
        for i_cv in range(K_cv):
            dir2 = "D_DrugCV_" + CVType_Class_Name + "_CV_" + '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
            txtPath = rootpath + dir1 + dir2 + '/'
            for method in method_list:
                dir3 = dir2 + '_' + method + '/'
                dataPath = rootpath + dir1 + dir2 + '/' + dir3
                input_file = dataPath + method + '_pytorch' + '_%s_%s_best.pkl' % (dir2, str(100))
                trans_pkl_to_bern(input_file, dataPath)
                K_means_clustering(k, txtPath, dataPath)


