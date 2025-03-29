import os

CVType_Class_Name = 'MultiClass' + '_000'
method_list = ["transE_pytorch", "transD_pytorch",
               "transH_pytorch", "transR_pytorch"]

K_cv = 5
# for i_cv in range(K_cv):
for i_cv in range(1,6):
    data_name = "../DataKG/KG2ID_iRefIndex/" + "D_DrugCV_" + CVType_Class_Name + "_CV_" + \
           '%2.2d' % K_cv + '_' + '%2.2d' % (i_cv + 1)
    print(data_name)

    for i_method in range(len(method_list)):
    # for i_method in range(1):
        print(method_list[i_method])

        dir_cur_name = './model/' + data_name[10:] + '_' + method_list[i_method] + '/'
        if not os.path.exists(dir_cur_name):
            os.mkdir(dir_cur_name)

        os.system("python " + method_list[i_method] + ".py -d " + data_name)
        print(method_list[i_method] + " done!")

