import numpy as np
import pandas as pd
import shutil
import os
import time


def func_KG2ID_file_empty(file_path):
    with open(file_path, 'r') as f:
        tline_1st = f.readline()
        f.close()
    return int(tline_1st)

# merge file
def func_increment_table(increment_table, New_DictElem, len_print):
    base_path = './'

    # increment_table by dummy
    temp_path = '%d' % (int(time.time() * 10000))
    temp_path = base_path + 'File_temp_' + temp_path + '.txt'
    increment_table.to_csv(temp_path, sep="\t", index=0, header=0)
    with open(temp_path, 'r') as f:  # temp file
        content_new = f.read()
        f.close()
    os.remove(temp_path)
    
    # update file with dummy
    # len_print = '%d' % (Len_Elem_old + len_Elem_new)
    with open(New_DictElem, 'r+') as f:  # New file
        next(f)  # skip the 1st line
        content_old = f.read()
        f.seek(0, 0)
        f.write(len_print + '\n' + content_old + content_new)
        f.close()



def func_Elem2ID_merge(Exist_DictElem, New_DictElem, Dict_Elem_new):
    # create new file
    shutil.copyfile(Exist_DictElem, New_DictElem)

    # get old len
    with open(New_DictElem, 'r') as f:
        Len_Elem_old = int(f.readline())
        f.close()

    # Set Diff: New/Old
    if func_KG2ID_file_empty(New_DictElem) == 0:
        Dict_ElemTable_t = pd.DataFrame([])
    else:
        Dict_ElemTable_t = pd.read_csv(New_DictElem, sep='\t', header=None, keep_default_na=False,
                                           skiprows=1)  # Get Table

    if len(Dict_ElemTable_t) == 0:
        Dict_Elem_t = set({})
    else:
        Dict_Elem_t = set(Dict_ElemTable_t.iloc[:, 0])

    New_Elem_add = Dict_Elem_new.difference(Dict_Elem_t)
    len_Elem_new = len(New_Elem_add)

    # increment_table
    len_final = (Len_Elem_old + len_Elem_new)
    len_print = '%d' % len_final
    List_table = pd.DataFrame(list(range(Len_Elem_old, len_final)))
    increment_table = pd.concat([pd.DataFrame(list(sorted(New_Elem_add))), List_table], axis=1, ignore_index=True)

    # merge file
    func_increment_table(increment_table, New_DictElem, len_print)


def func_KG2ID_merge(Exist_KG2ID, New_KG2ID, TableID2KG_new):
    # create new file
    shutil.copyfile(Exist_KG2ID, New_KG2ID)

    # get old len
    with open(New_KG2ID, 'r') as f:
        Len_KG_old = int(f.readline())
        f.close()

    # New/Old
    len_KG_new = np.size(TableID2KG_new, 0)

    len_final = (Len_KG_old + len_KG_new)
    len_print = '%d' % len_final

    # merge file
    func_increment_table(TableID2KG_new, New_KG2ID, len_print)

def func_KG2ID_new(New_ID2KG, input_KG):
    base_path = './'
    # increment_table by dummy
    temp_path = '%d' % (int(time.time() * 10000))
    temp_path = base_path + 'File_temp_' + temp_path + '.txt'

    file_t = open(temp_path, 'w')
    file_t.write("0")
    file_t.close()
    func_KG2ID_merge(temp_path, New_ID2KG, input_KG)

    os.remove(temp_path)


def func_increment_KG2ID(New_DictEntity, New_DictRelation, KG_new):
    # Dict Unique
    Dict_Entity_t = pd.read_csv(New_DictEntity, sep='\t', header=None, keep_default_na=False,
                                skiprows=1)  # Get Table
    Dict_Relation_t = pd.read_csv(New_DictRelation, sep='\t', header=None, keep_default_na=False,
                                  skiprows=1)  # Get Table
    Dict_Entity_list = list(Dict_Entity_t.iloc[:, 0])
    Dict_Relation_list = list(Dict_Relation_t.iloc[:, 0])

    increment_KG = pd.DataFrame([])
    # Convert KG to ID
    for index_KG in range(len(KG_new)):
        cur_entity = list(KG_new.iloc[index_KG, [0, 2]])
        cur_relation = np.array(KG_new.iloc[index_KG, 1])

        ID_Entity_cur = pd.DataFrame(Dict_Entity_list.index(ID_E) for ID_E in cur_entity)
        ID_Relation_cur = Dict_Relation_list.index(cur_relation)

        ID_Triplet_cur = pd.concat([ID_Entity_cur.iloc[0], pd.DataFrame([ID_Relation_cur]), ID_Entity_cur.iloc[1]],
                                   axis=1, ignore_index=True)

        increment_KG = pd.concat([increment_KG, ID_Triplet_cur], axis=0, ignore_index=True)

    return increment_KG
