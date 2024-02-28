import io
import json
import ijson
import time
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

def tarp_funcation(index):
    os.system(f"tarp -v create /mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/recipe_file/wukong-100m-part-{index} -o /mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/wukong/wukong-100m-part-{index}.tar")
    return 

print("cpu_count:{}".format(multiprocessing.cpu_count()))
cpu_worker_num = 128
json_num = 16
process_args = [i for i in range(json_num)]
# cpu_worker_num = 1
# process_args = [0]
print(f'| inputs:  {process_args}')
start_time = time.time()
with Pool(cpu_worker_num) as p:
    p.map(tarp_funcation, process_args)
print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')