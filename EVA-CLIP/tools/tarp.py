import io
import json
import ijson
import time
import os
from multiprocessing import Pool
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# source_recipe_dir = '/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/cc3m/recipe/'
# target_tar_dir = '/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/cc3m/tarfile/'
source_recipe_dir = '/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/cc12m/recipe/'
target_tar_dir = '/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/cc12m/tarfile/'

def tarp_funcation(index):
    index = "%05d" % index
    print(index)
    source_path = os.path.join(source_recipe_dir, index)
    target_path = os.path.join(target_tar_dir, index +'.tar')
    if os.path.exists(target_path):
        os.remove(target_path)
    os.system(f"tarp -v create {source_path} -o {target_path}")
    return 

cpu_count = multiprocessing.cpu_count()
print("cpu_count:{}".format(cpu_count))
cpu_worker_num = cpu_count
json_num = 1243
# json_num = 332
process_args = [i for i in range(json_num)]
# process_args = [12,20,25,26,34,61,71,78,105,113]
# process_args = [38]
# process_args.extend([i for i in range(62, json_num)])
# process_args = [61]
print(f'| inputs:  {process_args}')
start_time = time.time()
with Pool(cpu_worker_num) as p:
    p.map(tarp_funcation, process_args)
# with ProcessPoolExecutor(cpu_worker_num) as p:
#     p.map(tarp_funcation, process_args)
print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')