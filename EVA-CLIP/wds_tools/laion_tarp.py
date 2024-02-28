import io
import json
import ijson
import time
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def tarp_funcation(index):
    index = "%05d" % index
    print(index)
    os.system(f"tarp -v create /mnt/pfs-guan-ssai/cv/cjy/data/laion2B-en/recipe/{index} -o /mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/recipe/{index}.tar")
    return 

print("cpu_count:{}".format(multiprocessing.cpu_count()))
cpu_worker_num = 128
json_num = 128
process_args = [i for i in range(json_num)]
print(f'| inputs:  {process_args}')
start_time = time.time()
with ProcessPoolExecutor(cpu_worker_num) as p:
    p.map(tarp_funcation, process_args)
print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')