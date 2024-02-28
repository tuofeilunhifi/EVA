import io
import json
import ijson
import time
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import imghdr

# tarp -v create /mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/recipe -o /mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/wukong_test.tar

source_json_dir = "/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/subsets-16"
target_recipe_dir = "/mnt/pfs-guan-ssai/cv/yanghongfu/VL_pretrain/zh/zh_annotation/wukong/recipe_file"

source_json_list = []
if os.path.isdir(source_json_dir):
    for x in os.walk(source_json_dir):
        cur_json_list = [os.path.join(x[0], target) for target in x[2] if 'json' in target]
        if len(cur_json_list) > 0:
            source_json_list.extend(cur_json_list)
            print(cur_json_list)
else:
    source_json_list = [source_json_dir]

def single_json_warpper(index):
    source_json_path = source_json_list[index]
    print(index, source_json_path)
    with open(source_json_path, 'r', encoding="utf-8") as source_json_file:
        subset_name = os.path.splitext(os.path.basename(source_json_path))[0]
        target_recipe_path = os.path.join(target_recipe_dir, subset_name)
        with open(target_recipe_path, 'w') as target_recipe_file:
            objects = ijson.items(source_json_file, "item")  # find item_ids and its user_ids
            for item in objects:
                imagefile = item['image']
                caption = item['caption'].replace("\n", "")
                basename = os.path.splitext(os.path.basename(imagefile))[0]
                if os.path.exists(imagefile) and imghdr.what(imagefile) is not None:
                    target_recipe_file.write("{}.jpg\tfile:{}\n".format(basename, imagefile))
                    target_recipe_file.write("{}.txt\ttext:{}\n".format(basename, caption))
        source_json_file.close()
    target_recipe_file.close()
    return

print("cpu_count:{}".format(multiprocessing.cpu_count()))
cpu_worker_num = 16
json_num = len(source_json_list)
process_args = [i for i in range(json_num)]
# cpu_worker_num = 1
# process_args = [0]
print(f'| inputs:  {process_args}')
start_time = time.time()
with Pool(cpu_worker_num) as p:
    p.map(single_json_warpper, process_args)
print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')