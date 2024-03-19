import io
import json
import ijson
import time
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import imghdr
from concurrent.futures import ProcessPoolExecutor
import nltk
# nltk.download('punkt')

# tarp -v create recipe -o /mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/{00000}.tar
# /mnt/pfs-mc0p4k/cv/team/panxuhao/data/laion2b-en/aigc_laion200m/sub_data_10m

image_dir='/mnt/pfs-mc0p4k/cv/team/panxuhao/data/laion2b-en-mada/20240228/'
# source_json_dir = "/mnt/pfs-mc0p4k/cv/team/panxuhao/data/laion2b-en/aigc_laion200m/sub_data_10m"
# target_json_dir = "/mnt/pfs-guan-ssai/cv/cjy/data/laion2B-en/recipe_aigc_laion200m"
source_json_dir = "/mnt/pfs-mc0p4k/cv/team/panxuhao/data/laion2b-en-mada/20240228/json_refined"
target_json_dir = "/mnt/pfs-guan-ssai/cv/cjy/data/laion2B-en/recipe_1200m"
source_json_list = []
i = 0
for x in os.walk(source_json_dir):
    cur_json_list = [os.path.join(x[0], target) for target in x[2] if '.json' in target]
    if len(cur_json_list) > 0:
        source_json_list.extend(cur_json_list)
        print(i, cur_json_list)
        i += 1
source_json_list.sort()
print(source_json_list)

# 查找单个键
def find(dictData, target):
    queue = [dictData]
    while len(queue) > 0:
        data = queue.pop()
        for key, value in data.items():
            if key == target: return value
            elif type(value) == dict: queue.append(value)
    return None

def single_json_warpper(index):
    source_json_path = source_json_list[index]
    print(index, source_json_path)
    with open(source_json_path, 'r', encoding="utf-8") as source_json_file:
        objects = ijson.items(source_json_file, "item")  # find item_ids and its user_ids
        subset_name = os.path.splitext(os.path.basename(source_json_path))[0]
        target_recipe_path = os.path.join(target_json_dir, subset_name)
        with open(target_recipe_path, 'w') as target_recipe_file:
            for item in objects:
                imagefile = item['image']
                imagefile = os.path.join(image_dir, '/'.join(imagefile.split('/')[-4:]))
                caption = item['caption'].replace("\n", "")
                words = nltk.word_tokenize(caption)
                if os.path.exists(imagefile) and len(words) <= 200:
                    basename = os.path.splitext(os.path.basename(imagefile))[0]
                    basename = basename.replace('.', '_')
                    target_recipe_file.write("{}.jpg\tfile:{}\n".format(basename, imagefile))
                    target_recipe_file.write("{}.txt\ttext:{}\n".format(basename, caption))
        target_recipe_file.close()
    source_json_file.close()
    return 

cpu_count = multiprocessing.cpu_count()
print("cpu_count:{}".format(cpu_count))
cpu_worker_num = cpu_count
# process_args = [i for i in range(128)]
# process_args = [i for i in range(64)]
# process_args = [i for i in range(len(source_json_list))]
process_args = [16, 17, 18, 23, 24, 25, 26, 27]
# process_args = [i for i in range(64,128)]
# cpu_worker_num = 1
# process_args = [0]
print(f'| inputs:  {process_args}')
print("index_num:{}".format(len(process_args)))
start_time = time.time()
with Pool(cpu_worker_num) as p:
    p.map(single_json_warpper, process_args)
# with ProcessPoolExecutor(cpu_worker_num) as p:
#     p.map(single_json_warpper, process_args)
print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')