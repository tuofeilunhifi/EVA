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

# tarp -v create recipe -o /mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/{00000}.tar

image_dir='/mnt/spaceai-internal/ark/crawler/raw/zhongqiyilian/20240127/'
source_json_dir = "/mnt/spaceai-internal/ark/crawler/raw/zhongqiyilian/20240127/jsonl/laion2B-en"
# target_json_dir = "/mnt/pfs-mc0p4k/cv/team/cjy/datasets/wds/laion2b-en/recipe/"
target_json_dir = "/mnt/pfs-guan-ssai/cv/cjy/data/laion2B-en/recipe"
source_json_list = []
i = 0
for x in os.walk(source_json_dir):
    cur_json_list = [os.path.join(x[0], target) for target in x[2] if 'jsonl' in target]
    if len(cur_json_list) > 0:
        source_json_list.extend(cur_json_list)
        print(i, cur_json_list)
        i += 1

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
        # objects = ijson.items(file, "item")  # find item_ids and its user_ids
        index = source_json_path.split('/')[-2]
        target_recipe_path = os.path.join(target_json_dir, index)
        with open(target_recipe_path, 'w') as target_recipe_file:
            for line in source_json_file.readlines():
                item = json.loads(line)
                image, caption = find(item, 'jpg_path'), find(item, 'content')
                image = os.path.join(image_dir, '/'.join(image.split('/')[-3:]))
                item_warpper = {'image': image, 'caption': caption}
                # if os.path.exists(image) and imghdr.what(image) is not None:
                if os.path.exists(image):
                    imagefile = item_warpper['image']
                    caption = item_warpper['caption'].replace("\n", "")
                    basename = os.path.splitext(os.path.basename(imagefile))[0]
                    target_recipe_file.write("{}.jpg\tfile:{}\n".format(basename, imagefile))
                    target_recipe_file.write("{}.txt\ttext:{}\n".format(basename, caption))
        target_recipe_file.close()
    source_json_file.close()
    return 

print("cpu_count:{}".format(multiprocessing.cpu_count()))
cpu_worker_num = 1
# process_args = [i for i in range(128)]
# process_args = [0,1,2,3,4,5,7,8,9,10,11,12,13,15,16,18,19,20,22,23,24,25,26,27,28,30,32,33,34,35,36,38,40,41,42,44,44,45,46,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,74,75,77,78,79,91,92,95,96,99,104,108,109,111,115]
process_args = [36]
print(len(process_args))
# cpu_worker_num = 1
# process_args = [0]
print(f'| inputs:  {process_args}')
start_time = time.time()
# with Pool(cpu_worker_num) as p:
#     p.map(single_json_warpper, process_args)
with ProcessPoolExecutor(cpu_worker_num) as p:
    p.map(single_json_warpper, process_args)
print(f'| TimeUsed: {time.time() - start_time:.1f}    \n')