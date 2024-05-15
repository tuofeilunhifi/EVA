import json
import os
from concurrent.futures import ProcessPoolExecutor

# 这里定义之前的过滤函数
def hae_image(x):
    return os.path.exists(x["image"])

def has_no_watermark(x):
    return x["pwatermark"] is not None and x["pwatermark"] < 0.8

def is_sfw(x):
    return x["punsafe"] is not None and x["punsafe"] < 0.5

def is_sim(x):
    return x["similarity"] is not None and x["similarity"] > 0.29

def is_aes(x):
    # return x["aesthetic_score"] is not None and x["aesthetic_score"] > 4.5
    return True


# 处理单个文件的函数
def process_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Dealing the path {json_path}...")

    filtered_data = [x for x in data if has_no_watermark(x) and is_sfw(x) and is_sim(x) and is_aes(x) and hae_image(x)]
    
    outfile_path = json_path.replace("/json_refined", "/json_flitered_biclip")
    if not os.path.exists(os.path.dirname(outfile_path)):
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    
    with open(outfile_path, "w") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    print(f"Dealing the path {json_path} finished! The percentage of the data is {len(filtered_data) / len(data) * 100:.2f}%.")
    return len(filtered_data) / len(data) * 100

# 假设你有一个包含所有JSON文件路径的列表
rp = "/mnt/pfs-mc0p4k/cv/team/panxuhao/data/laion2b-en-mada/20240416/json_refined"
json_paths = [os.path.join(rp, i) for i in os.listdir(rp)]

# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(process_file, json_paths))

with open("results_20240416.txt", "w") as f:
    for i, r in zip(json_paths, results):
        f.write(f"{i}: {r:.2f}%\n")

# 打印结果或进行后续处理
print("完成处理所有文件。")
