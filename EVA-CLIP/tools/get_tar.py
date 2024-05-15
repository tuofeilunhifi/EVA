import os
import json
import tarfile
import argparse

from io import BytesIO
from multiprocessing import Pool


def split_data_into_batches(data, batch_size):
    """将数据列表拆分为指定大小的多个批次"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def add_files_to_tar(tar_path, batch_data):
    data, json_name = batch_data  # Unpack the batch data and json name
    with tarfile.open(tar_path, "w") as tar:
        for item in tqdm(data):
            image_path = item.get("image", "")
            caption = item.get("caption", "")
            image_basename = os.path.basename(image_path)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file does not exist - {image_path}")
                continue
            
            base_name = os.path.splitext(image_basename)[0].replace(".", "_")
            extension = os.path.splitext(image_basename)[1]
            jpg_path = f"{base_name}{extension}"
            text_path = f"{base_name}.txt"

            tar.add(image_path, arcname=jpg_path)
            
            txt_data = BytesIO(caption.encode("utf-8"))
            tar_info = tarfile.TarInfo(name=text_path)
            tar_info.size = len(txt_data.getvalue())
            tar.addfile(tarinfo=tar_info, fileobj=txt_data)

def process_batch(i, batch, json_name):
    print(f"Processing batch {i}...")
    tar_dir = os.path.join('/mnt/pfs-mc0p4k/cv/team/panxuhao/wds_utils/laioncn/tarfiles', json_name)
    os.makedirs(tar_dir, exist_ok=True)
    
    tar_path = os.path.join(tar_dir, f'{str(i).zfill(5)}.tar')
    add_files_to_tar(tar_path, (batch, json_name))



# Use multiprocessing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--json-name', type=str, default="")
    args = parser.parse_args()
    json_name = args.json_name
        
    # Load data
    json_path = os.path.join('/mnt/pfs-mc0p4k/cv/team/panxuhao/data/laioncn/laioncnfilter', f'{json_name}.json')

    with open(json_path, "r") as f:
        data = json.load(f)

    batch_size = 100000  # Number of items in each tar file
    batches = list(split_data_into_batches(data, batch_size))
    print(f"Number of batches: {len(batches)}")

    with Pool(11) as pool:
        pool.starmap(process_batch, [(i, batch, json_name) for i, batch in enumerate(batches)])
