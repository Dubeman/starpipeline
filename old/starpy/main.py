import os
import cv2
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from astrometry_handler import run, build_cmd
#iphone_scale = {"lower_scale": "65", "upper_scale": "70"}
def numerical_sort(filename):
    return int(os.path.splitext(filename)[0]) 

images = "./DeepSpaceYoloDataset/images/"
image_files = [f for f in os.listdir(images) if f.lower().endswith(('.jpg'))]
image_files.sort(key=numerical_sort)

sample_size = 20

image_files = image_files[:sample_size].copy()

#images = "./testdata"
#image_files = [f for f in os.listdir(images) if f.lower().endswith(('.png'))]
#image_files.sort()

args = {"limit": str(2), "depth": str(7)}

def batch():
    results = []
    solved = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        with tqdm(total=len(image_files), desc="Processing") as pbar:
            futures = {executor.submit(run, build_cmd(os.path.join(images, img), args)): img for img in image_files}

            for future in as_completed(futures):
                result = future.result()
                img = result[0]
                res = result[1]
                if res != "Failed":
                    solved += 1
                    name = img[img.rfind("/")+1:]
                    image_files.remove(name)
                    results.append(result)
                
                pbar.update(1)
                pbar.set_description(f"Solved {solved} images")
    return results

attempts = 0
MAX_ATTEMPTS = 5
full_results = []
while attempts < MAX_ATTEMPTS:
    attempts += 1
    print(f"Attempting {len(image_files)} images with limit {args.get('limit')} and d={args.get('depth')}\n")
    res = batch()
    full_results.append(res)
    if len(image_files) == 0:
        break
    args["limit"] = str(int(args.get("limit")) + 2)
    
    if attempts % 2 == 0:
        args["depth"] = str(int(args.get("depth")) + 1)
    
    if attempts == 3:
        print("Enabling downsample")
        args["downsample"] = "2"

    time.sleep(1)

print(f"Solved {len(full_results)}\n")
print(full_results)