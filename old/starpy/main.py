import os
import cv2
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from astrometry_handler import run, build_cmd

#iphone_scale = {"lower_scale": "65", "upper_scale": "70"}
#deep_space_yolo_scale = {"lower_scale": "2", "upper_scale": "10"}
rpi_scale = {"lower_scale": "19", "upper_scale": "21"}

def numerical_sort(filename):
    return int(os.path.splitext(filename)[0]) 

images = "/Users/owen/tempdata/quickdata_focustest"
image_files = [f for f in os.listdir(images) if f.lower().endswith(('.jpg'))]
#image_files.sort(key=numerical_sort)

sample_size = len(image_files)

image_files = image_files[:sample_size].copy()

#images = "./testdata"
#image_files = [f for f in os.listdir(images) if f.lower().endswith(('.png'))]
#image_files.sort()

args = {"limit": "10", "downsample": 8} #, "lower_scale": "2", "upper_scale": "10"}
args.update(rpi_scale)

# Optimizing Plate Solving!!!

# Optimize :

# Find optimize downsample rate for camera
# Find optimal arcsecperpix for camera

# Find optimial gain and exposure for camera, depending on ambient/light pollution


# For:
# - SOLVABILITY
# - Higher SNR
# - Lower downsample rate 
# - 

# - Lower exposure, faster response time
# - Higher exposure, better quality (SNR)


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
MAX_ATTEMPTS = 2
full_results = []
while attempts < MAX_ATTEMPTS:
    attempts += 1
    print(f"Attempting {len(image_files)} images with limit {args.get('limit')}\n")
    res = batch()
    full_results.append(res)
    if len(image_files) == 0:
        break
    args["limit"] = str(int(args.get("limit")) + 2)
    
    if attempts == 3:
        print("Enabling downsample")
        args["downsample"] = "2"

    time.sleep(1)

# Move the solved images to a new directory
solved_dir = "./solvable"

#for result in full_results:
#    for img in result:
#        name = img[0][img[0].rfind("/")+1:]
#        if os.path.exists(os.path.join(images, name)):
#            os.makedirs(solved_dir, exist_ok=True)
#            os.rename(os.path.join(images, name), os.path.join(solved_dir, name))