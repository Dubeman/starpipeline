import subprocess
import re
import os

base_cmd = [ "solve-field", "--overwrite", "--no-plots",
        "--new-fits", "none",
        "--solved", "none",
        "--match", "none",
        "--rdls", "none",
        "--index-xyls", "none",
        "--keep-xylist", "none",
        "--wcs", "none",
        "--corr", "none",
        "--temp-axy",
        "--uniformize", "0",
        "--no-remove-lines",
        "--no-background-subtraction",]

def build_scale(lower, upper, unit="arcsecperpix"):
    if lower is None or upper is None:
        return []
    return ["--scale-units", unit, "--scale-low", str(lower), "--scale-high", str(upper)]

def build_limit(limit):
    if limit is None:
        return []
    return ["--cpulimit", str(limit)]

def build_depth(depth):
    if depth is None:
        return []
    return ["--depth", str(depth)]

def build_downsample(downsample):
    if downsample is None:
        return []
    return ["--downsample", str(downsample)]

def build_cmd(image_path, args):
    cmd = base_cmd.copy()
    if os.name == 'nt':
        cmd.insert(0, "wsl")
    cmd += build_scale(args.get("lower_scale"), args.get("upper_scale"))
    cmd += build_limit(args.get("limit"))
    cmd += build_downsample(args.get("downsample"))
    cmd += build_depth(args.get("depth"))

    cmd.append(image_path) # always add image path last
    #for c in cmd:
    #    print(c, end=" ")
    return cmd

def run(cmd):
    try:
        solve = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        downsamples = 0
        for line in solve.stdout:
            if "Downsampling" in line:
                downsamples += 1

            if "log-odds" in line:
                print(line)
    
            if "arcsec/pix" in line:
                print(line)

            if "RA,Dec = (" in line:
                coords = extract_coordinates(line)
                if coords is not None:
                    print("Downsampled by ", downsamples * 2)
                    return [cmd[-1], coords]
        solve.wait()
    except subprocess.CalledProcessError:
        pass
    return [cmd[-1], "Failed"]

def extract_coordinates(line):
    """
    Expects line containing a coordinate pair in the format 'RA,Dec = (x,y)'
    Returns tuple containing the extracted RA and Dec values as floats.
    """
    pattern = r'\((\d+\.\d+),(\d+\.\d+)\)'
    match = re.search(pattern, line)
    if match:
        ra, dec = map(float, match.groups())
        return ra, dec
    else:
        return None