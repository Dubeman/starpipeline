import subprocess
import re

def extract_coordinates(line):
    """
    Extracts the float coordinate pair from a line.

    Args:
        line (str): A line containing a coordinate pair in the format 'RA,Dec = (x,y)'

    Returns:
        tuple: A tuple containing the extracted RA and Dec values as floats.
    """
    pattern = r'\((\d+\.\d+),(\d+\.\d+)\)'
    match = re.search(pattern, line)
    if match:
        ra, dec = map(float, match.groups())
        return ra, dec
    else:
        return None


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

downsample = ["--downsample", "2"]

def build_scale(lower, upper, unit="arcsecperpix"):
    return ["--scale-units", unit, "--scale-low", str(lower), "--scale-high", str(upper)]

def build_limit(limit):
    return ["--cpulimit", str(limit)]

def build_cmd(image_path, args):
    cmd = base_cmd.copy() + downsample
    if args.get("lower_scale") and args.get("upper_scale"):
        lower = args.get("lower_scale")
        upper = args.get("upper_scale")
        cmd += build_scale(lower, upper)
    if args.get("limit"):
        limit = args.get("limit")
        cmd += build_limit(limit)
    #if args.get("downsample"):
    #    cmd += downsample
    if args.get("depth"):
        cmd += ["--depth", args.get("depth")]
    cmd.append(image_path)
    return cmd

def run(cmd):
    try:
        solve = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )

        for line in solve.stdout:
            if "RA,Dec = (" in line:
                coords = extract_coordinates(line)
                if coords is not None:
                    return [cmd[-1], coords]
        solve.wait()
    except subprocess.CalledProcessError:
        pass
    return [cmd[-1], "Failed"]
