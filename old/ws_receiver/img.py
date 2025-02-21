from PIL import Image
from pillow_heif import register_heif_opener
import os 
import numpy as np
import astropy.io.fits as fits
import sys

register_heif_opener()

input = sys.argv[1]
name, ext = os.path.splitext(input)

def convert_fits(file, outputFile):
    im = Image.open(file)
    a = np.array(im)
    if im.mode.startswith("RGB"):
        # split out the channels, flipping the y-axis
        r, g, b = [a[::-1,:,i] for i in [0, 1, 2]]   
        # Write each channel to a FITS file: XXX-red.fits, XXX-green.fits, XXX-blue.fits
        for chan, color in zip([r, g, b], ["red", "green", "blue"]):
            hdu =  fits.PrimaryHDU(chan)
            # Use the OBJECT keyword to describe this channel
            hdu.header['OBJECT'] = "%s channel" % (color)
            hdu.writeto("%s-%s.fits" % (name, color), overwrite=True)
    else:
        hdu =  fits.PrimaryHDU(a[::-1,:])
        hdu.writeto("%s.fits" % (name), overwrite=True)

# read in long exposure image files (.CR2 (Canon), .HEIC (iphone)) from "Camera" device
# Convert to .fits files, begin plate solve
# If plate solve is success, begin saving information: Estimated RA/DEC

# def convert_directory(dir): # List all files in directory, loop through all and convert
# def stream 

### Live (<30) target identification. 

convert_fits(input, name + ".fits")
print("Done")
