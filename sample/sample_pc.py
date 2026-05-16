import os
import sys
import numpy as np
from PIL import Image

# Initialize julia from python-jl
from julia import Main as JL

print("Initializing Julia environment...")
JL.eval('using Pkg; Pkg.activate(".")')
JL.include("VariationalJuice.jl/src/VariationalJuice.jl")
JL.include("VariationalJuice.jl/src-jl/LatentPCs.jl")
JL.eval("using ProbabilisticCircuits")


print("Loading top-level PC and setting up sampling...")
JL.include("sample.jl")

print("Generating sample...")
img_rgb = JL.generate_image()

# The result is 1 x 3 x 32 x 32
img_rgb = img_rgb[0]
img_rgb = np.transpose(img_rgb, (1, 2, 0)) # 32 x 32 x 3
img_rgb = img_rgb.astype(np.uint8)

img = Image.fromarray(img_rgb)
img.save("sample_output.png")
print("Saved to sample_output.png")
