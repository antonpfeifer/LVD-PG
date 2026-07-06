using PyCall
py"""
import numpy as np
def f():
    return np.asarray(np.load('/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg/exps/progressive_growing/data/data_wikitext/data_trn.npy', mmap_mode='r')[:3])
"""
o=py"f"()
@show typeof(o)
for expr in [:(Array(o)), :(PyArray(o)), :(convert(Array,o))]
    try
        x = eval(expr)
        @show expr typeof(x) size(x) eltype(x)
    catch e
        @show expr e
    end
end
