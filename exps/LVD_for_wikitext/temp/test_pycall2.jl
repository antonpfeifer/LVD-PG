using PyCall
py"""
import numpy as np
def f():
    return np.asarray(np.load('/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg/exps/progressive_growing/data/data_wikitext/data_trn.npy', mmap_mode='r')[:3])
def g():
    return np.array(np.load('/scratch/tmp/mpfeife3/bachelorarbeit/lvd-pg/exps/progressive_growing/data/data_wikitext/data_trn.npy', mmap_mode='r')[:3], copy=True)
"""
for (name, fun) in [("f", py"f"), ("g", py"g")]
    println("--",name)
    o=fun()
    try x=PyArray(o); @show typeof(x) size(x) eltype(x) catch e @show e end
    try x=Array(PyArray(o)); @show typeof(x) size(x) eltype(x) catch e @show e end
    try x=pycall(fun, Array); @show typeof(x) size(x) eltype(x) catch e @show e end
    try x=pycall(fun, Array{Int32,2}); @show typeof(x) size(x) eltype(x) catch e @show e end
end
