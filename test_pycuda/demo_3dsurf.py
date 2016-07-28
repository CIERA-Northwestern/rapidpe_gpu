#!/usr/bin/env python

#
# Author: Ezequiel Alfie <ealfie@gmail.com>
#
# demonstrating simultaneous use of 3D textures and surfaces
#
#
#
# needs CUDA 4.x and pycuda with v4 launch interface
# (later than commit dd12c742c6ea35cd06ce25fd17abf21c01cd6ff7 Apr 21, 2012)
#

from __future__ import division
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import pycuda.autoinit
import numpy.testing



def array_format_to_dtype(af):
    if af == drv.array_format.UNSIGNED_INT8:
        return np.uint8
    elif af == drv.array_format.UNSIGNED_INT16:
        return np.uint16
    elif af == drv.array_format.UNSIGNED_INT32:
        return np.uint32
    elif af == drv.array_format.SIGNED_INT8:
        return np.int8
    elif af == drv.array_format.SIGNED_INT16:
        return np.int16
    elif af == drv.array_format.SIGNED_INT32:
        return np.int32
    elif af == drv.array_format.FLOAT:
        return np.float32
    else:
        raise TypeError(
                "cannot convert array_format '%s' to a numpy dtype"
                % array_format)

#
# numpy3d_to_array
# this function was
# taken from pycuda mailing list (striped for C ordering only)
#
def numpy3d_to_array(np_array, allow_surface_bind=True):

    import pycuda.autoinit

    d, h, w = np_array.shape

    descr = drv.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = drv.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    if allow_surface_bind:
        descr.flags = drv.array3d_flags.SURFACE_LDST

    device_array = drv.Array(descr)

    copy = drv.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return device_array


def array_to_numpy3d(cuda_array):

    import pycuda.autoinit

    descriptor = cuda_array.get_descriptor_3d()

    w = descriptor.width
    h = descriptor.height
    d = descriptor.depth

    shape = d, h, w

    dtype = array_format_to_dtype(descriptor.format)

    numpy_array=np.zeros(shape, dtype)

    copy = drv.Memcpy3D()
    copy.set_src_array(cuda_array)
    copy.set_dst_host(numpy_array)

    itemsize = numpy_array.dtype.itemsize

    copy.width_in_bytes = copy.dst_pitch = w*itemsize
    copy.dst_height = copy.height = h
    copy.depth = d

    copy()

    return numpy_array


src_module=r'''
#include <stdint.h>
#include <cuda.h>
#include <surface_functions.h>

texture<float, cudaTextureType3D, cudaReadModeElementType> tex_in;
surface<void, 3> surf_out;

__global__ void test_3d_surf(int32_t Nz, int32_t Ny, int32_t Nx)
{

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  if (x < Nx && y < Ny && z < Nz) {
    float value = tex3D(tex_in, (float) x, (float) y, float (z));

    surf3Dwrite((float) value, surf_out, sizeof(float) * x, y, z, cudaBoundaryModeZero);
  }

}
'''

mod=SourceModule(src_module, cache_dir=False, keep=False)

kernel=mod.get_function("test_3d_surf")
arg_types = (np.int32, np.int32, np.int32)

tex_in=mod.get_texref('tex_in')
surf_out=mod.get_surfref('surf_out')

# random shape
shape_x = np.random.randint(1,255)
shape_y = np.random.randint(1,255)
shape_z = np.random.randint(1,255)

dtype=np.float32 # should match src_module's datatype

numpy_array_in=np.random.randn(shape_z, shape_y, shape_x).astype(dtype).copy()
cuda_array_in = numpy3d_to_array(numpy_array_in)
tex_in.set_array(cuda_array_in)

zeros=np.zeros_like(numpy_array_in)
cuda_array_out = numpy3d_to_array(zeros,allow_surface_bind=True)
surf_out.set_array(cuda_array_out)


block_size_z, block_size_y, block_size_x = 8,8,8 #hardcoded, tune to your needs
gridz = shape_z // block_size_z + 1 * (shape_z % block_size_z != 0)
gridy = shape_y // block_size_y + 1 * (shape_y % block_size_y != 0)
gridx = shape_x // block_size_x + 1 * (shape_x % block_size_x != 0)
grid = (gridx, gridy, gridz)
block = (block_size_x, block_size_y, block_size_x)

kernel.prepare(arg_types,texrefs=[tex_in])
kernel.prepared_call(grid, block, shape_z, shape_y, shape_x)

numpy_array_out = array_to_numpy3d(cuda_array_out)
numpy.testing.assert_array_almost_equal(numpy_array_out, numpy_array_in)
