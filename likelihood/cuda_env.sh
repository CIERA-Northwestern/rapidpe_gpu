export PYTHONPATH=/home/bmu660/all_cuda/pycuda_install/lib/python3.4/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/intel/composer_xe_2015.5.223/mkl/lib/intel64
module load python/anaconda3
module load cuda
module load intel/2015.0
module load gcc
unset CUDA_VISIBLE_DEVICES
