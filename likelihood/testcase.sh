module purge all
module load python
module load mpi/openmpi-1.8.3-intel2015.0
source /projects/b1011/non-lsc/lscsoft-user-env.sh
source /projects/b1011/spinning_runs/bmu660/grid/development/my_builds/rapidpe_nu_build/etc/lscsoftrc
export PYTHONPATH=/home/bmu660/pycuda/install/lib/python2.7/site-packages:$PYTHONPATH
#export PYTHONPATH=/home/bmu660/all_cuda/pycuda_install/lib/python3.4/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/intel/composer_xe_2015.5.223/mkl/lib/intel64
module load cuda/cuda_7.5.18
module load intel/2015.0
module load gcc
unset CUDA_VISIBLE_DEVICES



#python -m cProfile -s cumtime ...
python /projects/b1011/spinning_runs/bmu660/grid/development/my_builds/rapidpe_nu_build/libexec/lalinference/rapidpe_integrate_extrinsic_likelihood.py --amp-order 0 --n-chunk 1000 --time-marginalization  --save-samples 1000 --reference-freq 0.0 --adapt-weight-exponent 0.8 --event-time 969631760.425028838 --save-P 0 --convergence-tests-on False --distance-maximum 300.0 --cache-file /projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/14631_data.cache --fmin-template 40.0 --skymap-file /projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/14631.toa_phoa_snr.fits.gz --n-max 10000 --save-deltalnL inf --l-max 2 --n-eff 1000 --mass1 1.4246297 --mass2 1.1844135 --output-file 2015_BNS_14631_LEVEL_1-0-0.xml.gz --coinc-xml /projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/coinc.xml.gz --approximant TaylorT4 --adapt-floor-level 0.1 --psd-file H1=/projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/H1_PSD_measured.xml.gz  --psd-file L1=/projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/L1_PSD_measured.xml.gz  --channel-name H1=FAKE-STRAIN  --channel-name L1=FAKE-STRAIN

#source /projects/b1011/ligo_project/lsc/rapidpe_nu_build/etc/lscsoftrc

#python /projects/b1011/ligo_project/lsc/rapidpe_nu_build/libexec/lalinference/rapidpe_integrate_extrinsic_likelihood.py --amp-order 0 --n-chunk 1000 --time-marginalization  --save-samples 1000 --reference-freq 0.0 --adapt-weight-exponent 0.8 --event-time 969631760.425028838 --save-P 0 --convergence-tests-on False --distance-maximum 300.0 --cache-file /projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/14631_data.cache --fmin-template 40.0 --skymap-file /projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/14631.toa_phoa_snr.fits.gz --n-max 1000000 --save-deltalnL inf --l-max 2 --n-eff 1000 --mass1 1.4246297 --mass2 1.1844135 --output-file 2015_BNS_14631_LEVEL_1-0-0.xml.gz --coinc-xml /projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/coinc.xml.gz --approximant TaylorT4 --adapt-floor-level 0.1 --psd-file H1=/projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/H1_PSD_measured.xml.gz  --psd-file L1=/projects/b1011/spinning_runs/bmu660/grid/coinc_id_14631/L1_PSD_measured.xml.gz  --channel-name H1=FAKE-STRAIN  --channel-name L1=FAKE-STRAIN
