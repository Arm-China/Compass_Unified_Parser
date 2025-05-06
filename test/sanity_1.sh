#!/bin/bash -e
wk_dir=`pwd`
echo ${wk_dir}
ls
source /arm/tools/setup/init/bash
#module load swdev python/python/3.8.5
module load swdev python/conda/5.2.0
source /arm/tools/python/conda/5.2.0/rhe7-x86_64/etc/profile.d/conda.sh
conda activate py_3.10.12

export LD_LIBRARY_PATH=/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH
cd sanity
python3 ./sanity_test.py
