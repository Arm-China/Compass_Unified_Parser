#!/bin/bash -e
source /arm/tools/setup/init/bash
module load swdev python/python/3.8.5
module load swdev git/git/2.17.1

export LD_LIBRARY_PATH="/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="`realpath ../`:${PYTHONPATH}"

# suppose you allready built the aipubuilder
git log -n 1 > temp.log

# 1 If you commit like: [Parser_op] ...
if grep -i Parser_op temp.log
then
    for f in `find ./op_test -type f -name '*.py'`
    do
        echo "SANITY TESTING: "$f
        python3 $f
    done
fi

# 2 If you commit like: [Parser_pass] ...
if grep -i Parser_pass temp.log
then
    for f in `find ./pass_test -type f -name '*.py'`
    do
        echo "SANITY TESTING: "$f
        python3 $f
    done
fi

# 3 If you change files inside this folder
filelist="`git diff-tree --no-commit-id --diff-filter=ACM --name-only -r HEAD ./ | grep .py`" || true
if [ -n $filelist ];
then
    for f in $filelist
    do
        f_without_prefix=${f#"python/test/parser/"}
        echo "SANITY TESTING: "$f_without_prefix
        python3 $f_without_prefix
    done
fi

echo "ALL PASSED"
