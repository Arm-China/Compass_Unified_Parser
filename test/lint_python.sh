#!/bin/bash -e

source /arm/tools/setup/init/bash
module load swdev git/git/2.17.1
module load swdev gnu/cmake/3.14.3
module load swdev gnu/gcc/7.3.0

cd ../

filelist=`git diff-tree --no-commit-id --diff-filter=ACM --name-only -r HEAD | grep ".py$" | tr "\n" " "`

echo $filelist

if [ -z "$filelist" ]; then
    echo "no python(.py) file change"
else
    scripts/git_hooks/pre-commit.d/format_pyfile "$filelist"
fi
