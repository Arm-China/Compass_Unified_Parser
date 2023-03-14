#!/bin/bash
source /arm/tools/setup/init/bash
module load swdev python/python/3.8.5
module load swdev git/git/2.17.1

export LD_LIBRARY_PATH="/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="`realpath ../`:${PYTHONPATH}"

# suppose you allready built the aipubuilder
git log -n 1 > temp.log

# suppose Compass_OpportunePostTrainingTools has been added in ../../
# add soft link to opt code
ln -s ../../Compass_OpportunePostTrainingTools/AIPUBuilder/Optimizer ../AIPUBuilder/

failed_op_tests=""
failed_pass_tests=""
failed_change_tests=""
failed_specfied_tests=""

# 1 If you commit like: [Parser_op] ...
if grep -i '\[Parser_op\]' temp.log
then
    for f in `find ./op_test -type f -name '*.py'`
    do
        echo "SANITY TESTING: "$f
        python3 $f
        exit_code=$?
        if [[ $exit_code != 0 ]]
        then
            failed_op_tests=$failed_op_tests" "$f
        fi
    done
fi

# 2 If you commit like: [Parser_pass] ...
if grep -i '\[Parser_pass\]' temp.log
then
    for f in `find ./pass_test -type f -name '*.py'`
    do
        echo "SANITY TESTING: "$f
        python3 $f
        exit_code=$?
        if [[ $exit_code != 0 ]]
        then
            failed_pass_tests=$failed_pass_tests" "$f
        fi
    done
fi

# 3 If you change files inside this folder
filelist="`git diff-tree --no-commit-id --diff-filter=ACM --name-only -r HEAD ./ | grep .py | grep -vw 'plugins' | egrep 'op_test|pass_test|plugin_test'`" || true
if [[ -n $filelist ]]
then
    for f in $filelist
    do
        f_without_prefix=${f#"test/"}
        echo "SANITY TESTING: "$f_without_prefix
        python3 $f_without_prefix
        exit_code=$?
        if [[ $exit_code != 0 ]]
        then
            failed_change_tests=$failed_change_tests" "$f_without_prefix
        fi
    done
fi

# 4 If you provide keyword in the beginning of commit like: [tf2] ...
#   If there are multiple keywords, separate them with comma, like [tf2,onnx]
keyword="`git log --oneline -n 1 --format=%s | xargs | grep -o '^\[.*\]'`"
if [[ -n $keyword ]]
then
    keyword=${keyword/\]*/\]} # Remove the back part of ] if multiple [] are matched
    keyword=${keyword:1:-1} # Remove '[' and ']'
    echo "Find keyword [${keyword}] in commit message!"
    lc_keywords=${keyword,,}
    lc_keywords=`echo ${lc_keywords} | sed -e 's/,/ /g'`
    for lc_keyword in ${lc_keywords}
    do
        if [[ $lc_keyword == 'parser_op' || $lc_keyword == 'parser_pass' ]]
        then
            continue
        fi
        echo "Find tests for keyword ${lc_keyword}..."
        filelist=`find ./*_test -iname "*${lc_keyword}*" -type f | grep -vw plugins`
        if [[ -n $filelist ]]
        then
            for file in $filelist
            do
                if [[ $file != *".py" ]]
                then
                    continue
                fi
                echo "SANITY TESTING: "$file
                python3 $file
                exit_code=$?
                if [[ $exit_code != 0 ]]
                then
                    failed_specfied_tests=$failed_specfied_tests" "$file
                fi
            done
        fi
    done
fi

# unlink opt after finishing tests
unlink ../AIPUBuilder/Optimizer

all_passed=1
if [[ -n $failed_op_tests ]]
then
    echo "Failed op_tests:$failed_op_tests"
    all_passed=0
fi

if [[ -n $failed_pass_tests ]]
then
    echo "Failed pass_tests:$failed_pass_tests"
    all_passed=0
fi

if [[ -n $failed_change_tests ]]
then
    echo "Failed added/modified tests:$failed_change_tests"
    all_passed=0
fi

if [[ -n $failed_specfied_tests ]]
then
    echo "Failed specfied tests:$failed_specfied_tests"
    all_passed=0
fi

if [[ $all_passed == 1 ]]
then
    echo "ALL PASSED"
else
    exit 1
fi
