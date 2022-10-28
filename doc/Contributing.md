<!--- SPDX-License-Identifier: Apache-2.0 -->


# Before Coding

The script folder has a `install_git_hook.sh` script. You need run it first to install the git hooks.

Currently, the git hooks will automatically perform the code format checking. Please make sure you have installed autopep8 and the cmd autopep8 is available in your environment.

# Code Style

We use autopep8 for checking code format.

In addition, we also follow the code format:

  * By default, we use `'` single quotes for string instead of `"` double quotes in Python files.


# Testing

The Parser uses [pytest](https://docs.pytest.org) as a test driver. To run tests, you'll first need to install pytest:

```
pip install pytest nbval
```
Also tests depend on onnx runtime:
```
# for onnx runtime
pip install onnxruntime==1.12
```

Some tests also depend AIPUBuilder for IR inference, so you need to install AIPUBuilder:
```
pip install AIPUBuilder-xx-xxx.whl
```

After installing pytest and AIPUBuilder, run from the `test` folder of the repo the tests:

```
cd test
pytest
```

For any new features, including new op support or new pass, relative test cases are required in the same commits.

# Small Incremental Changes

The project prefers  using small incremental changes. These changes can be small bug fixes, minor changes or some small steps for a large feature. Commits should be relatively small and well documented in commit messages or pull&request messages.

