<!--- SPDX-License-Identifier: Apache-2.0 -->


# Before Coding

The script folder has a `install_git_hook.sh` script, you need run it first to install the git hooks.

Currently, the git hooks will do automatically the code format checking. Please make sure you have install autopep8 and the cmd autopep8 is available in your environment.

# Code style

We use autopep8 for checking code format.

Beside that, we also follow the code format:

  * By default, we use `'` single quotes for string not `"` double quotes in python files.


# Testing

CUP uses [pytest](https://docs.pytest.org) as a test driver. To run tests, you'll first need to install pytest:

```
pip install pytest nbval
```

Some tests also depends AIPUBuilder for IR inference, so you need to install AIPUBuilder:
```
pip install AIPUBuilder-xx-xxx.whl
```

After installing pytest and AIPUBuilder, run from the `test` folder of the repo:

```
cd test
pytest
```

to begin the tests.

For any new features, including new op support or new pass, relative test cases are required in same commits.

# Small incremental changes

The project prefers  using small incremental changes. These changes can be small bug fixes or minor changes or some small steps for a large features. Commits should be relative small and well documented in commit message or pull&request message.

