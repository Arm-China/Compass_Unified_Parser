import os
import sys

if __name__ == "__main__":
    raw_files = sys.argv[1]
    prefix = ".".join(raw_files.split(".")[:-1])
    out_files = prefix + ".h"
    string_name = sys.argv[2] if len(sys.argv) >= 3 else prefix.replace(".", "_")

    with open(raw_files, "r") as f:
        raw = f.read()

    raw = raw.replace("\n", "\\n")
    raw = raw.replace("\"", "\\\"")
    header_upper = f"_{string_name.upper()}_"
    content = '''// This file is auto generated, please do not modify
#ifndef _HEADER_UPPER_
#define _HEADER_UPPER_
#include <string>
    constexpr const char * _STRING_NAME_ = "_CONTENT_";
#endif
    '''
    content = content.replace("_CONTENT_", raw)
    content = content.replace("_STRING_NAME_", string_name)
    content = content.replace("_HEADER_UPPER_", header_upper)
    with open(out_files, "w") as f:
        f.write(content)
