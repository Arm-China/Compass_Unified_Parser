#!/usr/bin/env python3
import time
import os
import sys
import configparser
import subprocess
import filecmp
import datetime

mainpy = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../main.py"))


def compare(sample, ref):
    return filecmp.cmp(sample, ref)


def fast_parser_test(ref_ir, ref_w, cfg=None, log=None, bk_ref_ir=None):
    if cfg is None:
        cfg = "g.cfg"
    if log is None:
        log = sys.stdout
    else:
        log = open(log, "w")
        log.write(datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+"\n")
    config = configparser.ConfigParser()
    config.read(cfg)
    outdir = config["Common"]["output_dir"]
    if "model_type" in config["Common"]:
        framework = config["Common"]["model_type"]
    else:
        framework = "tf"
    model_name = config["Common"]["model_name"]
    name = framework + '_' + model_name
    out_ir = os.path.join(outdir, model_name+".txt")
    out_w = os.path.join(outdir, model_name+".bin")
    cmd = ["python3", mainpy, "-c", cfg]
    print("testing ", name)
    pid = subprocess.Popen(cmd, stdout=log, stderr=log)
    retcode = pid.wait()
    if retcode != 0:
        print("Parser error!! %s IR generate failed" % name)
        sys.exit(-1)
    wok = compare(out_w, ref_w)
    ok = compare(out_ir, ref_ir)
    if not ok and bk_ref_ir is not None:
        print("%s using backup ref!" % name)
        ok = compare(out_ir, bk_ref_ir)
    if not ok:
        print("Parser error!! %s IR changed!" % name)
        sys.exit(-1)
    if not wok:
        if model_name != "transformer_mini":
            print("Parser error!! %s Weight file changed!" % name)
            sys.exit(-1)
    sys.exit(0)


if __name__ == "__main__":
    fast_parser_test(sys.argv[1], sys.argv[2],
                     sys.argv[3], sys.argv[4], sys.argv[5])
