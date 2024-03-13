# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import os
import signal
import subprocess
import shlex
import threading
from collections import OrderedDict
import time


class JobManager(object):

    def __init__(self, max_jobs=1024, max_running=15, timeout=1):
        # a job is a dict
        self.start_time = time.time()
        self.max_running = max_running
        self.jobs = OrderedDict()
        self.running_jobs = []
        self.finished = []
        self.passed = []
        self.failed = []
        self.timeout = timeout
        self.__stop__ = False
        self.mt = threading.Thread(target=self.monitoring)
        # self.mt.join()

    @property
    def pass_num(self):
        return len(self.passed)

    @property
    def fail_num(self):
        return len(self.failed)

    def add_job(self, name, cmd, timeout=-1):
        self.jobs[name] = (cmd, timeout)

    def _kill_all_jobs(self):
        for job in self.running_jobs:
            pid = job["pid"]
            print("killing pid:%d" % (pid.pid))
            os.killpg(os.getpgid(pid.pid), signal.SIGTERM)

    def monitoring(self):
        no_job_time = 0
        while True:
            try:
                if self.__stop__:
                    self._kill_all_jobs()
                    break
                for job_dict in self.running_jobs:
                    job = job_dict["pid"]
                    timeout = job_dict["timeout"]
                    t0 = job_dict["start_time"]
                    ret = job.poll()
                    if ret is not None:
                        self.finished.append(job_dict)
                        self.running_jobs.remove(job_dict)
                        job_dict["cost_time"] = time.time() - t0
                        if ret == 0:
                            self.passed.append(job_dict)
                            result = "(pass)"
                        else:
                            self.failed.append(job_dict)
                            result = "(fail)"
                        # print("%s finish , cost_time = %3.0fs %s" \
                        #         %(job.name.ljust(25), job_dict["cost_time"], result))
                    elif timeout > 0 and time.time()-t0 > timeout:
                        job_dict["cost_time"] = time.time() - t0
                        # print("%s timeout, cost_time = %3.0fs %s" \
                        #         %(job.name.ljust(25), job_dict["cost_time"], "(timeout)"))
                        os.killpg(os.getpgid(job.pid), signal.SIGTERM)
                        self.failed.append(job_dict)
                        self.running_jobs.remove(job_dict)
                if len(self.jobs) > 0 and len(self.running_jobs) < self.max_running:
                    name = list(self.jobs.keys())[0]
                    cmd, timeout = self.jobs.pop(name)
                    cmd = shlex.split(cmd)
                    # print("start job:",name)
                    pid = subprocess.Popen(cmd, preexec_fn=os.setsid)
                    # pid=subprocess.Popen(["tcsh","-c",cmd])
                    pid.name = name
                    running_job = dict()
                    running_job["name"] = name
                    running_job["pid"] = pid
                    running_job["timeout"] = timeout
                    running_job["start_time"] = time.time()
                    self.running_jobs.append(running_job)

                if len(self.running_jobs) == 0:
                    if no_job_time == 0:
                        no_job_time = time.time()
                    if time.time()-no_job_time > self.timeout:
                        break
                elif no_job_time != 0:
                    no_job_time = 0
                time.sleep(0.5)
            except (Exception, KeyboardInterrupt) as e:
                self._kill_all_jobs()
                raise e

    def stop(self):
        self.__stop__ = True
        print("Waiting monitoring stop")
        while self.mt.isAlive():
            time.sleep(0.1)
        for job in self.running_jobs:
            job.kill()


if __name__ == "__main__":
    jm = JobManager()
    job = "aipurun -v"
    for i in range(5):
        jm.add_job(i, job)
