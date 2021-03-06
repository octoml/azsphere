import shutil
import os
import subprocess
import tvm
import argparse
from azure import AzureSphere
import time

# build model and create imagepackage
def build(export_path, schedule_path, early_break=None):
    # get schedule files
    files = []
    if not os.path.exists(schedule_path):
        raise FileNotFoundError

    entries = os.listdir(schedule_path)
    entries.sort()

    for entry in entries:
        files.append(os.path.join(schedule_path, entry))

    ## build files
    target = tvm.target.create('llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib')
    # target = tvm.target.create('llvm -device=arm_cpu -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib')
    for ii in range(len(files)):
        if early_break and ii >= early_break:
            break
        tmp = AzureSphere(key=ii,
                          schedule_path=files[ii],
                          target=target)
        tmp.build()
        tmp.export(export_path)
        tmp.dependency(config_path='config', src_path='../')
        tmp.package()
        del tmp

def clear():
    process = subprocess.Popen("azsphere device sideload delete",
                shell = True,
                stdout = subprocess.PIPE,
                stderr=subprocess.PIPE)
    out, err = process.communicate()

def init():
    clear()

# run on azure sphere and save results
def run(task_path):
    init()

    files = []
    if not os.path.exists(task_path):
        raise FileNotFoundError

    tasks = os.listdir(task_path)
    tasks.sort()
    print(tasks)

    for item in tasks:
        path = 'build/' + item
        print("Task: " + str(item) + " Path: " + path)
        ##remove sideload
        process = subprocess.Popen("make -C " + path + " delete",
                shell = True,
                stdout = subprocess.PIPE,
                stderr=subprocess.PIPE)
        process.communicate()
        ##program
        process = subprocess.Popen("make -C " + path + " program",
                shell = True,
                stdout = subprocess.PIPE,
                stderr=subprocess.PIPE)
        process.communicate()

        # print("sleeping!")
        time.sleep(1)
        # print("Continue!")

    clear()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('-s', '--source', default='')
    opts = parser.parse_args()

    build_dir = 'build'
    
    if opts.build:
        if not opts.source:
            raise
        build(export_path=build_dir, 
            schedule_path=opts.source,
            early_break=None)

    if opts.run:
        run(task_path=build_dir)