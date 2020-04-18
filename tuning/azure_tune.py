import shutil
import os
import tvm

from azure import AzureSphere

# build model and create imagepackage
def build(export_path, schedule_path):
    # get schedule files
    files = []
    if not os.path.exists(schedule_path):
        raise FileNotFoundError

    entries = os.listdir(schedule_path)
    entries.sort()

    for entry in entries:
        files.append(os.path.join(schedule_path, entry))

    # build files
    as_instances = []
    target = tvm.target.create('llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib')
    # target = tvm.target.create('llvm -device=arm_cpu -target=arm-linux-gnueabihf')
    for ii in range(len(files)):
        tmp = AzureSphere(key=ii,
                          schedule_path=files[ii],
                          target=target)
        tmp.build()
        tmp.export(export_path)
        tmp.dependency(config_path="config", src_path="src")
        as_instances.append(tmp)
    return as_instances

# run on azure sphere and save results
def run():
    return 0

if __name__ == '__main__':
    print("main")
    items = build(export_path="lib", 
          schedule_path="/home/parallels/tvm/apps/npi/history")
    for ii in range(len(items)):
        items[ii].package()