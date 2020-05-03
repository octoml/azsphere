import shutil
import os
import subprocess
import tvm
from tvm import autotvm
import tvm.relay as relay 
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import argparse
from azure import AzureSphere
import time
import json
from npi import npi_schedule_convert as schedule_convert
import pickle

TARGET = tvm.target.create('llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib')
device_key = 'azure'
IP = '0.0.0.0'
PORT = 5040
tuning_option = {
    'tuner': 'gridsearch',
    'n_trial': 1000,
    'early_stopping': 1000,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='default'),
        # runner=autotvm.RPCRunner(
        #     device_key, host=IP, port=PORT,
        #     number=5,
        #     timeout=50,
        # ),
        runner=autotvm.LocalRunner(
            number=5, repeat=1, timeout=50,
        ),
    ),
}

TASK_IND = 3

def extract_schedule_cifar10(model=None, layer=0):
    from micro_eval.model.cifar10_cnn import gen_cifar10_cnn
    from tvm.autotvm.task.topi_integration import TaskExtractEnv
    TaskExtractEnv()
    # target = tvm.target.create('llvm -target=arm-poky-linux-musleabi -mcpu=cortex-a7 --system-lib')
    # target = tvm.target.create('llvm --system-lib')
    target = tvm.target.create('c -device=micro_dev')

    if model == 'cifar-10':
        template_key = ['direct_simd']
        template_key = ['direct']
        data_layout = 'NHWC'
        kernel_layout = 'HWOI'
        mod, params = gen_cifar10_cnn(
            data_layout, kernel_layout, op_strategy=template_key[0], use_random_params=True)

        # print(mod)
        # print(params)
        with tvm.target.build_config(opt_level=3, disable_vectorize=True):
            tasks = autotvm.task.extract_from_program(mod['main'], params, target)
        print(f'extracted {len(tasks)} tasks: {tasks}')
    else:
        raise ValueError("unkown model")


def task_to_schedule(task,
                     log_file,
                     measure_option,
                     tuner='xgb',
                     n_trial=1000,
                     early_stopping=None):
    #avoid appending
    if os.path.exists(log_file):
        os.remove(log_file)

    # create tuner
    if tuner == 'xgb' or tuner == 'xgb-rank':
        tuner_obj = XGBTuner(task, loss_type='rank')
    elif tuner == 'xgb_knob':
        tuner_obj = XGBTuner(task, loss_type='rank', feature_type='knob')
    elif tuner == 'ga':
        tuner_obj = GATuner(task, pop_size=50)
    elif tuner == 'random':
        tuner_obj = RandomTuner(task)
    elif tuner == 'gridsearch':
        tuner_obj = GridSearchTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)

    # do tuning
    task_trial = min(n_trial, len(task.config_space))
    print("INFO: config space size: " + str(len(task.config_space)))
    print("INFO: num of tasks: " + str(task_trial))

    tuner_obj.tune(n_trial=task_trial,
                    early_stopping=early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(task_trial),
                        autotvm.callback.log_to_file(log_file)
                    ])

    schedules = schedule_convert.load_log(log_file)
    return schedules

def generate_schedules(model=None, layer=0, schedule_path=None):
    from keras.models import load_model
    import keras

    if model == 'cifar-10':
        model = load_model('model/saved_models/cifar10_trained_model.h5')
        print(model.summary())
        print(model.inputs)

        shape_dict = {'conv2d_1_input': (1, 3, 32, 32)}
        mod, params = relay.frontend.from_keras(model, shape_dict)
        print(mod)
        
        # print(params)
        with tvm.target.build_config(opt_level=3, disable_vectorize=True):
            tasks = autotvm.task.extract_from_program(mod['main'], params, TARGET)
        print(f'extracted {len(tasks)} tasks: {tasks}')
    else:
        raise RuntimeError("model not specified")
    
    if not os.path.exists(schedule_path):
        os.makedirs(schedule_path)

    #dump tasks
    tasks_str = ''
    for task in tasks:
        tasks_str += f'{task}\n'

    with open(os.path.join(schedule_path, "tasks.txt"), 'w+') as f:
        f.write(tasks_str)

    with open(os.path.join(schedule_path, "tasks.dump"), 'wb+') as f:
        pickle.dump(tasks, f)

    for ii in range(len(tasks)):
        if ii != TASK_IND:
            continue
        task = tasks[ii]
        task_dir = "task_" + str(ii).zfill(4)
        if not os.path.exists(os.path.join(schedule_path, task_dir)):
            os.makedirs(os.path.join(schedule_path, task_dir))

        log_file = "task_" + str(ii) + ".txt"
        schedules = task_to_schedule(task=task, log_file=log_file, **tuning_option)
        # print(schedules)

        count = 0
        for item in schedules:
            schedule = "schedule_" + str(count).zfill(4)
            count += 1
            path = os.path.join(os.path.join(schedule_path, task_dir), schedule)
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.join(path, "schedule.txt")

            with open(filepath, 'w+') as f:
                tmp = json.dumps(item)
                f.write(tmp)

# build model and create imagepackage
def build(build_path, early_break=None):
    tasks_dirs = [dI for dI in os.listdir(build_path) if os.path.isdir(os.path.join(build_path, dI))]
    tasks_dirs.sort()

    with open(os.path.join(build_path, 'tasks.dump'), 'rb') as f:
        tasks = pickle.load(f)

    for jj in range(len(tasks_dirs)):
        if jj != TASK_IND:
            continue
        task_path = os.path.join(build_path, tasks_dirs[jj])
        list_schedules = os.listdir(task_path)
        list_schedules.sort()
        # print(list_schedules)

        for ii in range(len(list_schedules)):
            if early_break and ii >= early_break:
                break
            
            export_path = os.path.join(task_path, list_schedules[ii])
            schedule_path = os.path.join(export_path, 'schedule.txt')
            # print(schedule_path)
            tmp = AzureSphere(key=ii,
                            task=tasks[jj],
                            schedule_path=schedule_path,
                            target=TARGET)
            tmp.build()
            tmp.export(export_path)
            params = {'main': f'cifar_task{jj}.c',
                      'approot_files': ['build/conv2d_data.bin', 'build/conv2d_output.bin',
                                        'build/conv2d_params.bin', 'build/conv2d_graph.bin', 
                                        'build/id.bin'],
                      'src_files': [f'cifar_task{jj}.c', 'bundle_static.c', 'runtime.c']
                    }
            tmp.dependency(config_path=os.path.join('config', 'cifar10'), 
                           src_path='../',
                           params=params)
            tmp.package()
            del tmp
            break
        break

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
        path = os.path.join(task_path, item)
        print(f'Task: {str(item)}\tPath: {path}')

        current_dir = os.getcwd()
        ##move to directory
        os.chdir(path)
        ##remove sideload
        process = subprocess.Popen("make delete",
                shell = True,
                stdout = subprocess.PIPE,
                stderr=subprocess.PIPE)
        process.communicate()
        ##program
        process = subprocess.Popen("make program",
                shell = True,
                stdout = subprocess.PIPE,
                stderr=subprocess.PIPE)
        process.communicate()
        
        ##move to current directory
        os.chdir(current_dir)

        # print("sleeping!")
        time.sleep(1)
        # print("Continue!")
        break

    clear()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', action='store_true')
    parser.add_argument('-b', '--build', action='store_true')
    parser.add_argument('-r', '--run', action='store_true')
    parser.add_argument('-s', '--source', default='')
    opts = parser.parse_args()

    build_dir = 'build'
    
    if opts.generate:
        generate_schedules(model='cifar-10', layer=0, schedule_path=build_dir)
    
    if opts.build:
        build(build_path=build_dir)

    if opts.run:
        task = f'task_000{TASK_IND}'
        print(f'running task: {task}')
        run(task_path=os.path.join(build_dir, task))

    # if opts.build:
    #     if not opts.source:
    #         raise
    #     build(export_path=build_dir, 
    #         schedule_path=opts.source,
    #         early_break=None)

    # if opts.run:
    #     run(task_path=build_dir)