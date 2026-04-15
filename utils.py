"""
The MIT License

Copyright (c) 2026 Samsung SDS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import time
import sys
import os
from datetime import datetime
import shutil
from pynvml import *
import logging

process_start_time = datetime.now()
result_folder = './result/' + '{desc}' + process_start_time.strftime("%Y%m%d_%H%M%S")


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None, level=logging.INFO):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc=log_file['desc']+'_' )
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=level)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(level)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


def create_simple_logger(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level=level)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)



class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('root')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        if (count - self.count_zero)==0:
            return 0, 0
            
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("[{}/{}]: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def copy_src(dstroot):
    def copy_fs(srcdir, dstdir):
        os.makedirs(dstdir, exist_ok=True)                    
        for f in os.listdir(srcdir):
            if not f in exception and not 'result' in f:
                srcpath = os.path.join(srcdir, f)
                if os.path.isfile(srcpath) and f.endswith(copy_ext):
                    shutil.copy(srcpath, dstdir)
                elif os.path.isdir(srcpath):
                    copy_fs(srcpath, os.path.join(dstdir, f))

    exception = ['trash',
                 'result', 
                 'final_result', 
                 '__pycache__', 
                 'nohup.out', 
                 'temp', 'tmp', 
                 '.ipynb_checkpoints', 
                 'output.log', 
                 os.path.basename(dstroot)]
                 
    copy_ext = ('.py', '.sh', '.json')
        
    dstpath = os.path.join(dstroot, 'src')
    curr_dir = os.getcwd()
    copy_fs(curr_dir, dstpath)



def print_gpuinfo():
    logger = logging.getLogger()

    nvmlInit()
     
    deviceCount = nvmlDeviceGetCount()
    for idx in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(idx)
        dev_name = nvmlDeviceGetName(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_info = nvmlDeviceGetUtilizationRates(handle)
         
        logger.info(f"GPU {idx} - "+\
                    f"{dev_name} | "+\
                    f"Memory total: {mem_info.total/1024**3:.3f}GB, "+\
                    f"used: {mem_info.used/1024**3:.3f}GB., "+\
                    f"free: {mem_info.free/1024**3:.3f}GB. | "+\
                    f"Utilization gpu: {util_info.gpu:.2f}, "+\
                    f"Utilization memory: {util_info.memory:.2f}")
 

def get_num_device():
    nvmlInit()     
    deviceCount = nvmlDeviceGetCount()

    return deviceCount


def get_device_name():
    nvmlInit()
     
    handle = nvmlDeviceGetHandleByIndex(0)
    dev_name = nvmlDeviceGetName(handle)

    return dev_name
    

