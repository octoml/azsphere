import os
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', default='.')
    parser.add_argument('--port', default=9090)
    parser.add_argument('--ip', default='10.42.0.1')
    opts = parser.parse_args()

    cmd = 'python3 -m tvm.exec.rpc_server --tracker='+opts.ip+':'+opts.port + ' --key=npi'
    stream = os.popen(cmd)
    # process = subprocess.Popen(['python3', '-m', "tvm.exec.rpc_server", "--tracker="+opts.ip+":"+opts.port, "--key=npi"],
    #                  stdout=subprocess.PIPE, 
    #                  stderr=subprocess.PIPE)