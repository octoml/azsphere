import os
import subprocess
from subprocess import Popen, PIPE
import pdb
from pygdbmi.gdbcontroller import GdbController
from pprint import pprint

# process = subprocess.Popen(['arm-poky-linux-musleabi-gdb'],
#                      stdout=subprocess.PIPE, 
#                      stderr=subprocess.PIPE)
# stdout, stderr = process.communicate()
# print(stdout)
# print(stderr)

# subprocess.call('arm-poky-linux-musleabi-gdb', shell=True)
# stream = os.popen('arm-poky-linux-musleabi-gdb')
# output = stream.read()
# print(output)

# p = Popen(['arm-poky-linux-musleabi-gdb'], stdin=PIPE, stdout=PIPE, stderr=PIPE)   # set environment, start new shell
# stdoutdata, stderrdata = p.communicate("target remote 192.168.35.2:2345") # pass commands to the opened shell
# print(stdoutdata)
# print(stderrdata)

# connect to device
# process = subprocess.Popen(["sudo", "/opt/azurespheresdk/Tools/azsphere_connect.sh"])
# process.wait()

OUT_FILE="build/octoml_AS.out"
# IMG_FILE=~/octoml/azuresphere/out/Debug-4/HelloWorld_HighLevelApp.imagepackage
COMP_ID="1689d8b2-c835-2e27-27ad-e894d6d15fa9"

# app stop
process = subprocess.Popen(["azsphere", "device", "app", "stop"])
process.wait()

# # remove app
# azsphere device sideload delete

# # load app
# azsphere device sideload deploy --imagepackage $IMG_FILE

# # app stop
# azsphere device app stop

# app start debug
process = subprocess.Popen(["azsphere", "device", "app", "start", "--debug", "--componentid", COMP_ID])
process.wait()

# # run gdb
# arm-poky-linux-musleabi-gdb $OUT_FILE

# open telnet
# process = Popen(["telnet", "-a", "192.168.35.2", "2342"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# os.fork()
# stream = subprocess.call(["telnet", "-a", "192.168.35.2", "2342"])
# pdb.set_trace()

n = os.fork()
if (n > 0):
	print("Parent start")
	gdbmi = GdbController(gdb_path="/opt/azurespheresdk/Sysroots/4/tools/sysroots/x86_64-pokysdk-linux/usr/bin/arm-poky-linux-musleabi/arm-poky-linux-musleabi-gdb")
	response = gdbmi.write('-file-exec-file ' + 'build/octoml_AS.out')
	pprint(response)
	# response = gdbmi.write('b main')
	response = gdbmi.write('target remote 192.168.35.2:2345')
	pprint(response)
	response = gdbmi.write('b main')
	print(response)
	pdb.set_trace()
# 	p1 = Popen(['arm-poky-linux-musleabi-gdb', OUT_FILE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# 	# p1.stdin.write(b'continue\n')
# 	while True:
# 		p1out = p1.stdout.readline()
# 		print("Debugger: " + str(p1out))
# 		if "Reading symbols from" in str(p1out):
# 			print("Debugger: True")
# 		p1.stdin.write(b'target remote 192.168.35.2:2345\n')
# 			# p1.communicate(input=b'target remote 192.168.35.2:2345')
else:
	print("child start")
	p2 = Popen(["telnet", "-a", "192.168.35.2", "2342"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	print("child telnet open")
	# while True:
	# 	p2out = p2.stdout.readline()
	# 	print("Telnet: " + str(p2out))

