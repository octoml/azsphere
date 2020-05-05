from pygdbmi.gdbcontroller import GdbController
from pprint import pprint

gdbmi = GdbController(gdb_path="/opt/azurespheresdk/Sysroots/4/tools/sysroots/x86_64-pokysdk-linux/usr/bin/arm-poky-linux-musleabi/arm-poky-linux-musleabi-gdb")
response = gdbmi.write('-file-exec-file ' + 'build/octoml_AS.out')
pprint(response)
# response = gdbmi.write('b main')
response = gdbmi.write('target remote 192.168.35.2:2345')
pprint(response)
response = gdbmi.write('continue')
print(response)
# response = gdbmi.write('mon reset 0')
# response = gdbmi.write('c')