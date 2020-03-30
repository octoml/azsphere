#!/bin/bash

# connect to device
sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

OUT_FILE="~/octoml/azuresphere/out/Debug-4/HelloWorld_HighLevelApp.out"
IMG_FILE=~/octoml/azuresphere/out/Debug-4/HelloWorld_HighLevelApp.imagepackage
COMP_ID="1689d8b2-c835-2e27-27ad-e894d6d15fa9"

# app stop
azsphere device app stop

# remove app
azsphere device sideload delete

# load app
azsphere device sideload deploy --imagepackage $IMG_FILE

# app stop
azsphere device app stop

# app start debug
azsphere device app start --debug --componentid $COMP_ID

# run gdb
arm-poky-linux-musleabi-gdb $OUT_FILE

