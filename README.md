## Overview
We show machine learning model deployment on [MT3620 Azure Sphere](https://azure.microsoft.com/en-us/services/azure-sphere/get-started/) using [Apache TVM](https://tvm.apache.org/). We show multiple deployments from a simple ```a + b``` example to a ```Conv2D``` operation and finally we deploy [Keyword Spotting](https://github.com/ARM-software/ML-KWS-for-MCU) model developed by ARM.

## Hardware Requirements
- Linux machine
- [MT3620 Azure Sphere board](https://www.seeedstudio.com/Azure-Sphere-MT3620-Development-Kit-US-Version-p-3052.html)
- micro USB cable
- [MT3620 Ethernet Shield](https://www.seeedstudio.com/MT3620-Ethernet-Shield-v1-0-p-2917.html) (only for tuning)

## Software Requirements
- Linux with Azure Sphere SDK (follow [Azure Sphere documentation](https://docs.microsoft.com/en-us/azure-sphere/) to setup SDK and device)
- Python 3.6+
- Tensorflow

## Getting Started
1. Clone this repository
2. [Install TVM](https://docs.tvm.ai/install/from_source.html)
   - **NOTE:** Ensure you enable LLVM by setting ```set(USE_LLVM ON)```. (This repository has been tested against LLVM-10)
   - **NOTE:** Checkout ```f5b02fdb1b5a7b6be79df97035ec1c3b80e3c665``` before installation.
3. Setup virtualenv
```bash
$ python3 -mvenv _venv
$ . _venv/bin/activate
$ pip3 install -r requirements.txt -c constraints.txt
$ export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

## Prepare the Hardware
1. Connect Azure Sphere board to PC with micro USB cable.
2. In the current directory run ```make connect``` to connect to device. (This requires ```sudo``` access)
3. Enable evelopment by running ```make enable_development``` command.
4. Optional: Follow this to enable network capability:
   - Disconnect the device and attach the network shield.
   - Setup static IP
   ```bash
   Netmask XXX
   Gateway XXX
   IP address XXXX
   ```

## Run Samples
The basic sample is ```a + b``` operation. In this example, we deploy a simple operation on Azure Sphere using [C Runtime](https://github.com/apache/incubator-tvm/tree/master/src/runtime/crt) from TVM. To deploy this follow these instructions:
```bash
$ make delete_a7
$ make clean
$ make test
$ make program
```
After programming the Azure Sphere, it reads TVM graph and parameters from FLASH and creates the runtime. Then it will read input data from FLASH, pass it to the TVM Relay model and finally compares the output with expected output from X86 machine. If the result maches, LED1 on the Azure Sphere would change to green.

Next sample is ```Conv2D``` operation. To run this example, follow previous instructions and use ```conv2d``` instead of ```test```.

## Prepare the environment

1. Ensure that your Azure Sphere device is connected to your computer and your computer is connected to the internet.
1. Even if you've performed this setup previously, ensure that you have Azure Sphere SDK version 20.01 or above. At the command prompt, run **azsphere show-version** to check. Install the Azure Sphere SDK for [Windows](https://docs.microsoft.com/azure-sphere/install/install-sdk) or [Linux](https://docs.microsoft.com/azure-sphere/install/install-sdk-linux) as needed.
1. Enable application development, if you have not already done so, by entering the following line at the command prompt:

   `azsphere device enable-development`

1. Clone this repository and TVM repository in home directory:
```bash
cd ~
clone git@github.com:octoml/azure-sphere.git
clone git clone --recursive https://github.com/apache/incubator-tvm.git tvm
```

## Build and run the sample
Here I explain main functionalities of the Makefile.

### Connecting to board
To do this, run command below and this will connect to azure sphere device:
```bash
make connect
```

### Debugging
Before debugging in VS, run this command. This command will make sure that 
current side application on AS devicde is removed.
```bash
make debug_init
```
### Build Image Package
Run this command to build image package and executable file.
```bash
make build/imagepackage
```

## Additional Resources
See the following Azure Sphere Quickstarts to learn how to build and deploy on AS:
-  [with Visual Studio](https://docs.microsoft.com/azure-sphere/install/qs-blink-application)
-  [with VS Code](https://docs.microsoft.com/azure-sphere/install/qs-blink-vscode)
-  [on the Windows command line](https://docs.microsoft.com/azure-sphere/install/qs-blink-cli)
-  [on the Linux command line](https://docs.microsoft.com/azure-sphere/install/qs-blink-linux-cli)
   
