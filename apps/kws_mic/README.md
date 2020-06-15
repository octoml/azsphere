## Overview

This application is a part of the TVM demo on [MT3620 Azure Sphere](https://azure.microsoft.com/en-us/services/azure-sphere/get-started/). In our TVM demo, we deploy a [DS-CNN Keyword Spotting](https://github.com/ARM-software/ML-KWS-for-MCU) model on Azure Sphere for the first time using [Apache TVM](https://tvm.apache.org/). This model has two steps; the first step pre-processes the audio to extract MFCC features and second part runs processed audio through a DS-CNN model. We deploy feature extraction on Cortex-M4 core on Azure Sphere and DS-CNN model inference on Cortex-A7. To run this demo, you need to deploy this application and the [partner application](https://github.com/octoml/azsphere) which implements the TVM runtime.

This repository showcases the audio pre-processing part of DS-CNN model on Cortex-M4. We present two demos here:
* Demo1: We use pre-recorded audio files and perform pre-processing.
* Demo2: We use an analog microphone to show a real-time demo.

## Hardware Requirements
- Linux machine
- [MT3620 Azure Sphere board](https://www.seeedstudio.com/Azure-Sphere-MT3620-Development-Kit-US-Version-p-3052.html)
- micro USB cable
- [Microphone](https://www.adafruit.com/product/1063) (only for Demo2)
- A USB to UART FTDI cable like [TTL-234X Serial Cables](https://www.ftdichip.com/Products/Cables/TTL234XSerial.htm) (only for debugging Cortex-M4)

## Software Requirements
- Linux with Azure Sphere SDK (follow [Azure Sphere documentation](https://docs.microsoft.com/en-us/azure-sphere/) to setup SDK and device)
- [Install GNU ARM Embedded Toolchain 9.2.1.](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)
    - Make sure to export ```ARM_GNU_PATH="PATH/to/GCC/ARM/none/eabi/bin"``` in your environment.
- Python 3.6+
- Tensorflow

## Getting Started
1. Clone this repository (Use ```git clone recursive``` to clone the submodules)
2. Follow these to build the application. Note that there are multiple WAV file under [data](./data) that you can try.
```bash
$ cd ~/azsphere-mic
$ make clean
(Only DEMO1) => $ python3 python/mfcc.py -f (choose a WAV file)
$ make demo1(demo2)
```
3. For ```DEMO2``` connect the microphone to ADC interface on Azure Sphere:
    - H2.11 (ADC0) -> OUT
    - H2.2 (GND) -> GND
    - H3.3 (3.3) -> VCC
4. Follow these to connect and flash this app on device. You need ```sudo``` access to connect to the device.
```bash
cd ~/azsphere-mic
make connect
make program
```
5. Now, if you push button B on Azure Sphere the Cortex-M4 extracts the feature of audio samples and sends it to the Cortex-A7. If you already deployed the partner application, you will see change in LED colors.

## Debugging
To enable debugging follow these steps:
1. Connect to the debug UART (connect to H3.6 (IO0_TXD) and H3.2 (GND)).
2. Rebuild the application using ```make demo1(demo2) CFLAGS=-DAS_DEBUG=1``` and program it.

## References
Here, we list some of the resources that we used to enable this demo:

- [Keyword spotting on Arm Cortex-M Microcontrollers](https://github.com/ARM-software/ML-KWS-for-MCU)
- [MT3620-M4-Samples](https://github.com/CodethinkLabs/mt3620-m4-samples)
