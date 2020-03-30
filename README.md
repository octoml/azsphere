# TVM on Azure Sphere Platform

## Azure Sphere Files
| File/folder | Description |
|-------------|-------------|
|   main.c    | Source file. |
| app_manifest.json |Manifest file. |
| CMakeLists.txt | Contains the project information and produces the build. |
| CMakeSettings.json| Configures Visual Studio to use CMake with the correct command-line options. |
|launch.vs.json |Tells Visual Studio how to deploy and debug the application.|
|.vscode |Contains settings.json that configures Visual Studio Code to use CMake with the correct options, and tells it how to deploy and debug the application. |

## Prerequisites

The sample requires the following hardware:

1. [Seeed MT3620 Development Kit](https://aka.ms/azurespheredevkits) or other hardware that implements the [MT3620 Reference Development Board (RDB)](https://docs.microsoft.com/azure-sphere/hardware/mt3620-reference-board-design) design.

**Note:** By default, this sample targets [MT3620 reference development board (RDB)](https://docs.microsoft.com/azure-sphere/hardware/mt3620-reference-board-design) hardware, such as the MT3620 development kit from Seeed Studio. To build the sample for different Azure Sphere hardware, change the Target Hardware Definition Directory in the project properties. For detailed instructions, see the [README file in the Hardware folder](./Hardware/README.md).

## Prepare the environment

1. Ensure that your Azure Sphere device is connected to your computer and your computer is connected to the internet.
1. Even if you've performed this setup previously, ensure that you have Azure Sphere SDK version 20.01 or above. At the command prompt, run **azsphere show-version** to check. Install the Azure Sphere SDK for [Windows](https://docs.microsoft.com/azure-sphere/install/install-sdk) or [Linux](https://docs.microsoft.com/azure-sphere/install/install-sdk-linux) as needed.
1. Enable application development, if you have not already done so, by entering the following line at the command prompt:

   `azsphere device enable-development`

1. Clone this repository in home directory("~/").

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
   
