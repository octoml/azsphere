#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(octoml_AS C CXX)
azsphere_configure_tools(TOOLS_REVISION "20.04")
azsphere_configure_api(TARGET_API_SET "5")

# Set for unit test or demo
OPTION(TEST "Is this test?" OFF)
OPTION(CONV2D "Is this conv2d?" OFF)
OPTION(CONV2D_NET "Is this conv2d network?" OFF)
OPTION(DEMO "Is this demo?" OFF)
OPTION(CIFAR "Is this cifar?" OFF)
OPTION(KEYWORD "Is this keyword model?" OFF)
OPTION(KEYWORD_DEMO "Is this keyword demo?" OFF)


# object files including the model
if(TEST)
  message("INFO: Building unit test!")
  SET(CFILES test.c bundle_static.c)
  SET(OBJS build/test_model.o)
elseif(CONV2D)
  message("INFO: Building Conv2d!")
  SET(CFILES conv2d.c bundle_static.c)
  SET(OBJS build/conv2d_model.o)
elseif(CONV2D_NET)
  message("INFO: Building Conv2d Network!")
  SET(CFILES conv2d_network.c bundle_static.c)
  SET(OBJS build/conv2d_model.o)
elseif(DEMO)
  message("INFO: Building demo!")
  SET(CFILES demo.c bundle_static.c)
  SET(OBJS build/model.o)
elseif(CIFAR)
  message("INFO: Building CIFAR demo!")
  SET(CFILES demo_cifar.c bundle_static.c)
  SET(OBJS build/cifar_model.o)
elseif(KEYWORD)
  message("INFO: Building KEYWORD test!")
  SET(CFILES keyword_test.c bundle_static.c)
  SET(OBJS build/keyword_model.o)
elseif(KEYWORD_DEMO)
  message("INFO: Building KEYWORD demo!")
  SET(CFILES keyword_demo_feature.c bundle_static.c)
  SET(OBJS build/keyword_model.o)
else()
  message("INFO: Hello test!")
  SET(CFILES mfcc_test.cpp src/example.cpp)
endif()

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES} ${OBJS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} applibs pthread gcc_s c)

# Add TVM headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/src)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dmlc-core/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dlpack/include)
# Project Headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ./include)
# 3rdparty libraries
SET(ML_KWS 3rdparty/ML_KWS/Deployment/Source)
# TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/MFCC)
# TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/NN)
# TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/NN/DS_CNN)
# TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/KWS)
# TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/KWS/KWS_DS_CNN)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC 3rdparty/ML_KWS/Deployment/Examples/simple_test)

azsphere_target_hardware_definition(${PROJECT_NAME} TARGET_DIRECTORY "Hardware/mt3620_rdb" TARGET_DEFINITION "my_hardware.json")

# Add MakeImage post-build command
# Note: No space between multiple file names
if(TEST)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/test_data.bin;build/test_output.bin;build/test_graph.json;build/test_params.bin")
elseif(CONV2D)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/conv2d_data.bin;build/conv2d_output.bin")
elseif(CONV2D_NET)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/conv2d_data.bin;build/conv2d_output.bin;build/id.bin")
elseif(DEMO)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/cat.bin")
elseif(CIFAR)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/cifar_data.bin;build/cifar_output.bin;build/cifar_params.bin;build/cifar_graph.bin;build/id.bin")
elseif(KEYWORD OR KEYWORD_DEMO)
  azsphere_target_add_image_package(
    ${PROJECT_NAME}
    RESOURCE_FILES
        "build/keyword_data.bin"
        "build/keyword_output.bin"
        "build/keyword_params.bin"
        "build/keyword_graph.bin"
        "build/id.bin")
else()
  azsphere_target_add_image_package(
    ${PROJECT_NAME})
endif()