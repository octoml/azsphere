#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(octoml_AS C)

# Set for unit test or demo
OPTION(TEST "Is this test?" OFF)
OPTION(CONV2D "Is this conv2d?" OFF)
OPTION(CONV2D_NET "Is this conv2d network?" OFF)
OPTION(DEMO "Is this demo?" OFF)
OPTION(CIFAR "Is this cifar?" ON)


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
else()
  message("INFO: CIFAR TASK0!")
  SET(CFILES cifar_task3.c bundle_static.c)
  SET(OBJS build/conv2d_model.o)
endif()

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES} ${OBJS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} applibs pthread gcc_s c)

#SET_SOURCE_FILES_PROPERTIES(
#  ${OBJS}
#  PROPERTIES
#  EXTERNAL_OBJECT true
#  GENERATED true
#)

# Add TVM headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/src)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dmlc-core/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dlpack/include)
# Project Headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ./include)

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
else()
  SET(ADDITIONAL_APPROOT_INCLUDES "build/conv2d_data.bin;build/conv2d_output.bin;build/conv2d_params.bin;build/conv2d_graph.bin;build/id.bin")
endif()
INCLUDE("${AZURE_SPHERE_MAKE_IMAGE_FILE}")