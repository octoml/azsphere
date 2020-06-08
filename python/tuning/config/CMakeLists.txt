#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(octoml_AS C)

# object files including the model
message("INFO: Building Conv2d!")
SET(CFILES conv2d_network.c bundle_static.c)
SET(OBJS build/model.o)

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES} ${OBJS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} applibs pthread gcc_s c)

# Add TVM headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/src)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dmlc-core/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dlpack/include)
# Project Headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ../../../include)

# Add MakeImage post-build command
# Note: No space between multiple file names
SET(ADDITIONAL_APPROOT_INCLUDES "build/conv2d_data.bin;build/conv2d_output.bin;build/id.bin")
INCLUDE("${AZURE_SPHERE_MAKE_IMAGE_FILE}")
