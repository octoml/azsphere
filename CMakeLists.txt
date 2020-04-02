#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(octoml_AS C)

# Set for unit test or demo
OPTION(TEST "Is this demo?" OFF)
OPTION(CONV2D "Is this demo?" ON)

# object files including the model
if(TEST)
  message("INFO: Building unit test!")
  SET(CFILES test.c)
  SET(OBJS build/test_model.o)
elseif(CONV2D)
  message("INFO: Building Conv2d!")
  SET(CFILES conv2d.c)
  SET(OBJS build/conv2d_model.o)
else()
  message("INFO: Building demo!")
  SET(CFILES demo.c)
  SET(OBJS build/model.o)
endif()

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES} bundle_static.c ${OBJS})
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

# Add MakeImage post-build command
# Note: No space between multiple file names
if(TEST)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/test_data.bin;build/test_output.bin;build/test_graph.json;build/test_params.bin")
elseif(CONV2D)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/conv2d_data.bin")
else()
  SET(ADDITIONAL_APPROOT_INCLUDES "build/cat.bin")
endif()
INCLUDE("${AZURE_SPHERE_MAKE_IMAGE_FILE}")
