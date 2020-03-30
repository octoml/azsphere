#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(HelloWorld_HighLevelApp C)

# object files including the model
SET(OBJS
  build/test_model.o
)

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} main.c bundle_static.c ${OBJS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} applibs pthread gcc_s c)

#SET_SOURCE_FILES_PROPERTIES(
#  ${OBJS}
#  PROPERTIES
#  EXTERNAL_OBJECT true
#  GENERATED true
#)

# Add TVM headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dmlc-core/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ~/tvm/3rdparty/dlpack/include)

# Add MakeImage post-build command
SET(ADDITIONAL_APPROOT_INCLUDES "build/test_data.bin;build/test_output.bin;build/test_graph.json;build/test_params.bin")
INCLUDE("${AZURE_SPHERE_MAKE_IMAGE_FILE}")
