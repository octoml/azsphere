CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(azsphere_tvm C)
azsphere_configure_tools(TOOLS_REVISION "20.04")
azsphere_configure_api(TARGET_API_SET "5+Beta2004")

# Set mode here
OPTION(TEST "TVM a+b test" OFF)
OPTION(CONV2D "TVM Relay conv2d test" ON)
OPTION(CONV2D_NET "Is this conv2d network?" OFF)
OPTION(DEMO "Is this demo?" OFF)
OPTION(CIFAR "Is this cifar?" OFF)
OPTION(KEYWORD "Is this keyword model?" OFF)
OPTION(KEYWORD_DEMO "Is this keyword demo?" OFF)
OPTION(HELLO "Is this hello world?" OFF)
OPTION(INTERCOMM "Is this intercore communication?" OFF)
OPTION(DEMO1 "TVM, Intercore connection of preprocessing" OFF)

#3rdparty directories
SET(ML_KWS 3rdparty/ML_KWS/Deployment)
SET(CMSIS 3rdparty/CMSIS_5/CMSIS)

if(TEST)
  SET(CFILES  test.c
              src/bundle_static.c src/utils.c)
  SET(OBJS build/test_model.o)
elseif(CONV2D)
  SET(CFILES  conv2d.c
              src/bundle_static.c src/utils.c)
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
  SET(CFILES  keyword_test.c
              src/bundle_static.c src/utils.c
  )
  SET(OBJS build/keyword_model.o)
elseif(KEYWORD_DEMO)
  message("INFO: Building KEYWORD demo!")
  SET(CFILES keyword_demo_feature.c bundle_static.c)
  SET(OBJS build/keyword_model.o)
elseif(HELLO)
  SET(CFILES hello.c)
elseif(INTERCOMM)
  SET(CFILES  intercomm.c
              src/eventloop_timer_utilities.c
  )
elseif(DEMO1)
  SET(CFILES  demo1.c 
              src/bundle_static.c
              src/utils.c src/eventloop_timer_utilities.c
              src/intercore.c src/tvmruntime.c
  )
  SET(OBJS build/keyword_model.o)
else()
  message("INFO: Hello test!")
  SET(CFILES mfcc_test.cpp 
      src/example.cpp src/kws.cpp
      ${ML_KWS}/Source/MFCC/mfcc.cpp
      ${CMSIS}/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
      ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_f32.c
      ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
      ${CMSIS}/DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
      ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_init_f32.c
      ${CMSIS}/DSP/Source/CommonTables/arm_common_tables.c
      ${CMSIS}/DSP/Source/CommonTables/arm_const_structs.c
      ${CMSIS}/DSP/Source/TransformFunctions/arm_bitreversal2.S
  )
  set_property(SOURCE ${CMSIS}/DSP/Source/TransformFunctions/arm_bitreversal2.S PROPERTY LANGUAGE C)
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
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})
# 3rdparty libraries
#TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/Source/MFCC)
#TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${ML_KWS}/Examples/simple_test)
#TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMSIS}/DSP/Include)
#TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMSIS}/Core/Include)

azsphere_target_hardware_definition(${PROJECT_NAME} TARGET_DIRECTORY "Hardware/mt3620_rdb" TARGET_DEFINITION "my_hardware.json")

# Add MakeImage post-build command
# Note: No space between multiple file names
if(TEST)
  azsphere_target_add_image_package(
    ${PROJECT_NAME} RESOURCE_FILES
      "build/test_data.bin" "build/test_output.bin" "build/test_graph.json" "build/test_params.bin"
  )
elseif(CONV2D)
  azsphere_target_add_image_package(
    ${PROJECT_NAME} RESOURCE_FILES
      "build/conv2d_data.bin" "build/conv2d_output.bin" "build/conv2d_graph.json" "build/conv2d_params.bin"
  )
elseif(CONV2D_NET)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/conv2d_data.bin;build/conv2d_output.bin;build/id.bin")
elseif(DEMO)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/cat.bin")
elseif(CIFAR)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/cifar_data.bin;build/cifar_output.bin;build/cifar_params.bin;build/cifar_graph.bin;build/id.bin")
elseif(KEYWORD OR KEYWORD_DEMO OR DEMO1)
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