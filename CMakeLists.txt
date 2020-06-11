CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(azsphere_tvm C)
azsphere_configure_tools(TOOLS_REVISION "20.04")
azsphere_configure_api(TARGET_API_SET "5+Beta2004")

if(TEST)
  SET(CFILES  test.c
              src/bundle_static.c src/utils.c
  )
  SET(OBJS build/test_model.o)
elseif(CONV2D)
  SET(CFILES  conv2d.c
              src/bundle_static.c src/utils.c
  )
  SET(OBJS build/conv2d_model.o)
elseif(CIFAR)
  message("INFO: Building CIFAR demo!")
  SET(CFILES demo_cifar.c bundle_static.c)
  SET(OBJS build/cifar_model.o)
elseif(KEYWORD)
  SET(CFILES  kws_test.c
              src/bundle_static.c src/utils.c
  )
  SET(OBJS build/keyword_model.o)
elseif(KWS_DEMO)
  SET(CFILES  kws_demo.c 
              src/bundle_static.c
              src/utils.c src/eventloop_timer_utilities.c
              src/intercore.c src/tvmruntime.c
  )
  SET(OBJS build/keyword_model.o)
else()
endif()

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES} ${OBJS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} applibs pthread gcc_s c)

# 3rdparty libraries
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $ENV{TVM_HOME}/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $ENV{TVM_HOME}/src)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $ENV{TVM_HOME}/3rdparty/dmlc-core/include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $ENV{TVM_HOME}/3rdparty/dlpack/include)
# Project Headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})


azsphere_target_hardware_definition(${PROJECT_NAME} TARGET_DIRECTORY "Hardware/mt3620_rdb" TARGET_DEFINITION "my_hardware.json")

# Add MakeImage post-build command
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
elseif(CIFAR)
  SET(ADDITIONAL_APPROOT_INCLUDES "build/cifar_data.bin;build/cifar_output.bin;build/cifar_params.bin;build/cifar_graph.bin;build/id.bin")
elseif(KEYWORD)
  azsphere_target_add_image_package(
    ${PROJECT_NAME}
    RESOURCE_FILES
        "build/keyword_data.bin"
        "build/keyword_output.bin"
        "build/keyword_params.bin"
        "build/keyword_graph.bin"
        "build/id.bin"
    )
elseif(KWS_DEMO)
  azsphere_target_add_image_package(
    ${PROJECT_NAME}
    RESOURCE_FILES
        "build/keyword_params.bin"
        "build/keyword_graph.bin"
    )
else()
  azsphere_target_add_image_package(
    ${PROJECT_NAME})
endif()