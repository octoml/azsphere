CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
PROJECT(azsphere_mic C CXX ASM)
azsphere_configure_tools(TOOLS_REVISION "20.04")
azsphere_configure_api(TARGET_API_SET "5+Beta2004")

#3rdparty dependencies
SET(CMSIS 3rdparty/CMSIS_5/CMSIS)

if(DEMO1)
SET(C_FLAGS     "-Wall -Wextra -ffunction-sections -fdata-sections -fno-delete-null-pointer-checks -Wconversion -fsingle-precision-constant -MMD -O0 -g1 -march=armv7e-m -mcpu=cortex-m4")
SET(CXX_FLAGS   "-Wall -Wextra -ffunction-sections -fdata-sections -fno-delete-null-pointer-checks -Wconversion -fsingle-precision-constant -MMD -O0 -g1 -march=armv7e-m -mcpu=cortex-m4")
SET(ASM_FLAGS   "-x assembler-with-cpp -Wall -Wextra -ffunction-sections -fdata-sections -fno-delete-null-pointer-checks -Wconversion -fsingle-precision-constant -MMD -O0 -g1 -mcpu=cortex-m4")
SET(LD_FLAGS    "-Wl,--verbose -Wl,--start-group -Wl,--wrap,main -Wl,--gc-sections -Wl,--end-group -march=armv7e-m -mcpu=cortex-m4")

SET(CFILES  demo_recorded.c
            lib/VectorTable.c
            lib/UART.c lib/Print.c lib/GPIO.c lib/GPT.c
            lib/mt3620-intercore.c lib/logical-intercore.c lib/logical-dpc.c
            src/intercore.c
            src/kws.c
            src/mfcc.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_init_f32.c
            ${CMSIS}/DSP/Source/CommonTables/arm_common_tables.c
            ${CMSIS}/DSP/Source/CommonTables/arm_const_structs.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_bitreversal2.S
    )
set_property(SOURCE ${CMSIS}/DSP/Source/TransformFunctions/arm_bitreversal2.S PROPERTY LANGUAGE ASM)

elseif(DEMO2)
SET(C_FLAGS     "-Wall -Wextra -ffunction-sections -fdata-sections -fno-delete-null-pointer-checks -Wconversion -fsingle-precision-constant -MMD -O0 -g1 -march=armv7e-m -mcpu=cortex-m4")
SET(CXX_FLAGS   "-Wall -Wextra -ffunction-sections -fdata-sections -fno-delete-null-pointer-checks -Wconversion -fsingle-precision-constant -MMD -O0 -g1 -march=armv7e-m -mcpu=cortex-m4")
SET(ASM_FLAGS   "-x assembler-with-cpp -Wall -Wextra -ffunction-sections -fdata-sections -fno-delete-null-pointer-checks -Wconversion -fsingle-precision-constant -MMD -O0 -g1 -mcpu=cortex-m4")
SET(LD_FLAGS    "-Wl,--verbose -Wl,--start-group -Wl,--wrap,main -Wl,--gc-sections -Wl,--end-group -march=armv7e-m -mcpu=cortex-m4")

SET(CFILES  demo_mic.c
            lib/VectorTable.c
            lib/UART.c lib/Print.c lib/GPIO.c lib/GPT.c lib/ADC.c
            lib/mt3620-intercore.c lib/logical-intercore.c lib/logical-dpc.c
            src/intercore.c
            src/kws.c
            src/mfcc.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_cfft_init_f32.c
            ${CMSIS}/DSP/Source/CommonTables/arm_common_tables.c
            ${CMSIS}/DSP/Source/CommonTables/arm_const_structs.c
            ${CMSIS}/DSP/Source/TransformFunctions/arm_bitreversal2.S
    )
set_property(SOURCE ${CMSIS}/DSP/Source/TransformFunctions/arm_bitreversal2.S PROPERTY LANGUAGE ASM)

else()
endif()

SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${C_FLAGS}")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
SET(CMAKE_ASM_FLAGS  "${CMAKE_ASM_FLAGS} ${ASM_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS}")

# Create executable
ADD_EXECUTABLE(${PROJECT_NAME} ${CFILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC stdc++)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC supc++)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC m)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC c)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC gcc)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC nosys)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINK_DEPENDS ${CMAKE_SOURCE_DIR}/linker.ld)

# Project Headers
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

#3rdparty libraries
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMSIS}/DSP/Include)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMSIS}/Core/Include)

#image
azsphere_target_add_image_package(${PROJECT_NAME})