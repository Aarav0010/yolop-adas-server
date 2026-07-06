cmake_minimum_required(VERSION 3.15)
project(YolopCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    # Find OpenCV
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build")
    # ONNX Runtime (Downloaded locally)
    set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu_cuda12-1.27.0/include")
    set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu_cuda12-1.27.0/lib")
else()
    # On Linux, these are usually available globally or via CMake args
    if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR)
        set(ONNXRUNTIME_INCLUDE_DIR "/usr/include/onnxruntime")
    endif()
    if(NOT DEFINED ONNXRUNTIME_LIB_DIR)
        set(ONNXRUNTIME_LIB_DIR "/usr/lib")
    endif()
endif()

find_package(OpenCV REQUIRED)

# Windows specific
if(WIN32)
    add_compile_definitions(_WIN32_WINNT=0x0A00)
    # Required by cpp-httplib
    link_libraries(ws2_32 crypt32)
endif()

add_executable(yolop_app main.cpp)

target_include_directories(yolop_app PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR} # For httplib.h
)

target_link_directories(yolop_app PRIVATE ${ONNXRUNTIME_LIB_DIR})

find_package(Threads REQUIRED)

target_link_libraries(yolop_app PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    Threads::Threads
)

if(WIN32)
# Copy ONNX Runtime DLLs to output directory for running
add_custom_command(TARGET yolop_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu_cuda12-1.27.0/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:yolop_app>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu_cuda12-1.27.0/lib/onnxruntime_providers_shared.dll"
    $<TARGET_FILE_DIR:yolop_app>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu_cuda12-1.27.0/lib/onnxruntime_providers_cuda.dll"
    $<TARGET_FILE_DIR:yolop_app>
)
endif()
