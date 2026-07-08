cmake_minimum_required(VERSION 3.15)
project(YolopCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    # Find OpenCV
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build")
endif()

find_package(OpenCV REQUIRED)

# Windows specific
if(WIN32)
    add_compile_definitions(_WIN32_WINNT=0x0A00)
    # Required by cpp-httplib
    link_libraries(ws2_32 crypt32)
endif()

add_executable(yolop_app main.cpp)

if(NOT WIN32)
    # GStreamer no longer required for CPU OpenCV build
    target_link_libraries(yolop_app PRIVATE onnxruntime)
endif()

target_include_directories(yolop_app PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR} # For httplib.h
)

find_package(Threads REQUIRED)

target_link_libraries(yolop_app PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# Add ONNX Runtime for Windows Local Test
if(WIN32)
    option(USE_CUDA "Enable CUDA Support for Local GPU" OFF)
    
    if(USE_CUDA)
        add_compile_definitions(USE_CUDA)
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_gpu")
    else()
        set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
    endif()
    
    target_include_directories(yolop_app PRIVATE "${ONNXRUNTIME_DIR}/include")
    target_link_directories(yolop_app PRIVATE "${ONNXRUNTIME_DIR}/lib")
    target_link_libraries(yolop_app PRIVATE onnxruntime)
    
    if(USE_CUDA)
        add_custom_command(TARGET yolop_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:yolop_app>
        )
    else()
        add_custom_command(TARGET yolop_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:yolop_app>
        )
    endif()
endif()
