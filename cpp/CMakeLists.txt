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

if(NOT WIN32)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-base-1.0 gstreamer-pbutils-1.0 gstreamer-audio-1.0 gstreamer-tag-1.0)
endif()

add_executable(yolop_app main.cpp)

target_include_directories(yolop_app PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR} # For httplib.h
)

find_package(Threads REQUIRED)

target_link_libraries(yolop_app PRIVATE
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    Threads::Threads
)
