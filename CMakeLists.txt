cmake_minimum_required(VERSION 2.8)

project(StereoReconstruction)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                   )

set(TEST src/stereo_dataset.h src/stereo_dataset.cpp src/test.cpp)
add_executable(test ${TEST})
target_link_libraries(test ${OpenCV_LIBS})







