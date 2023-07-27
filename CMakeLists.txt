cmake_minimum_required(VERSION 2.8)

project(StereoReconstruction)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(SUPERGLUE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/superglue)
message("Superglue include directories: ${SUPERGLUE_INCLUDE_DIRS}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${SUPERGLUE_INCLUDE_DIRS}
                   )

set(TEST src/stereo_dataset.h src/stereo_dataset.cpp
         src/PFMReadWrite.h src/PFMReadWrite.cpp
         src/feature_extractor.h src/feature_extractor.cpp
         src/sparse_matcher.h src/sparse_matcher.cpp
         src/camera_pose_estimator.h src/camera_pose_estimator.cpp
         src/dense_matcher.h src/dense_matcher.cpp 
         src/superglue/superglue.cpp src/scene_reconstructor.cpp
         src/test.cpp)

set(EXPERIMENT src/stereo_dataset.h src/stereo_dataset.cpp
               src/PFMReadWrite.h src/PFMReadWrite.cpp
               src/feature_extractor.h src/feature_extractor.cpp
               src/sparse_matcher.h src/sparse_matcher.cpp
               src/camera_pose_estimator.h src/camera_pose_estimator.cpp
               src/dense_matcher.h src/dense_matcher.cpp
               src/superglue/superglue.cpp src/scene_reconstructor.cpp
               src/experiment_designer.h src/experiment_designer.cpp
               src/scene_reconstructor.cpp
               src/experiment.cpp)

add_executable(test ${TEST})
target_link_libraries(test ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property(TARGET test PROPERTY CXX_STANDARD 17)

add_executable(experiment ${EXPERIMENT})
target_link_libraries(experiment ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property(TARGET experiment PROPERTY CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/src/superglue/SuperPoint.zip
  ${CMAKE_CURRENT_SOURCE_DIR}/src/superglue/SuperGlue.zip
  DESTINATION ${CMAKE_BINARY_DIR}/superglue/
)




