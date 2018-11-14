project(realsense-testing)
cmake_minimum_required(VERSION 2.8)

add_executable (pcl_testing pcl_testing.cpp)
add_executable (region_growing_rgb_segmentation region_growing_rgb_segmentation.cpp)

find_package(PCL 1.6 REQUIRED)
# SET(CMAKE_CXX_FLAGS "-std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(OpenCV_DIR /usr/share/OpenCV/OpenCVConfig.cmake)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries (pcl_testing ${PCL_LIBRARIES} librealsense2.so ${OpenCV_LIBS})
#target_link_libraries (pcl_testing_2 ${PCL_LIBRARIES} librealsense2.so ${OpenCV_LIBS})
target_link_libraries (region_growing_rgb_segmentation ${PCL_LIBRARIES})
