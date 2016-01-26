cmake_minimum_required(VERSION 2.8.3)
project(orb_slam)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    tf
    sensor_msgs
    image_transport
    cv_bridge
    g2o_catkin
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
#find_package(Cholmod REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    /usr/include/suitesparse
    )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

add_subdirectory(Thirdparty/DBoW2)

add_library(${PROJECT_NAME}
    src/Tracking.cc
    src/LocalMapping.cc
    src/LoopClosing.cc
    src/ORBextractor.cc
    src/ORBmatcher.cc
    src/FramePublisher.cc
    src/Converter.cc
    src/MapPoint.cc
    src/KeyFrame.cc
    src/Map.cc
    src/MapPublisher.cc
    src/Optimizer.cc
    src/PnPsolver.cc
    src/Frame.cc
    src/KeyFrameDatabase.cc
    src/Sim3Solver.cc
    src/Initializer.cc
    )
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${EIGEN3_LIBS}
    DBoW2
    cholmod
    )

add_executable(orb_slam_main src/main.cc)
target_link_libraries(orb_slam_main ${PROJECT_NAME})

add_executable(orb_slam_node src/Node.cc)
target_link_libraries(orb_slam_node ${PROJECT_NAME})
