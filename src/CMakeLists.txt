cmake_minimum_required(VERSION 3.10)
project(src)

set(INCLUDE_SRC 
    ../include/Vio/DearSystem.hpp
)

set(UTILS_SRC
    Utils/json.hpp
    Utils/MathUtil.cc
    Utils/MathUtil.h
    Utils/DataIOUtil.cc
    Utils/DataIOUtil.h
    Utils/VideoProvider.cc
    Utils/VideoProvider.h
    Utils/PerformanceTest.h
)

set(VIEWER_SRC 
    Viewer/PangoViewer.h
    Viewer/PangoViewer.cc
)

file(GLOB VIO_INIT_SRC
        Vio/Initialize/FrameHandler.cpp
        Vio/Initialize/FrameHandler.hpp
        Vio/Initialize/InitTypes.cpp
        Vio/Initialize/InitTypes.hpp
        Vio/Initialize/StereoDepthCost.hpp
        Vio/Initialize/Initializer.cpp
        Vio/Initialize/Initializer.hpp
        Vio/Initialize/Optimization.cpp
        Vio/Initialize/Optimization.hpp
        Vio/Initialize/Utils/EssentialRansac.cc
        Vio/Initialize/Utils/EssentialRansac.h
        Vio/Initialize/Utils/PoseInitialization.cc
        Vio/Initialize/Utils/PoseInitialization.h
        Vio/Initialize/Utils/Triangulation.cc
        Vio/Initialize/Utils/Triangulation.h
        Vio/Initialize/Utils/epnp.cc
        Vio/Initialize/Utils/epnp.h
        Vio/Initialize/Imu/CamCost.h
        Vio/Initialize/Imu/Imu.cc
        Vio/Initialize/Imu/Imu.h
        Vio/Initialize/Imu/ImuAligment.cc
        Vio/Initialize/Imu/ImuAligment.h
        Vio/Initialize/Imu/ImuCostFunction.h
        Vio/Initialize/Imu/ImuCalibration.cc
        Vio/Initialize/Imu/ImuCalibration.h
        Vio/Initialize/Imu/AttitudeFilter.cc
        Vio/Initialize/Imu/AttitudeFilter.h
        Vio/Initialize/Imu/LinearAlignment.cc
        Vio/Initialize/Imu/LinearAlignment.h
)

set(VIO_SRC 
    Vio/Factor/ReproErrorX.h
    Vio/Factor/SphereReproErrorX.h
    Vio/Caimura.cpp
    Vio/Caimura.hpp
    Vio/DearSystem.cpp
    Vio/EkfVio.cpp
    Vio/EkfVio.hpp
    Vio/FeatureObservation.h
    Vio/ImageProcessor.cpp
    Vio/ImageProcessor.h
    Vio/Insfeature.hpp
    Vio/MsckfVio.cpp
    Vio/MsckfVio.h
    Vio/Triangulator.cpp
    Vio/Triangulator.h
    Vio/VioFeature.cpp
    Vio/VioFeature.h
    Vio/FeatureGrid.hpp
    Vio/SlidingWindow.cpp
    Vio/SlidingWindow.hpp
    Vio/SlidingWindowCost.hpp
    Vio/VoxelMap.hpp
    Vio/VoxelMap.cpp
    ${VIO_INIT_SRC}
)

set(ALL_SRC 
    ${INCLUDE_SRC}
    ${UTILS_SRC} 
    ${VIO_SRC} 
    ${VIEWER_SRC}
)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${ALL_SRC})

add_library(inslam STATIC ${ALL_SRC})

set(INSLAM_INC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include 
)

target_include_directories(inslam PUBLIC 
    "$<BUILD_INTERFACE:${INSLAM_INC}>"
    "$<INSTALL_INTERFACE:include>"
)

target_include_directories(inslam PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(inslam PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES} pangolin)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/inslamInstall.cmake)

