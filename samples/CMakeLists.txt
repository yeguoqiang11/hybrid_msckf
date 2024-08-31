cmake_minimum_required(VERSION 3.10)
project(samples)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
link_libraries(pangolin inslam)

add_executable(MonoVioDemo  Vio/MonoVioDemo.cc)

add_executable(StereoVioDemo Vio/StereoVioDemo.cc)

add_executable(UzhFpvDemo  Vio/UzhFpvDemo.cc)

add_executable(InitVSlamDemo Vio/InitVSlamDemo.cc)

# Debug Tools
add_executable(ImuCalibrateDemo DebugTools/ImuCalibrateDemo.cc)

add_executable(StabilizerDemo DebugTools/StabilizerDemo.cc)

add_executable(FisheyeToPanoDemo DebugTools/FisheyeToPanoDemo.cc)

add_executable(SparseStereoDemo DebugTools/SparseStereoDemo.cc)