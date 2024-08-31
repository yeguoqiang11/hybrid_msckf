cmake_minimum_required(VERSION 3.10)
project(inslam)

set(CMAKE_CONFIGURATION_TYPES Release RelWithDebInfo CACHE STRING INTERNAL FORCE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#platform: WIN32 OSX IOS ANDROID LINUX TEGRA
if (APPLE AND (NOT IOS))
    set(OSX ON)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
	    set(TEGRA ON) 
    else ()
        set(LINUX ON)
    endif ()
endif ()

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /WX ")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /WX ")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror ")
endif()

include("cmake/utils.cmake")

#options
option(WITH_SAMPLES "build with samples" ON)

#dependencies
add_subdirectory(dependencies)
list(APPEND CMAKE_PREFIX_PATH ${DEP_CMAKE_CONFIG_DIR})
message(STATUS "prefix path:${CMAKE_PREFIX_PATH}")
set(OpenCV_DIR "/home/insta360/InstallFold/opencv-4.1.0/build")
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)

#src
add_subdirectory_with_folder("inslam" src)

#samples
if (WITH_SAMPLES)
	add_subdirectory_with_folder("samples" samples)
endif ()

#install
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/CHANGELOG.md DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/docs/release_readme.md DESTINATION . RENAME README.md)
