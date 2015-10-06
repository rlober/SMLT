cmake_minimum_required(VERSION 2.8.11)
set(projectName smlt)
project(${projectName})

set(smlt_MAJOR_VERSION 0)
set(smlt_MINOR_VERSION 1)
set(smlt_PATCH_VERSION 0)
set(smlt_VERSION
${smlt_MAJOR_VERSION}.${smlt_MINOR_VERSION}.${smlt_PATCH_VERSION})


# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# elseif(COMPILER_SUPPORTS_CXX0X)
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# else()
#         message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()


# Get pkg config for Eigen
find_package(PkgConfig REQUIRED)
# Find Eigen
PKG_CHECK_MODULES(PC_EIGEN eigen3>=3.0 REQUIRED)

# Find Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)

# Get source and header files for the smlt lib
file(GLOB folder_source ./src/*.cpp)
file(GLOB folder_header ./include/smlt/*.hpp)
file(GLOB python_scripts ./python/*.py)

source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})


# Include the necessary header dirs
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PC_EIGEN_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIR})

# Compile shared lib
add_library(${projectName} SHARED ${folder_source})

# Link to the appropriate libs
target_link_libraries(${projectName} ${Boost_LIBRARIES})

# Install everything
install(TARGETS ${projectName} DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${projectName} DESTINATION include)
install(FILES ${python_scripts} DESTINATION bin/scripts)

# Build unit tests
add_subdirectory(tests)

#pkg-config
add_subdirectory(pkg-config)
