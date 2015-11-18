cmake_minimum_required(VERSION 2.8.11)
project(smlt)

set(smlt_MAJOR_VERSION 0)
set(smlt_MINOR_VERSION 1)
set(smlt_PATCH_VERSION 0)
set(smlt_VERSION ${smlt_MAJOR_VERSION}.${smlt_MINOR_VERSION}.${smlt_PATCH_VERSION})


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(ConfigureCMakePackage)

# Find Eigen
find_package(Eigen REQUIRED)
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
                    ${Eigen_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIR})

# Compile shared lib
add_library(${PROJECT_NAME} SHARED ${folder_source})

# Link to the appropriate libs
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# Install everything
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME} DESTINATION include)
install(FILES ${python_scripts} DESTINATION bin/scripts)

# Build unit tests
add_subdirectory(tests)

#pkg-config
add_subdirectory(pkg-config)
##########################################################################
###### Cmake package config ##############################################

install(TARGETS ${PROJECT_NAME}
        EXPORT SMLTTargets
        DESTINATION lib)

# Default CMake package generator
configure_cmake_package(SMLTTargets ${PROJECT_NAME})
