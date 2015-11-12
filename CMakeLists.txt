cmake_minimum_required(VERSION 2.8.11)
project(smlt)

set(smlt_MAJOR_VERSION 0)
set(smlt_MINOR_VERSION 1)
set(smlt_PATCH_VERSION 0)
set(smlt_VERSION ${smlt_MAJOR_VERSION}.${smlt_MINOR_VERSION}.${smlt_PATCH_VERSION})


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

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
# add_subdirectory(pkg-config)
##########################################################################
###### Cmake package config ##############################################

include(CMakePackageConfigHelpers)

set(CMAKE_CONFIG_DEST "share/cmake/${PROJECT_NAME}/")
set(INCLUDE_INSTALL_DIR "include/${PROJECT_NAME}")
string(TOUPPER ${PROJECT_NAME} CMAKE_PROJECT_NAME)

configure_package_config_file(
       ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_PROJECT_NAME}Config.cmake.in
       ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake
       INSTALL_DESTINATION ${CMAKE_CONFIG_DEST}
       PATH_VARS INCLUDE_INSTALL_DIR CMAKE_CONFIG_DEST)

#configure_file(${CMAKE_PROJECT_NAME}ConfigVersion.cmake.in
# "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake" @ONLY)

install(FILES
 "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
# "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
 DESTINATION ${CMAKE_CONFIG_DEST})
