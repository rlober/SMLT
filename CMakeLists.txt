cmake_minimum_required(VERSION 2.8.11)
project(smlt)

set(VARS_PREFIX "SMLT")
# Set the project version.
set(${VARS_PREFIX}_MAJOR_VERSION 1)
set(${VARS_PREFIX}_MINOR_VERSION 0)
set(${VARS_PREFIX}_PATCH_VERSION 0)
set(${VARS_PREFIX}_VERSION ${${VARS_PREFIX}_MAJOR_VERSION}.${${VARS_PREFIX}_MINOR_VERSION}.${${VARS_PREFIX}_PATCH_VERSION})


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
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        FILES_MATCHING PATTERN *.h*)
install(FILES ${python_scripts} DESTINATION bin/scripts)

# Build unit tests
add_subdirectory(tests)

#pkg-config
add_subdirectory(pkg-config)
##########################################################################
###### Cmake package config ##############################################

install(TARGETS ${PROJECT_NAME}
        EXPORT SMLT
        DESTINATION lib)

set_property(GLOBAL APPEND PROPERTY ${VARS_PREFIX}_TARGETS ${PROJECT_NAME})
set_property(GLOBAL APPEND PROPERTY PACKAGE_${VARS_PREFIX}_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)

# Default CMake package generator
# configure_cmake_package(SMLTTargets ${PROJECT_NAME})

# Add an uninstallation target so you can just run - make uninstall - to remove the binary.
include(InstallBasicPackageFiles)
get_property(PACKAGE_${VARS_PREFIX}_INCLUDEDIR GLOBAL PROPERTY PACKAGE_${VARS_PREFIX}_INCLUDEDIR)
install_basic_package_files(SMLT      VARS_PREFIX ${VARS_PREFIX}
                                        VERSION ${${VARS_PREFIX}_VERSION}
                                        COMPATIBILITY SameMajorVersion
                                        TARGETS_PROPERTY ${VARS_PREFIX}_TARGETS
                                        NO_CHECK_REQUIRED_COMPONENTS_MACRO
                                    )
include(AddUninstallTarget)
