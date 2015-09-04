cmake_minimum_required(VERSION 2.8.11)
project(smltUnitTests)


include_directories(${smlt_SOURCE_DIR}/include
                    ${PC_EIGEN_INCLUDE_DIRS}
                    )

# kernelTests
add_executable(kernelTests kernelTests.cpp)
target_link_libraries(kernelTests smlt)
install(TARGETS kernelTests DESTINATION bin)

# gaussianProcessTests
add_executable(gaussianProcessTests gaussianProcessTests.cpp)
target_link_libraries(gaussianProcessTests smlt)
install(TARGETS gaussianProcessTests DESTINATION bin)

# bayesianOptimization
add_executable(bayesianOptimizationTests bayesianOptimizationTests.cpp)
target_link_libraries(bayesianOptimizationTests smlt)
install(TARGETS bayesianOptimizationTests DESTINATION bin)

# smltUtilities
add_executable(smltUtilitiesTests smltUtilitiesTests.cpp)
target_link_libraries(smltUtilitiesTests smlt)
install(TARGETS smltUtilitiesTests DESTINATION bin)
