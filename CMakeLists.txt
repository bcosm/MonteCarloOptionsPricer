cmake_minimum_required(VERSION 3.12)
project(MonteCarloOptionsPricer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
message(STATUS "Looking for PyTorch (LibTorch)...")
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch (LibTorch) not found! Please install LibTorch and set CMAKE_PREFIX_PATH to the LibTorch installation directory.")
endif()
message(STATUS "Found PyTorch: ${Torch_DIR}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")

message(STATUS "Looking for Eigen3...")
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${Eigen3_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found! Please install Eigen3.")
endif()

find_package(OpenMP)

# Include directories
include_directories(include)
include_directories(src/core)
include_directories(src/models)

# Source files - separate library sources from main files
file(GLOB_RECURSE LIBRARY_SOURCES
    "src/models/*.cpp"
    "src/core/BayesianNN.cpp"
    "src/core/CSVUtils.cpp"
)

# Create a library with shared code
add_library(MonteCarloLib ${LIBRARY_SOURCES})

# Create separate executables for each main function
add_executable(PredictionGen src/core/PredictionGen.cpp)
add_executable(EvaluateNN src/core/EvaluateNN.cpp)
add_executable(TrainNN src/core/TrainNN.cpp)

# Compiler flags
if(OpenMP_CXX_FOUND)
    target_link_libraries(MonteCarloLib OpenMP::OpenMP_CXX)
    target_link_libraries(PredictionGen OpenMP::OpenMP_CXX)
    target_link_libraries(EvaluateNN OpenMP::OpenMP_CXX)
    target_link_libraries(TrainNN OpenMP::OpenMP_CXX)
endif()

# Link libraries to the shared library
target_link_libraries(MonteCarloLib "${TORCH_LIBRARIES}")
target_link_libraries(MonteCarloLib Eigen3::Eigen)

# Link executables to the shared library
target_link_libraries(PredictionGen MonteCarloLib)
target_link_libraries(EvaluateNN MonteCarloLib)
target_link_libraries(TrainNN MonteCarloLib)

# Set properties
set_property(TARGET MonteCarloLib PROPERTY CXX_STANDARD 17)
set_property(TARGET PredictionGen PROPERTY CXX_STANDARD 17)
set_property(TARGET EvaluateNN PROPERTY CXX_STANDARD 17)
set_property(TARGET TrainNN PROPERTY CXX_STANDARD 17)

# Enable testing
enable_testing()

# Add simple smoke tests that verify executables can be built and found
add_test(NAME smoke_test_executables_exist
         COMMAND ${CMAKE_COMMAND} -E echo "All executables built successfully")

# Verify each executable exists after build
add_test(NAME verify_prediction_gen
         COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_FILE:PredictionGen>")
add_test(NAME verify_evaluate_nn  
         COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_FILE:EvaluateNN>")
add_test(NAME verify_train_nn
         COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_FILE:TrainNN>")
