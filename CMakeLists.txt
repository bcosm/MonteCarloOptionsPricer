cmake_minimum_required(VERSION 3.12)
project(MonteCarloOptionsPricer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Torch REQUIRED)
find_package(OpenMP)

# Include directories
include_directories(include)
include_directories(src/core)
include_directories(src/models)

# Source files
file(GLOB_RECURSE SOURCES 
    "src/core/*.cpp"
    "src/models/*.cpp"
)

# Create executable
add_executable(MonteCarloOptionsPricer ${SOURCES})

# Compiler flags
if(OpenMP_CXX_FOUND)
    target_link_libraries(MonteCarloOptionsPricer OpenMP::OpenMP_CXX)
endif()

# Link libraries
target_link_libraries(MonteCarloOptionsPricer "${TORCH_LIBRARIES}")

# Set properties
set_property(TARGET MonteCarloOptionsPricer PROPERTY CXX_STANDARD 17)
