cmake_minimum_required(VERSION 3.12)
project(srg_approx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

# Suppress make-style rule and target completion chatter
set(CMAKE_RULE_MESSAGES OFF CACHE BOOL "" FORCE)
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
set_property(GLOBAL PROPERTY TARGET_MESSAGES ON)

find_package(lcm REQUIRED)

find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(RK78 include/integrators/RK/RK78.cpp)

add_executable(drag_sys src/drag_sys.cpp)

target_include_directories(drag_sys PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(drag_sys PRIVATE 
    RK78
    lcm
)