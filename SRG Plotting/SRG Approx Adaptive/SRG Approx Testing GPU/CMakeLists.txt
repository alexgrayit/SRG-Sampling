cmake_minimum_required(VERSION 3.18)   # 3.18+ for proper CUDA language support
project(srg_approx CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Suppress noisy build output
set(CMAKE_RULE_MESSAGES OFF CACHE BOOL "" FORCE)
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

# ---------- Dependencies ----------
find_package(lcm REQUIRED)
find_package(Eigen3 3.4 QUIET)

# Eigen is only needed by the original CPU simulator.
if(Eigen3_FOUND)
    add_library(RK78 include/integrators/RK/RK78.cpp)
    add_executable(drag_sys src/drag_sys.cpp)
    target_include_directories(drag_sys PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(drag_sys PRIVATE RK78 lcm Eigen3::Eigen)
endif()

# ---------- CUDA-accelerated target ----------
# Detect the GPU architecture automatically, or set for Ada Lovelace (RTX 4080 Super)
# sm_89 = Ada Lovelace (RTX 40-series)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(drag_sys_cuda src/drag_sys_cuda.cu)
target_include_directories(drag_sys_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(drag_sys_cuda PRIVATE lcm)

option(USE_LSODA "Use CUDA LSODA instead of CUDA RKF7(8)" OFF)
option(SRG_LINEAR_DRAG "Use the linear mass-spring-damper model" OFF)
option(SRG_ENABLE_FRICTION "Add regularised Stribeck sliding friction" OFF)
if(USE_LSODA)
    target_compile_definitions(drag_sys_cuda PRIVATE USE_LSODA=1)
endif()
if(SRG_LINEAR_DRAG)
    target_compile_definitions(drag_sys_cuda PRIVATE SRG_LINEAR_DRAG=1)
endif()
if(SRG_ENABLE_FRICTION)
    target_compile_definitions(drag_sys_cuda PRIVATE SRG_ENABLE_FRICTION=1)
endif()

# Ensure nvcc passes -O3 for release-level performance
# CUDA compiler optimisation and resource diagnostics
target_compile_options(drag_sys_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --ptxas-options=-v
        --maxrregcount=128
        -lineinfo
        # -Ddouble=float
    >
)
