cmake_minimum_required(VERSION 3.18)

include(cmake/setup.cmake)

project(icicle)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include(cmake/field.cmake)
include(cmake/curve.cmake)

# Prevent build if both SANITIZE and CUDA_BACKEND are enabled
if(SANITIZE AND CUDA_BACKEND)
  message(FATAL_ERROR "Address sanitizer and Cuda cannot be enabled at the same time.")
endif()

# Build options
option(BUILD_TESTS "Build unit test2s. Default=OFF" OFF)
# Backends: typically CPU is built into the frontend, the rest are DSOs loaded at runtime from installation
option(CPU_BACKEND "Build CPU backend. Default=ON" ON)
# To enable building backends, use the following options: (note they are in private repos)
option(CUDA_BACKEND "Branch/commit to pull for CUDA backend or `local` if under icicle/backend/cuda. Default=OFF" OFF)
option(METAL_BACKEND "Branch/commit to pull for METAL backend or `local` if under icicle/backend/metal. Default=OFF" OFF)
option(VULKAN_BACKEND "Branch/commit to pull for VULKAN backend or `local` if under icicle/backend/vulkan. Default=OFF" OFF)

# features that some fields/curves have and some don't.
option(NTT "Build NTT" ON)
option(MSM "Build MSM" ON)
option(G2 "Build G2 MSM" ON)

# address sanitizer
if(SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# device API library
add_library(icicle_device SHARED
  src/device_api.cpp
  src/runtime.cpp
  src/config_extension.cpp
)
include_directories(include)

# Define the install directory (default is /usr/local)
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Specify the installation rules
install(TARGETS icicle_device
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if((DEFINED CURVE) AND (DEFINED FIELD))
  if(NOT ("${CURVE}" STREQUAL "${FIELD}"))
    message(FATAL_ERROR "CURVE and FIELD should not be defined at the same time unless they are equal")
  endif()
endif()

# curve is building the scalar field too
if(CURVE)
  set(CURVE_INDEX -1)
  set(FEATURES_STRING "")
  check_curve(${CURVE} CURVE_INDEX FEATURES_STRING)
  setup_curve_target(${CURVE} ${CURVE_INDEX} ${FEATURES_STRING})
  add_compile_definitions(ICICLE_FFI_PREFIX=${CURVE})
elseif(FIELD)
  set(FIELD_INDEX -1)
  set(FEATURES_STRING "")
  check_field(${FIELD} FIELD_INDEX FEATURES_STRING)
  setup_field_target(${FIELD} ${FIELD_INDEX} ${FEATURES_STRING})
  add_compile_definitions(ICICLE_FFI_PREFIX=${FIELD})
endif()

if (CPU_BACKEND)
  add_subdirectory(backend/cpu)
endif()

# Include and configure (for build) backends based on the backend options
include(cmake/backend_include.cmake)
