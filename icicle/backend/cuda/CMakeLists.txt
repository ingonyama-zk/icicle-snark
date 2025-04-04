cmake_minimum_required(VERSION 3.18)

include(cmake/Common.cmake)
find_cuda_compiler()

# features that some fields/curves have and some don't.
option(NTT "Build NTT" ON)
option(MSM "Build MSM" ON)
option(G2 "Build G2 MSM" ON)

project(icicle_cuda_backend LANGUAGES CUDA CXX)

set_env()
set_gpu_env()

find_package(CUDAToolkit REQUIRED)

# device API library
add_library(icicle_backend_cuda_device SHARED src/cuda_device_api.cu)
target_include_directories(icicle_backend_cuda_device PRIVATE include)
target_include_directories(icicle_backend_cuda_device PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_directories(icicle_backend_cuda_device PRIVATE 3rdparty/lib/linux) # assuming only linux for CUDA backend for now
target_link_libraries(icicle_backend_cuda_device PUBLIC pthread)

install(TARGETS icicle_backend_cuda_device
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend/cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend/cuda
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend/cuda)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET icicle_backend_cuda_device POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:icicle_backend_cuda_device>
    )
endif()

# field API library
if (FIELD)
    add_library(icicle_cuda_field SHARED
        src/field/cuda_vec_ops.cu
        src/field/cuda_mont.cu
        )
    if(NTT)
        target_sources(icicle_cuda_field PRIVATE
            src/ntt/cuda_ntt.cu
            src/ntt/mixed_radix_ntt.cu
        )
    endif()
    target_include_directories(icicle_cuda_field PRIVATE include)
    set_target_properties(icicle_cuda_field PROPERTIES OUTPUT_NAME "icicle_backend_cuda_field_${FIELD}")
    target_include_directories(icicle_cuda_field PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(icicle_cuda_field PRIVATE icicle_field ${CUDA_LIBRARIES} pthread) # Link to CUDA

    install(TARGETS icicle_cuda_field
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${FIELD}/cuda"
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${FIELD}/cuda"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${FIELD}/cuda")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_custom_command(TARGET icicle_cuda_field POST_BUILD
            COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:icicle_cuda_field>
        )
    endif()
endif() # FIELD

# curve API library
if (CURVE)
    add_library(icicle_cuda_curve SHARED        
        src/curve/cuda_mont.cu
    )
    if(MSM)
        target_sources(icicle_cuda_curve PRIVATE src/msm/cuda_msm.cu)
    endif()
    if(G2)
        target_sources(icicle_cuda_curve PRIVATE src/msm/cuda_msm_g2.cu)
    endif()
    target_include_directories(icicle_cuda_curve PRIVATE include)
    set_target_properties(icicle_cuda_curve PROPERTIES OUTPUT_NAME "icicle_backend_cuda_curve_${CURVE}")
    target_include_directories(icicle_cuda_curve PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(icicle_cuda_curve PRIVATE icicle_cuda_field icicle_curve ${CUDA_LIBRARIES} pthread) # Link to CUDA

    install(TARGETS icicle_cuda_curve
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${CURVE}/cuda"
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${CURVE}/cuda"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${CURVE}/cuda")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_custom_command(TARGET icicle_cuda_curve POST_BUILD
            COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:icicle_cuda_curve>
        )
    endif()
endif()