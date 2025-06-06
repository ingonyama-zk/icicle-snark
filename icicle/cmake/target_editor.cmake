
# The following functions, each adds a function to a target.
# In addition, some can check feature is enabled for target, given FEATURE_LIST.

function(handle_field TARGET)
  target_sources(${TARGET} PRIVATE
      src/fields/ffi_extern.cpp
      src/vec_ops.cpp
  )
endfunction()

function(handle_curve TARGET)
  target_sources(${TARGET} PRIVATE
    src/curves/ffi_extern.cpp
    src/curves/montgomery_conversion.cpp
  )
endfunction()

function(handle_ntt TARGET FEATURE_LIST)
  if(NTT AND "NTT" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC NTT=${NTT})
    target_sources(${TARGET} PRIVATE
      src/ntt.cpp
    )
    set(NTT ON CACHE BOOL "Enable NTT feature" FORCE)
  else()
    set(NTT OFF CACHE BOOL "NTT not available for this field" FORCE)
  endif()
endfunction()

function(handle_msm TARGET FEATURE_LIST)
  if(MSM AND "MSM" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC MSM=${MSM})
    target_sources(${TARGET} PRIVATE src/msm.cpp)
    set(MSM ON CACHE BOOL "Enable MSM feature" FORCE)
  else()
    set(MSM OFF CACHE BOOL "MSM not available for this curve" FORCE)
  endif()
endfunction()

function(handle_g2 TARGET FEATURE_LIST)
  if(G2 AND "G2" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC G2_ENABLED=${G2})
    set(G2 "G2" CACHE BOOL "Enable G2 feature" FORCE)
  else()
    set(G2 OFF CACHE BOOL "G2 not available for this curve" FORCE)
  endif()
endfunction()

function(handle_pairing TARGET FEATURE_LIST)
  if(G2 AND "PAIRING" IN_LIST FEATURE_LIST)
    target_compile_definitions(${TARGET} PUBLIC PAIRING=1)
    target_sources(${TARGET} PRIVATE src/pairing.cpp)
  endif()
endfunction()