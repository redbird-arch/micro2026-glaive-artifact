option(GLAIVE_BUILD_TESTS "Build the optional ASTRA-Sim unit tests" OFF)

file(GLOB astra_SRC 
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc")
file(GLOB astra_test_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc")

add_library(AstraSim ${astra_SRC})
target_include_directories(AstraSim PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET AstraSim PROPERTY CXX_STANDARD 11)

# Keep the ASTRA-Sim library compatible with the legacy Boost ABI used by
# the analytical network backend.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(AstraSim PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()

if(GLAIVE_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test")
endif()
