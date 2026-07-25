# Setup project
project (AnalyticalAstra)

# Package requirement
# Include src files to compile
file(GLOB_RECURSE srcs
        "${PROJECT_SOURCE_DIR}/src/*.hh"
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        )

# Compile sources
add_executable(AnalyticalAstra ${srcs})

# Keep the legacy Boost ABI setting local to the analytical frontend.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(AnalyticalAstra PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

# Link libraries
target_link_libraries(AnalyticalAstra LINK_PUBLIC AstraSim)

# Resulting binary location settings
set_target_properties(AnalyticalAstra
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "bin/"
        LIBRARY_OUTPUT_DIRECTORY "lib/"
        ARCHIVE_OUTPUT_DIRECTORY "lib/"
        )
