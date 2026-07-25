## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2022-2025 Intel Corporation
## Copyright (c) 2022-2025 Georgia Institute of Technology
## ******************************************************************************

# TACOS library
add_library(tacos
    topology/topology.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/topology.h

    topology/mesh.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/mesh.h
    topology/torus.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/torus.h
    topology/fullmesh.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/fullmesh.h
    topology/hypercube.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/hypercube.h
    topology/fattree.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/fattree.h
    topology/railoptimized.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/railoptimized.h
    topology/cm.cpp ${CMAKE_SOURCE_DIR}/include/tacos/topology/cm.h

    collective/collective.cpp ${CMAKE_SOURCE_DIR}/include/tacos/collective/collective.h
    collective/all_gather.cpp ${CMAKE_SOURCE_DIR}/include/tacos/collective/all_gather.h
    collective/all_to_all.cpp ${CMAKE_SOURCE_DIR}/include/tacos/collective/all_to_all.h
    collective/all_to_all_v.cpp ${CMAKE_SOURCE_DIR}/include/tacos/collective/all_to_all_v.h
    collective/gather.cpp ${CMAKE_SOURCE_DIR}/include/tacos/collective/gather.h
    event_queue/event_queue.cpp ${CMAKE_SOURCE_DIR}/include/tacos/event_queue/event_queue.h
    event_queue/timer.cpp ${CMAKE_SOURCE_DIR}/include/tacos/event_queue/timer.h
    synthesizer/synthesizer.cpp synthesizer/synthesizer.h
    synthesizer_standard/standard_synthesizer.cpp synthesizer_standard/standard_synthesizer.h
    baselines/baseline_solver.cpp ${CMAKE_SOURCE_DIR}/include/tacos/baselines/baseline_solver.h
    baselines/bruck.cpp ${CMAKE_SOURCE_DIR}/include/tacos/baselines/bruck.h
    baselines/biring.cpp ${CMAKE_SOURCE_DIR}/include/tacos/baselines/biring.h
    baselines/spreadout.cpp ${CMAKE_SOURCE_DIR}/include/tacos/baselines/spreadout.h
    baselines/pairwise.cpp ${CMAKE_SOURCE_DIR}/include/tacos/baselines/pairwise.h
    baselines/others.cpp ${CMAKE_SOURCE_DIR}/include/tacos/baselines/others.h
    thread_output.cpp ${CMAKE_SOURCE_DIR}/include/tacos/thread_output.h
)
target_include_directories(tacos
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PUBLIC ${CMAKE_SOURCE_DIR}/include/tacos
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(tacos PRIVATE Threads::Threads)

# TACOS executable
add_executable(tacos_exec
    main.cpp
)
target_include_directories(tacos_exec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tacos_exec PRIVATE tacos Threads::Threads)
set_target_properties(tacos_exec PROPERTIES OUTPUT_NAME tacos)
