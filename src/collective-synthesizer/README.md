# Collective Synthesizer Development Log 

**Author:** Le Qin (Galois)  
**Affiliation:** HKUST(GZ)

## Introduction
This repo is based on TACOS and targets efficient, robust, fast collective communication synthesizer.

## Section 1: Getting started

### 1. Dependencies 
- Clone the repo with all the dependencies<br>
```bash
# clone the repo
git clone  --recurse-submodules git@github.com:G-alois/Collective-Synthesizer.git
# if submodules update fails, consider mannually pulling the submodules
git submodule add https://github.com/jbeder/yaml-cpp.git libs/yaml-cpp
git submodule add https://github.com/nlohmann/json libs/nlohmann_json
git submodule update --init --recursive
# check the submodule status
git submodule status
``` 

- We solve the dependencies in NICE server via docker<br>
```bash
# pull existing docker image
docker pull astrasim/tacos:latest
# instead, you may consider building this Docker Image locally.
./utils/build_docker_image.sh
```

### 2. Run the code 
- Start the Docker Container (which becomes the TACOS runtime environment)<br>
```bash
# the first time to run the container
./utils/start_docker_container.sh
# next time: enter the container
docker exec -it container_name /bin/bash
# example
docker exec -it 6715f44727bc /bin/bash
```

- Run TACOS with the provided script<br>
```bash
# complete process (compile and run)
[docker] ./tacos.sh
# just run with specified config
[docker] ./tacos.sh run input/topology/mesh2d_1.json input/collective/allgather_1.json
[docker] ./tacos.sh run input/topology/mesh2d_1.json input/collective/allgather_1.json > results/test.log 2>&1
# 11.13 update: enable multi-thread function for collective solving
# open multi-thread (true + appoint the output file path):
[docker] ./tacos.sh run input/topology/mesh2d_1.json input/collective/allgather_1.json true results/Thread_64
# with single-thread (false + appoint output file path and name)
[docker] ./tacos.sh run input/topology/mesh2d_1.json input/collective/allgather_1.json false
# for the test and debug of new-enabled Alltoall and Gather, we utilize following test script
./tacos.sh run input/topology/mesh2d_1.json input/collective/alltoall_1.json false
# for the test and debug of new-enabled network faults (device/link), we utilize following test script
./tacos.sh run input/topology/mesh3_LF.json input/collective/alltoall_1.json > results/alltoall_LF.log false
# for the test and debug of new-enabled alltoallv, we utilize following test script
./tacos.sh run input/topology/mesh2d_1.json input/collective/alltoallv.json > results/alltoallv.log false
# 1.6 update: for the test and debug of new-enabled Synthesizer2 (unified solver for non-uniform AllToAll)
# basic solver2 usage (requires v_datasize in collective config, supports mesh and torus topologies)
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 > results/FireAnt.log 2>&1
# solver2 with comparison mode (compare different routing strategies)
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 --compare
# solver2 with disabled diffusion preprocessing
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 --no-diffusion
# solver2 with quiet mode (disable verbose output)
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 --quiet
# solver2 with detailed schedule printing
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 --print-schedule
# solver2 with forced routing strategy (dimension_order, load_balanced, multi_path, adaptive)
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 --strategy dimension_order
# run all the baselines of given collective + topology
[docker] ./build/bin/tacos input/topology/mesh2d_1.json input/collective/alltoallv.json --baselines > results/alltoallv_9_baselines.log 2>&1
# run baselines for case2
[docker] ./build/bin/tacos input/topology/FireAnt_mesh8.json input/collective/alltoallv_64_case2.json --baselines > results/alltoallv_64_case2_baselines.log 2>&1
# run synthesizer4 for fat-tree
[docker] ./build/bin/tacos input/topology/fat-tree_2level.json input/collective/alltoallv_64_decode_bs64.json --solver4 > results/fat-tree_alltoallv_64_decode_bs64_synthesizer4.log 2>&1
# new test for alltoallv using synthesizer4 (for CM384 synthesizer)
[docker] ./build/bin/tacos input/topology/cm384_2node.json input/collective/alltoallv512_32devices.json --solver4 > results/alltoallv_512_cm384_32npu_synthesizer4.log 2>&1
```

### 3. Misc 
- Common docker instructions<br>
```bash
# check all the images
docker images
# search image
docker search xxxxx
# pull image
docker pull xxxxx:xxx
# delete image
docker rmi xxxxx:xxx
# build image
docker build -t xxxxx:xxx .
```

```bash
# check running docker containers
docker ps
docker ps -a
# start a container
docker run -d -p 8080:80 --name container_name nginx
# enter the container
docker exec -it container_name /bin/bash
# example
docker exec -it 7969162b4489 /bin/bash
docker start -ai TACOS
# stop the container
docker stop container_name
# start a stopped container
docker start container_name
# remove the container
docker rm container_name
```


## Section 2: Working Progress
- 5.9, 5.13: Initialize the repo, confirm that both the GitHub repo and AE Zenodo don't have reproducible process (i.e., we couldn't reproduce the results using existing scripts or code)
- 5.15 - 5.16: Find the missing chunk number as a variable. which could have a huge influence on collective solving time and synthesized performance. I refactor and separate the mesh.cc as an outside topology input file which could reduce the time consumption of frequent compilation. I test the results and have a basic findings (and visualization) of synthesized performance. 
- 5.22: Separate the collective input files, add new topology support (torus, fullmesh, hypercube), and new collective support (alltoall).
- 9.22: Go back to this project. Since TACOS update their GitHub Repo, I have simple tests, finding the results much more reasonable now. Here are some points need further exploration: 
  - a) a proper way to find an optimal chunk number for the tradeoff between solution quality and solver speed 
  - b) consider a method to synthesize All-Reduce with the overlapping of Reduce-Scatter and All-Gather 
  - c) rethinking the method tradeoff between a per-step link allocation way (e.g., MultiTree, TACOS) and a lifetime link allocation way (e.g., TTO); the basic thought: the former has higher solving complexity but can be better, the latter is easier.
- 9.24: Finish the code review, and I found that the strange performance of old-tacos repo comes from the missing of backtracking. I start to separate collective/topology input from original fixed main.cpp file.
- 10.27: I am back to this project, today I finished the function that enable the collective/topology files with json format as the program input. Next things need to be done: <br>
  -  a) current results are still wrong, at [EventTime 6159375 ps] Chunk 6 & 12 use the same 4 -> 0 link. Multi-chunk scenarios work in function level, but the results are still wrong. I plan to figure out the reason, and add a visualization tool to draw the timeline figures according to the output log. <span style="color:#118ab2;">Because it's the old version TACOS.</span>
  -  b) I plan to check the former TACOS version, and add support for XML generator. I still need to check whether there are still some problems lie in the current/former code base, and why the performance varies. Based on my current understanding, this repo (collective synthesizer) corresponds with current TACOS version, and TACOS repo corresponds with former TACOS version. 
  -   c) start to research how to support dynamic chunk number selection (and in which scenario will it work) 
  -   d) we need to enable heterogeneous/faulty networks; a question: whether 2Dmesh-AllGather just needs timesteps of `diameter` 
  -   e) a new target: irregular communication pattern including fault-tolerant cases, Alltoallv, M2MS, which haven't been researched before.
- 10.29: actually collective-synthesizer repo is currently the `old version`, while TACOS repo is the `new version`, and I am working on enable the repo with new version. Today I have transfer all the former code of this repo (`TACOS Old`) into the `TACOS_V1` branch, and start a new `main` branch with `TACOS New` code. I have sucessfully run the new code with json-format input files. It's interesting that current repo cannot enable a optimal allgather performance on mesh, I will start to look into the reason. 
- 10.30: start to enable the print function of link allocation at each time step, visualization
- 11.20: finish the multi-threading synthesizer and tests. The solver time consumption roughly corresponds with the $O(N^2)$ relationship with nodes' number. Our former idea of constructing multiple chunks instead of single chunk has quite limited optimization space. The RS-AG overlapping in other topologies can be explored while in mesh it's well researched. Let's focus on the non-uniform collective and fault-tolerance firstly.
- 12.24: evolve the solver by multiple iterations, though haven't figured out an optimal solution for All-Gather and All-to-All (sub-optimal transfer time), trying to improve algorithms by learning some path-finding algorithms. support link/node (along with affiliated links) failure (topology file), irregular collective (collective file), waiting for support pure node failure and straggler. plan to add modeling for switch (basic thought: switch is a node just for transfer)
- 1.4: enable solver 2 with profile-routing-scheduling workflow, can get rather good performance under uniform workload while essentially the complexity is still high, calling for better algorithm workflow. Besides, we find the necessity of dynamic chunk number allocation mechanism, and find out the collective matrix decomposition to BW-bound and LAT-bound matrices and solve them separately.  
- 1.13: enable bruck, pairwise and spreadout baseline algorithms for All-to-All, basically our methods outperform them in general cases (be aware of straggler and non-uniform cases, but cannot find other routings under faults).
- 1.14: need to consider the modeling of switch topologies, and see whether we should improve the profiling-routing-scheduling framework without backtracking.
TBTest(Diffusion):
/app/tacos# ./build/bin/tacos input/topology/FireAnt_mesh8.json input/collective/alltoallv_64_case2.json --baselines > results/FireAnt_64_case2.log 2>&1
- 1.21: finish solver 3 (last week, about matrix decomposition to BW-MAT and LAT-MAT with separate solving strategy) and solver 4 (today, about matrix decomposition to HotSpot-MAT and Regular-MAT with separate solving strategy). I find that former diffusion model is totally wrong (it's a non-equivalent transformation which needs actual link scheduling for data transmission, while our former implementation doesn't consider this). Current focus is how to deal with the multiple hotspot scattering problem.
- 1.29: finish the modeling of switch-based topologies, TBD: current fat-tree and rail-optimized is different from the reality (the first level scale-out interconnection should be Pod-level and the higher-dim switches be global level)



## Section 3: Code Review (new repo version)

### {collective}
- all_gather.cpp/.h
  > `npusCount`: number of NPUs in the topo<br>
  > `collectivesCount`: number of initial chunks per NPU<br>

  > `AllGather()`: register `dests` and `chunks`<br>

- collective.cpp/.h
  > `precondition_`: the mapping relationship of precondition: `<ChunkID, NpuID>` 
  > `postcondition_`: the mapping relationship of postcondition: `<ChunkID, <NpuID>>` 
  > `chunksCount_`: the number of total chunks in the collective
  > `newChunkID_`: new chunk need inserting

  > `precondition()`: return source GPU for a given chunk 
  > `postcondition()`: return the destination NPUs for a given chunk
  > `chunksCount()`: return the number of chunks `chunksCount_` in the collective
  > `chunk_()`: insert new pre/post conditions for a chunk
  


### {event_queue}
- timer.cpp/.h <br>
  <!-- A HTML format code illustration -->

  <!-- <blockquote>
    start(): update <code>startTime</code> <br>
    stop(): update `stopTime` (current time) <br>
    time(): compute `elaspedTime` <br>
  </blockquote> -->

  > `start()`: update `startTime`  
  > `stop()`: update `stopTime` (current time)  
  > `time()`: compute `elaspedTime`

- event_queue.cpp/.h <br>
  > `events_`: set of event times <br>
  > `event_queue_`: priority queue of event times (result of `events_` after sorting, the smaller, the topper)<br>
  
  > `schedule()`: insert current time to `events_` & `event_queue_` unless current time exists<br>
  > `pop()`: pop/erase the top element in `events_` & `event_queue_`, update currentTime with the element<br>
  > `reset()`: clear the `events_` & `event_queue_` and `schedule` time 0<br>
  > `empty()`


### {synthesizer}
- time_expanded_network.cpp/.h <br>
  > `linkBusyUntil_`: `linkBusyUntil_[i][j]` indicates the `Time` until when the link between NPU-i and NPU-j is available<br>
  > `chunk_`: `chunk_[i][j]` indicates the `chunkID` being transferred using link between NPU-i and NPU-j<br>
  > `available_`: bool value that indicates whether the link between NPU-i and NPU-j is available<br>
  > `linkTransferTimes_`: link transfer time for given chunk using alpha-beta model (ms)<br>

  > `linkTransferTime()`: return `linkTransferTimes_[src][dest]` for given src and dest<br>
  > `chunk()`: return `chunk_[src][dest]` for given src and dest<br>
  > `available()`: return `available_[src][dest]` for given src and dest<br>
  > `disable()`: mark the link from src to dest as unavailable<br>
  > `TimeExpandedNetwork()`: initialize `available_`, `linkBusyUntil_`, `chunk_`, and `linkTransferTimes_`; and then `computeLinkTimes_` for given chunkSize<br>
  > `alphaBetaModel_()`: compute the data transmission time for given `chunkSize`<br> 
  > `computeLinkTimes_()`: update the data transmission time `linkTransferTimes_[src][dest]` for all the links using alpha-beta model<br> 
  > `transferFinished()`: reset the `available_`, `linkBusyUntil_` and `chunk_` for given src and dest after data transmission<br>
  > `transferChunk()`: update `available_`, `linkBusyUntil_` and `chunk_` for chunk transmission<br>
  > `timestep()`: update the availability `available_` of all links for current time<br>
  > `backtrack()`: get the list of available source NPUs to dest from topology backtracking<br>

- synthesizer.cpp/.h<br>
  > `chunkMap_`: chunkMap_[c][n] indicates whether chunk c has arrived at NPU n (bool value)<br>

  > `solve()`: `initialize_()`, `markPrecondition_()`, then repeat the link-chunk matching process as long as `!eventQueue_.empty()`. In the while loop, first get current time, then filter out unsatisfied postcondition `filterPostcondition_()`, then expand the TEN `expandTenTimestep_()`, then check if there are unsatisfactory postconditions `shufflePostcondition_()`, do link matching for all unsatisfied postconditions, then proceed to the next event when `postcondition.empty()`, finally return `collectiveTime_` when all the matchings finish.<br>
  > `initialize_()`: reset `eventQueue_`, `ten_`, and `chunkMap_`; derive some variables<br>
  > `markPrecondition_()`: update `chunkMap_` with initially arrived chunks<br>
  > `filterPostcondition_()`: return `postconditionMap[dest][chunk]` indicates all the chunks that haven't arrived at corresponding dest NPUs<br>
  > `expandTenTimestep_()`: first expand the TEN `timestep(currentTime_)`, traverse for all src-dest pairs and when the link is available, check if we can replace the path with others, after that update the link state by running `transferFinished()`. Finally, update `chunkMap_[chunk][dest]`, `postconditionMap`, and `collectiveTime_`<br>
  > `findReplacementChunk_()`: for all unsatisfied postcondition of given dest NPU, if the chunk is available at the `src` while not at `dest`, randomly find a this-kind-of chunk to replace former found chunk.<br>
  > `shufflePostcondition_()`: change from `std::unordered_map<NpuID, std::unordered_set<ChunkID>>` format to `std::vector<std::pair<ChunkID, NpuID>>`, and return the changed format after shuffle<br>
  > `linkChunkMatching_()`: match links for the unsatisfied chunks of current `dest` with shuffle operation<br>
  > `isEqual()`: check whether two given time value is equal and return the bool value<br> 

### {topology}
- topology.cpp/.h (the base class of other topologies)
  > `backtrackMap_`: set of NPUs that can send a chunk to given `dest` (topology connected)

  > `bandwidth()`: return bandwidth for given `src` & `dest` pair<br>
  > `latency()`: return latency for given `src` & `dest` pair<br>
  > `connected()`: return connection relationship for given `src` & `dest` pair<br>
  > `npusCount()`: return `npusCount_`<br>
  > `backtrack()`: return `backtrackMap_.at(dest)`<br>
  > `setNpusCount_()`: initialize `npusCount_`, `connected_`, `latencies_`, `bandwidths_`, and `backtrackMap_`<br>
  > `connect_()`: record all the related params for give `src` & `dest` pair<br>