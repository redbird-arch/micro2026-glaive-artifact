# Collective Synthesizer

This directory contains the collective-synthesis engine used by the Glaive
artifact. It reads network-topology and collective-demand descriptions,
synthesizes schedules for non-uniform All-to-All communication, and reports
the resulting events and estimated communication time.

## Upstream Project

This implementation is based on
[TACOS](https://github.com/astra-sim/tacos), the topology-aware collective
algorithm synthesizer maintained by the ASTRA-sim project.

## Glaive Extensions

Compared with the original TACOS implementation, this version adds:

- dense and sparse non-uniform All-to-All demand inputs;
- the Glaive profiling, scheduling, and schedule-fusion workflow;
- the `standard` and Speed (`mode=speed`) execution modes;
- additional direct and switched topology models used by the evaluation;
- MPICH-style, BiRing, HalfR+DR, Bruck, and Spreadout comparison methods; and
- experiment-facing event, utilization, and timing outputs used by the
  artifact scripts.

The artifact's top-level `README.md` documents the build and reproduction
workflow.

The command-line mode selection and Speed replay logic are in `src/main.cpp`;
the profiling and scheduling implementation is in
`src/synthesizer_standard/standard_synthesizer.cpp`. Evaluation traces are
listed under `evaluation_assets/collectives/` and
`evaluation_assets/manifests/`, with their CSV demand matrices under
`evaluation_assets/csv/` or `input/`.
