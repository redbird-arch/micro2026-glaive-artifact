# DeepSeek v3.2 Case Generation Report

## Prefill Input
- Source dir: `/Users/galois/Coding/DeepSeek_v3.2_P`
- Layers: 58
- Declared rows per layer: 4763
- Usable rows per layer: 4762
- Skipped first row: True
- Synthetic BS=1 groups: 19
- Partial rows in last synthetic BS=1: 154

## Decode Input
- Source dir: `/Users/galois/Coding/DeepSeek_v3.2_P+D`
- Layers: 58
- Declared rows per layer: 9564
- Usable rows per layer: 9564
- Synthetic BS=1 groups: 38
- Partial rows in last synthetic BS=1: 92

## Output Naming
- Output filenames preserve the source layer ids, e.g. `Layer0` .. `Layer57`.
- Only the final covering batch size may emit `actual<N>` files.

## Prefill Batches
- BS=1: full_batches=18, partial_actual_groups=[], files_written=1044
- BS=2: full_batches=9, partial_actual_groups=[], files_written=522
- BS=4: full_batches=4, partial_actual_groups=[], files_written=232
- BS=8: full_batches=2, partial_actual_groups=[], files_written=116
- BS=16: full_batches=1, partial_actual_groups=[], files_written=58
- BS=32: full_batches=0, partial_actual_groups=[19], files_written=58

## Decode Batches
- BS=8: full_batches=4, partial_actual_groups=[], files_written=232
- BS=16: full_batches=2, partial_actual_groups=[], files_written=116
- BS=32: full_batches=1, partial_actual_groups=[], files_written=58
- BS=64: full_batches=0, partial_actual_groups=[38], files_written=58
