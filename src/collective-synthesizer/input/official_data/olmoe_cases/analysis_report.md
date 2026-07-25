# OLMoE Batch Regeneration Report

## Raw Trace
- Source file: `/Users/galois/Downloads/olmoe_dpep64_globalbs64.txt`
- Shapes: {'64x64': 78272}
- Prefill lines/groups: 336 / 21
- Decode lines/groups: 77936 / 4871
- Prefill run lengths: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

## Input Check
- Truncated file exists: True
- Truncated file is prefix of main file: True

## Old Logic Check
- Prefill old BS1024 matches long file only: True
- Decode old BS4096 matches long file only: False
- Decode issue: The old decode regeneration used both the truncated file and the complete file, which duplicates the shared prefix.
- Prefill issue: The raw prefill matrices are already 64x64 BS=1 samples, so naming them as BS64/128/... is semantically misleading.

## Outputs
### Prefill
- BS=1: full_batches=21, partial_actual_groups=[], files_written=336
- BS=2: full_batches=10, partial_actual_groups=[], files_written=160
- BS=4: full_batches=5, partial_actual_groups=[], files_written=80
- BS=8: full_batches=2, partial_actual_groups=[], files_written=32
- BS=16: full_batches=1, partial_actual_groups=[], files_written=16
- BS=32: full_batches=0, partial_actual_groups=[21], files_written=16

### Decode
- BS=8: full_batches=608, partial_actual_groups=[], files_written=9728
- BS=16: full_batches=304, partial_actual_groups=[], files_written=4864
- BS=32: full_batches=152, partial_actual_groups=[], files_written=2432
- BS=64: full_batches=76, partial_actual_groups=[], files_written=1216
- BS=128: full_batches=38, partial_actual_groups=[], files_written=608
- BS=256: full_batches=19, partial_actual_groups=[], files_written=304
- BS=512: full_batches=9, partial_actual_groups=[], files_written=144
- BS=1024: full_batches=4, partial_actual_groups=[], files_written=64
- BS=2048: full_batches=2, partial_actual_groups=[], files_written=32
- BS=4096: full_batches=1, partial_actual_groups=[], files_written=16
- BS=8192: full_batches=0, partial_actual_groups=[4871], files_written=16
