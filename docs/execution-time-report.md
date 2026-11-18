# Execution time report

Rewritten by every run of the program (`shelfvision.timing.write_report`); do not edit by hand.

| Field | Value |
|---|---|
| Date | 2026-11-18T21:57:58+00:00 |
| Total time | 956.1 s, since `shelfvision.timing` was imported |
| OS | Windows-11-10.0.26300-SP0 |
| CPU | AMD Ryzen 5 7520U with Radeon Graphics, 8 logical cores, 8 torch threads |
| RAM | 15.3 GiB |
| GPU | none seen by torch |

## Stages

Time is the mean of the runs, one run per photo for the model stages. Difference is against the
README's figure, measured under the conditions listed below the table.

| Stage | Time (s) | Runs | README claim | Difference (%) |
|---|---:|---:|---:|---:|
| load image | 0.138 | 7 | - | - |
| encode image (OWLv2) | 14.791 | 7 | 13.4 s | +10.4 |
| find requested objects | 0.516 | 6 | - | - |
| find all objects | 0.189 | 7 | - | - |
| read barcodes | 109.950 | 7 | 104.0 s | +5.7 |
| estimate normals (MoGe) | 6.854 | 7 | 6.0 s | +14.2 |
| same queries again | 0.102 | 1 | 0.102 s | +0.4 |
| all stages of one photo | 132.364 | 7 | 124.0 s | +6.7 |
| encode image (SAM) | - | 0 | 10.2 s | not run |
| segment objects (SAM) | - | 0 | 3.7 s | not run |

README conditions:

- encode image (OWLv2): notebook run of 2026-11-18, 8 torch threads, mean of the 7 photos
- read barcodes: notebook run of 2026-11-18, 8 torch threads, mean of the 7 photos
- estimate normals (MoGe): notebook run of 2026-11-18, 8 torch threads, mean of the 7 photos
- same queries again: notebook run of 2026-11-18, input_2.jpg, one measurement
- all stages of one photo: notebook run of 2026-11-18, 8 torch threads, mean of the 7 photos
- encode image (SAM): benchmark run beside others, 8 torch threads; the notebook runs SAM only with SEGMENT = True
- segment objects (SAM): benchmark run beside others, 8 torch threads; the notebook runs SAM only with SEGMENT = True
