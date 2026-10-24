# H1-Sup publication validation — 2026-10-24

Validation was run against the isolated publication checkout at
`/data2/mzy/GAD/CAFE_codebase/outputs/github_h1_sup_20261024`.
The trainer, model, data loader and evaluator were not changed by this audit.

## Result

**59 tests ran: 58 passed, one skipped, zero failures and zero errors.**
The suite completed in 4.237 seconds. It exercised CPU model forward/backward
and policy fixtures; it did not start training or evaluate the full dataset.

| Test module | Result |
| --- | --- |
| `tests/heatmap/test_contract.py` | 16 passed, one CUDA test skipped; two historical-reference tests excluded |
| `tests/heatmap/test_crowd_targets.py` | 9 passed |
| `tests/sam_person/test_evaluator.py` | 4 passed |
| `tests/sam_person/test_view_training_contract.py` | 8 passed |
| `tests/sam_person/test_random_frame_cache.py` | 2 passed |
| `tests/sam_person/test_image_io_retry.py` | 5 passed |
| `tests/sam_person/test_view_final_queue.py` | 6 passed |
| `tests/sam_encoder/test_final_selection.py` | 8 passed |

Coverage:

- H1 and B0 common initialization, preserved RNG state and identical initial predictions.
- Heatmap geometry, resize/flip behavior, actor/slot permutation, empty ground truth,
  missing frames, finite losses and gradients, and auxiliary supervision behavior.
- Exact regression of the standard `box` target and the explicit `crowd` variant.
- Numeric confidence ranking, evaluation clip restrictions and missing-prediction denominators.
- Matched place/view training configuration and rejection of evaluator/configuration drift.
- Random temporal-segment sampling, cache coverage and bounded image I/O retries.
- View TEST evaluation after every epoch, primary-metric checkpoint selection,
  secondary metrics from the same epoch, first-epoch tie handling and checkpoint retention.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| NumPy | 1.24.3 |
| SciPy | 1.10.1 |
| Pillow | 9.5.0 |

The cached ImageNet ResNet18 weights were read from the source workspace's
`TORCH_HOME`; no model weights are included in this branch. CUDA was unavailable
to the validation process. Deprecation warnings for existing torchvision and
`torch.testing.assert_allclose` calls did not affect the results.

## Exact command

Run from the publication checkout's root. The following records the exact local
interpreter and cached-weight path used for this validation; other machines must
substitute their own interpreter and `TORCH_HOME`.

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/data2/mzy/GAD/CAFE_codebase/.worktrees/cafe-sam-person-mask-20261018/.local_data/torch /data2/mzy/GAD/.miniconda3/envs/gad/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import sys
import unittest
root=Path.cwd()
sys.path.insert(0,str(root))
files=['tests/heatmap/test_contract.py','tests/heatmap/test_crowd_targets.py','tests/sam_person/test_evaluator.py','tests/sam_person/test_view_training_contract.py','tests/sam_person/test_random_frame_cache.py','tests/sam_person/test_image_io_retry.py','tests/sam_person/test_view_final_queue.py','tests/sam_encoder/test_final_selection.py']
excluded={'test_disabled_exact_original_revision','test_export_original_decoder_and_evaluator_roundtrip'}
suite=unittest.TestSuite()
for index,rel in enumerate(files):
 spec=importlib.util.spec_from_file_location('publish_validation_'+str(index),str(root/rel))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 def collect(tests):
  for test in tests:
   if isinstance(test,unittest.TestSuite):collect(test)
   elif test._testMethodName not in excluded:suite.addTest(test)
 collect(unittest.defaultTestLoader.loadTestsFromModule(module))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('PUBLISH_VALIDATION',{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'excluded_historical_git_reference_tests':sorted(excluded)})
sys.exit(not result.wasSuccessful())
PY
```

Output:

```text
PUBLISH_VALIDATION {'run': 59, 'failures': 0, 'errors': 0, 'skipped': 1,
 'excluded_historical_git_reference_tests': ['test_disabled_exact_original_revision',
 'test_export_original_decoder_and_evaluator_roundtrip']}
Ran 59 tests in 4.237s
OK (skipped=1)
```

## Skips and scope limits

- `ModelContract.test_cuda_rng_unchanged_by_new_head` was skipped by its CUDA
  availability guard: `CUDA unavailable in sandbox; run on allocated host GPU`.
- Two historical-reference tests in `tests/heatmap/test_contract.py` were excluded:
  `test_disabled_exact_original_revision` reads `exp/b0-repro:models/models.py`,
  and `test_export_original_decoder_and_evaluator_roundtrip` reads a historical
  `HEAD:train.py` decoder. Those historical references are not the publication
  checkout's active trainer and were not substituted or rewritten for this audit.
- `tests/sam_person/test_view_best_on_test_policy.py` requires an ignored local
  `.local_data/view_final_20261023/source_snapshot` tree.
- `tests/sam_person/test_view_final_contract.py` requires original local reference
  args and an ignored historical trainer snapshot under `.local_data/sam/`.
- `tests/sam_person/test_view_final_resume.py` imports a frozen queue from the
  ignored `.local_data/view_final_20261023/source_snapshot` tree.
  These three historical test modules were not run. The active trainer's
  selection and retention policy was tested independently by
  `tests/sam_encoder/test_final_selection.py`, and the portable queue fixtures
  were tested by `tests/sam_person/test_view_final_queue.py`.

Dataset files, tracklets, training checkpoints, SAM caches and third-party model
weights remain external inputs. Standard ResNet18 H1-Sup with
`train_mask_mode=disabled` does not require a Segment Anything checkout or SAM
weights. This audit does not claim GPU execution, full-training reproducibility
or a newly measured dataset score.

## Source integrity

Across `models/`, `util/`, `dataloader/`, `evaluation/`, `train.py` and `test.py`,
34 original files were compared byte for byte with the latest source workspace
and matched with no missing files. The publication source manifest
records the broader original-source hash audit.

The corrected evaluator SHA256 is:

```text
f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0
```
