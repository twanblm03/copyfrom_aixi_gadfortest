# Research source publication verification — 2026-10-12

Publication branch: `exp/cafe-heatmap`, based on main
`0ffe57666cb972ac6dc2b9593df556045b727e44`. Obtain the final publication commit
with `git rev-parse HEAD`; a document cannot contain its own final commit hash.
The frozen reference remains `exp/b0-repro` at
`9e6a83f5374f2d5a61a60ce3f44d17a2b55c75bc` (also verified against origin).

## Scope and clean-tree validation

Only explicitly selected heatmap/B0 modules, entry-point integration, configs,
tests, runners, documentation and ignore rules were staged. Original, unrelated
untracked DINO files, `.ignore` and third-party files remain on the original
machine and were not deleted or added to this commit. Environment, Dataset,
results, logs and checkpoints are not newly published. Already tracked upstream
files, including its `.local_deps` contents, are preserved rather than removed
as an unrelated cleanup.

The staged Git tree was exported to a new empty temporary directory using
`git checkout-index --all --prefix=...`. It contains only files present in the
index, not the original working tree's untracked helper code. Validation used
the existing verified Python environment, an empty `PYTHONPATH` and disabled
CUDA so publishing did not allocate GPUs used by the ongoing pilot.
`GIT_DIR` pointed to the original repository solely so the contract test could
read the frozen reference via `git show`; a fresh clone must fetch that reference
as described in README_HEATMAP.md.

Actual result: 19 contract tests in 4.527 seconds, 18 passed and one CUDA RNG test
skipped. This is not a new CUDA validation claim. The same research Python
sources previously passed all19 tests on the allocated host GPUs; each remote
server must perform its own CUDA/actual-batch preflight.

The clean export also passed:

- Python compilation of the15 published Python files.
- `--help`/imports for research runner, report runner and checkpoint diagnosis.
- Shell syntax checks for both heatmap scripts.
- Staged whitespace checks except the intentionally preserved B0 whitespace below.

The CPU raw log remains local at
`outputs/heatmap/place/publication_clean_tree_cpu_20261012.log`; it is not in Git.

## Preserving the active paired pilot

Publishing did not edit research Python files or stop/restart training. At
publication preparation the original server was running H1-Sup after completing
B0 pilot. Later runs may record a new Git revision/dirty-diff hash, but the exact
per-run Python/JSON snapshots are unchanged by these documentation/ignore-rule
changes. Pairing must compare those saved snapshots, not infer code identity
only from an enclosing Git revision.

`git diff --cached --check` reports two inherited whitespace-only blank lines in
`models/b0_repro.py` (lines52 and55). They were confirmed in the frozen reference
and deliberately not reformatted during the pilot: even a formatting edit would
change its recorded source checksum. All other staged files pass that check.

The original run source manifests enumerate unrelated local Python files too.
A research-only fresh clone therefore need not have an identical whole-tree
manifest hash to the original dirty workspace. Formal configurations on the new
server must all use the same published commit, source set and environment;
do not silently mix or substitute different source trees across a comparison.

## Assets and authorization still required remotely

Dataset and official `evaluation/gt_tracks.txt` must be provided separately.
The original absolute symlinks and Python environment were not packaged.
The recovery workflow script contains original-server GPU UUIDs and must not be
executed unchanged elsewhere. Use locally verified device selection and the
canonical Python runner instead.

Pushing source is not approval for full training. Remote Codex must first report
preflight, measured throughput/memory and a full-run time estimate, then obtain
the user's GPU-hours budget before starting B0 seed1 or any larger matrix.
