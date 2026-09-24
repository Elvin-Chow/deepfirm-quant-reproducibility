# Mathematics financial-semantic audit exercise

This directory contains the executable five fixed mutation cases and five matched normal controls reported in the Mathematics manuscript. This audit addition is separate from the `v1.0.0` release. The companion submission archive `Audit_Validation_Artifacts.zip` includes this source directory together with the original and migrated-run evidence records.

## Run

Use a Python environment with the dependencies in `requirements-lock.txt`. With those dependencies already installed, no network access, market data, model weights, or H5 final-test artifact is required:

    cd audits/mathematics_2026_09_23
    python run_injections.py

The runner checks SHA-256 identities of all four fixed inputs before testing. It imports the original research implementation from `snapshot/` and writes only `replay/RESULTS.json` and `replay/RESULTS.csv`. Generated replay files are excluded from version control. The companion submission archive preserves the original runner and results separately; the adapted runner changes its root to this directory's isolated snapshot and redirects output to `replay/`.

`SOURCE_MANIFEST.json` lists the source snapshot, fixed inputs, protocol, environment lock, and adapted runner with their hashes. The snapshot contains the 26 Python files required by the local import chain for the original guards and cost simulator, including package initializers and their transitive imports. Snapshot source and fixed input bytes are preserved from the tested research implementation. No raw market dataset or model weights are included.

## Results and limits

The original author-executed run identified all five injected faults and accepted all five normal controls in 5.827 ms of case-function time. The migrated local replay also passed 5/5 faults and 5/5 normal controls in 8.133 ms. The companion submission archive retains both records, the original runner, and failed invocation notes. Timings exclude Python import and process startup. The three production guards check security overlap, future signal dates, and ordered feature schema. The split-crossing and cost-ledger checks were added to this audit harness. The ten designed inputs do not estimate field false-alarm or miss rates, independent audit effectiveness, predictive quality, or economic value.
