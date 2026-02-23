# Minimal RAP Vendor Layout

This repository intentionally vendors only the RAP renderer needed by the
PufferDrive bridge:

- `process_data/helpers/renderer.py`
- `LICENSE`

Source:
- https://github.com/vita-epfl/RAP/blob/main/process_data/helpers/renderer.py

Why minimal:
- The bridge imports only `process_data.helpers.renderer` for
  `ScenarioRenderer` and `camera_params`.
- No other RAP modules are required for this workflow.

If you update `renderer.py`, keep this folder lean and avoid copying the full
RAP repository unless a new dependency is actually introduced.
