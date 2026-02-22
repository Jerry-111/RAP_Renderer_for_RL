# New-Repo Plan: Torch-Policy RAP Bridge

## Objective
Reimplement the RAP bridge in a new repository with minimal upstream coupling by:

1. running policy inference in Python/Torch from a `.pt` checkpoint
2. stepping simulation through upstream `Drive.step(actions)`
3. converting simulator outputs to RAP scenario input and rendering frames

This avoids carrying a custom native-policy runtime in C for the bridge workflow.

## Reference Pattern from `pufferl`
This design is directly inspired by the existing `pufferl` evaluate loop pattern:

1. `policy.forward_eval(observations, state)`
2. sample action from logits
3. `env.step(action)`

I will keep that same control structure, but attach RAP conversion/rendering in each frame iteration.

## What I Will Change (New Repo Only)
- Add a Torch-policy bridge runner that:
  - loads policy checkpoint (`.pt`)
  - infers actions each frame in Python
  - steps PufferDrive via `Drive.step`
  - reads global state/map and builds RAP scenario
  - renders/saves frame outputs
- Keep bridge logic in Python only.

## What I Will Not Change (Upstream Core)
- No modifications to upstream C simulator/binding internals:
  - `pufferlib/ocean/drive/drive.h`
  - `pufferlib/ocean/env_binding.h`
  - `pufferlib/ocean/drive/binding.c`
  - `pufferlib/ocean/drive/drive.c`
  - `pufferlib/ocean/drive/visualize.c`

## Runtime Flow in the New Bridge
Per-frame flow:

`Torch policy inference` -> `sample action` -> `Drive.step(action)` -> `get global state/map` -> `RAP scenario conversion` -> `render/save`

Important clarification:
- only the policy inference location changes (native C policy path to Python/Torch)
- simulator transition logic is still upstream C through `Drive.step` (same stepping semantics)

## Why This Is Minimal
- zero new C API surface
- maximum reuse of upstream stable env interface
- clearer maintenance boundary (bridge orchestration in Python, simulator in upstream C)

## Why Use `Drive` Directly (Not `pufferlib.vector`) for the Bridge
- RAP rendering needs: one controllable timeline, stable ego selection, simple action/state shapes, minimal moving parts.
- This mirrors the native renderer (`visualize.c`) control-flow model: single env in a single loop, with an ego chosen for view (though native renderer does it in C)
- Using `Drive` directly gives a straightforward loop: `obs -> forward_eval -> action -> Drive.step(action)`.
- It avoids extra wrapper complexity: no `send/recv` protocol, no worker processes, no env-id slicing/offset logic, fewer action shape transformations.
- Vectorization is great for training throughput, but for a rendering-heavy bridge it is usually not worth the added complexity until correctness is proven.

## Implementation Decisions
These are the decisions we are implementing right now, including risk/reward and the main options not chosen.

- Dynamics/action space: `classic` discrete joint action space with `7*13=91` actions (action id in `[0, 90]`).
- Policy memory: non-recurrent only (no hidden-state carry).
  - Reward: simpler + fewer silent failure modes.
  - Risk: cannot run recurrent checkpoints; fail fast if recurrence is detected.
  - Not chosen: recurrent support (requires hidden-state + done/mask handling).
- Action selection: stochastic sampling from logits (PPO-style), following `pufferl` evaluate pattern.
  - Reward: closer to training-time behavior than greedy.
  - Risk: sampling + GPU math can reduce strict reproducibility; tiny logit changes can flip sampled actions.
  - Not chosen: greedy argmax (more reproducible, but behavior may differ).
- Inference device: GPU inference.
  - Reward: throughput.
  - Risk: harder to guarantee bit-for-bit reproducibility across hardware/drivers/torch versions.
  - Not chosen: CPU inference (more reproducible, slower).
- Reproducibility approach: fixed seeds + documented nondeterminism caveats.
  - Reward: keeps pipeline fast and simple.
  - Risk: may still diverge across runs/platforms; log seeds/config + action/state summaries.
  - Not chosen: strict deterministic GPU mode by default (can reduce speed and/or restrict ops).
- Environment mode: force a single environment (`num_envs == 1`).
  - Reward: simplest mapping; avoids multi-env slicing/ego tracking complexity.
  - Not chosen: `num_envs > 1` (adds slicing/offset logic).
- Ego selection: fixed ego index.
  - Reward: removes ego-selection randomness; consistent renders.
  - Not chosen: seeded-random ego selection (reproducible, but adds moving part).
- Action validation: strict action contract checks before `Drive.step`.
  - Reward: catch policy/env mismatch early (dtype/shape/range).
  - Not chosen: best-effort casting (harder to debug when wrong).

## Fast-Fail Guardrails (Required)
- Refuse recurrent checkpoints/config (non-recurrent only).
- Assert action dtype is integer and action range is valid for `classic` (`0..90`).
- Assert action shape matches what `Drive.step` expects (use `env.actions.shape` as the reference).
- Log enough metadata to diagnose stochastic divergence (seed, device, checkpoint id/path, dynamics, ego index).


## Goals
This plan is complete when:

1. Torch-policy bridge runs end-to-end for target frame count
2. simulator stepping goes through upstream `Drive.step` only
3. no upstream C files need to be modified
4. RAP outputs are produced from simulator state in the new repo
