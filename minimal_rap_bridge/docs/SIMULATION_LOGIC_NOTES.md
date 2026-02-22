# PufferDrive Simulation Logic Notes (Condensed)

Date: 2026-02-19  
Audience: Bridge users who want practical behavior rules quickly.

## 1) Quick Mental Model

Each step does three things: move agents, evaluate events (collision/offroad/goal), then apply configured responses.  
Responses are mostly per-agent; full scene reset is a separate env-level event.

## 2) Collision / Offroad / Goal: What Actually Happens

Use this intuition:

- `collision_behavior`
  - `0` ignore: collision is logged/penalized, agents keep moving
  - `1` stop: colliding agent(s) stop in place
  - `2` remove: colliding agent(s) are removed from active motion
- `offroad_behavior`
  - `0` ignore: offroad is logged/penalized, agent keeps moving
  - `1` stop: offroad agent stops
  - `2` remove: offroad agent is removed
- `goal_behavior`
  - `0` respawn: agent respawns after goal
  - `2` stop: agent stops at goal

Important:

- Ego is not special to physics. Ego is just whichever agent you are viewing.
- Respawn is goal-driven, not collision-driven.

## 3) Reset / Termination vs Per-Agent Outcomes

- `stop`, `remove`, `respawn` are per-agent outcomes.
- Env reset/termination restarts the whole scene.
- Collision alone does not automatically end the episode.
- Reset happens mainly at episode horizon (or earlier with specific termination mode settings).

## 4) Scene Length

- Approximate duration is `episode_length * dt`.
- With default `dt=0.1`, set `episode_length=90` for about 9 seconds.

## 5) Multi-Env: Why It Happens

Key idea:

- In Python `Drive`, `num_agents` means total controlled batch slots, not one-scene agent count.
- Backend may combine multiple scene slices to reach that target.

Why this exists:

- It gives stable batch sizing for RL training throughput.

Why it causes bridge confusion:

- If concatenated multi-env state is treated like one scene, you can get merged/ghost behavior.
- For single-scene parity workflows, force `num_envs=1` or isolate one env slice.

Random note:

- Scene sampling is random with replacement, so the same map can be picked more than once.
- With multiple maps, `num_envs` can vary run-to-run.


## 6) Why Native `./visualize` Usually Looks Cleaner

- Native `./visualize` is usually single-map, single-env.
- It does not use the same multi-slice vectorized batching path.
- So it does not accidentally merge multiple scenes in one render.

## 7) Trajectory Following vs Policy Control

- Controlled agents are action/policy driven.
- Expert/static actors are trajectory-replay driven.
- So default behavior is mixed, not full-scene pure log replay.

## 8) Practical Defaults for Bridge Work

- Keep `num_envs=1` for parity checks.
- Use `episode_length=90` for ~9s horizon at `dt=0.1`.
- Keep `resample_frequency=0` when you want one stable scene.
- Set collision/offroad behavior intentionally in config.
