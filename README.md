# Motion Challenge · grit

[grit](https://github.com/mrzuang/GRIT_teleop_deploy) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-grit.git
cd motion-challenge-grit
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

**Development adapter — author-inference review pending.**
[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.

Run all frames after tracking failure: `uv run evaluate.py --termination none`.
This records full-motion errors under a separate protocol; success still requires no failure.

## Checkpoint source

[mrzuang/GRIT_teleop_deploy](https://github.com/mrzuang/GRIT_teleop_deploy/blob/5660177cf63f72803367d1e24eb7385a8cdc877c/sim2real/checkpoints/README.md) · [sim2real/checkpoints/policy.onnx](https://github.com/mrzuang/GRIT_teleop_deploy/blob/5660177cf63f72803367d1e24eb7385a8cdc877c/sim2real/checkpoints/policy.onnx). Author-released checkpoint.

Downloaded directly from the pinned author repository. Its SHA256 exactly matches both the author checksum and the former Drive mirror; the evaluated bytes are unchanged.
