# Motion Challenge · grit

[Author project](https://github.com/mrzuang/GRIT_teleop_deploy) · [Checkpoint](https://github.com/mrzuang/GRIT_teleop_deploy/blob/5660177cf63f72803367d1e24eb7385a8cdc877c/sim2real/checkpoints/policy.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/motion-challenge-grit.git
cd motion-challenge-grit
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `motion-challenge`
submodule owns datasets, physics, metrics and verification. See `--help` for options.
