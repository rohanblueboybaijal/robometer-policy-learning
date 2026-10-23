# π₀ warm start on LIBERO-Goal stove

## Unified LIBERO-90 dashboard

Run one local dashboard for the task-38 and task-55 experiments and future
LIBERO-90 fine-tuning tasks:

```bash
.venv-policy/bin/python -m experiments.pi0_stove.dashboard \
  --data-root /data/rohan/Reward-Modeling --port 8768
```

Open `http://<host>:8768/`. The page refreshes every 60 seconds and reads each
task's `evaluation.csv` files on every request. New task directories named
`pi0_libero90_task<id>_*` with a `manifest.json` containing `task` and `seeds`
appear automatically; runs use `seed<seed>/<sparse|robometer>_seed<seed>/evaluation.csv`.
The manifest may set `seed100_run` when seed 100 lives in another directory.
No plotting step or server restart is needed as runs advance.

## Task-55 direct DSRL smoke test

`dsrl_smoke.py` runs only 700 physical transitions per reward arm and two evaluation
episodes. It uses the original LIBERO DSRL SAC cadence: 500 physical-transition
learning start, one actor update per 20-action decision, and 25 critic updates
per actor update. The actor's 32-dimensional output is passed directly as pi0
noise, as in the original DSRL rollout worker. The frozen pi0 and Robometer
models remain in separate Python processes so they fit on the available GPUs.
Checkpoints overwrite `latest.pt` during training and add `final.pt` at the end;
the replay buffer and model weights are not duplicated into checkpoint folders.
This smoke test does not launch the full fine-tuning runs.

The full direct-DSRL comparison uses three matched seeds (100–102) for both
rewards on LIBERO-90 task 38, then the same six runs on task 55. It runs with
`dsrl_multitask.py` and writes each task to
`/data/rohan/Reward-Modeling/pi0_libero90_task<id>_dsrl_3seeds_20261012`.
Only an idle GPU is used. Within each arm there is one sequential LIBERO
environment; the two GPUs may run different arms concurrently. Each run uses
100,000 physical transitions, 25 held-out evaluation episodes every 5,000
transitions, and a rolling `latest.pt` every 10,000 transitions plus one
`final.pt`. The task-55 stage starts only after all six task-38 runs complete.
The one-page dashboard discovers these task folders and labels them as direct
DSRL, separately from the earlier residual-SAC curves.

This experiment starts from the released `pi0_libero` checkpoint. π₀ weights remain frozen; a SAC actor learns a 32-dimensional offset to the Gaussian noise used for π₀ action sampling. This preserves the pretrained action prior while the actor learns from online rewards. Both reward arms use the same checkpoint, reset split, physical-step budget, controller architecture, and evaluation states.

This is a DSRL-style SAC steering experiment, not an exact invocation of this repository's `scripts/train_dsrl.py`. That script broadcasts the actor's 32-dimensional output as the full π₀ noise tensor. Here the output is added to fresh Gaussian noise, so zero steering reproduces the pretrained π₀ sampling distribution. The residual design is intentional for measuring whether either reward signal erodes an already successful policy; comparisons should use the paired arms below rather than published DSRL numbers.

The independent baseline runs 25 held-out episodes on LIBERO-Goal task 7, `turn_on_the_stove`, using initial states 25–49. It executes 20 actions from each π₀ action chunk and stops on simulator success or after 400 physical steps. The simulator's native reward is 0/1, with one success reward on the terminal step. Ten zero-action physics steps settle each reset before evaluation and do not count as training steps.

## Environment and checkpoint

The root filesystem has little free space. Install Python, the `uv` virtual environment, caches, temporary files, and the 11.19 GiB checkpoint on `/data`:

```bash
export UV_CACHE_DIR=/data/rohan/Reward-Modeling/uv_cache
export TMPDIR=/data/rohan/rbm-tmp
export XDG_CACHE_HOME=/data/rohan/Reward-Modeling/pi0_cache
uv python install 3.11 --install-dir /data/rohan/Reward-Modeling/uv-python
uv venv --python /data/rohan/Reward-Modeling/uv-python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11 /data/rohan/Reward-Modeling/.venv-pi0
uv pip install --python /data/rohan/Reward-Modeling/.venv-pi0/bin/python -e ./dsrl_openpi
python -m experiments.pi0_stove.download_checkpoint --out /data/rohan/Reward-Modeling/pi0_checkpoints/pi0_libero
```

The downloader checks the public `openpi-assets` Google Cloud Storage bucket and verifies file sizes and supplied MD5 hashes. The installed OpenPI/JAX worker runs in Python 3.11; the LIBERO simulator, DINO encoder, and SAC controller use `.venv-policy` with Python 3.10. Robometer inference remains in `.venv-reward` on a separate GPU. Do not resolve the `bin/python` symlink from the π₀ venv when launching it: the resolved base interpreter lacks the installed packages.

## Baseline and DSRL runs

```bash
ROOT=/data/rohan/Reward-Modeling/pi0_stove_20261010
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DISABLED=true \
  .venv-policy/bin/python -m experiments.pi0_stove.baseline \
  --out "$ROOT/baseline_seed100" \
  --checkpoint /data/rohan/Reward-Modeling/pi0_checkpoints/pi0_libero \
  --pi0-python /data/rohan/Reward-Modeling/.venv-pi0/bin/python \
  --gpu 0 --action-exec-len 20

.venv-policy/bin/python -m experiments.pi0_stove.launch \
  --root "$ROOT" \
  --checkpoint /data/rohan/Reward-Modeling/pi0_checkpoints/pi0_libero \
  --pi0-python /data/rohan/Reward-Modeling/.venv-pi0/bin/python
```

The default coordinator waits for all 25 baseline evaluations and runs the reward arms in sequence. On this machine, a load-and-inference preflight showed that π₀, Robometer, and DINO fit together on GPU 1 at about 18.5 GiB. The live experiment therefore used `parallel_handoff.py` to adopt the sparse trainer on GPU 0 and launch the Robometer trainer on GPU 1, with both π₀ and Robometer on that second card. The handoff checks the live sparse process and idle second GPU before launching; it stops only the old coordinator, leaving the sparse trainer running. It writes local CSV, JSONL, checkpoints, PNG/PDF curves, and an offline HTML page. W&B is disabled.

To repeat the parallel handoff while the default coordinator is running, pass the live sparse and coordinator PIDs shown in `dsrl_stove_seed100/execution.json`:

```bash
.venv-policy/bin/python -m experiments.pi0_stove.parallel_handoff \
  --root "$ROOT" \
  --checkpoint /data/rohan/Reward-Modeling/pi0_checkpoints/pi0_libero \
  --pi0-python /data/rohan/Reward-Modeling/.venv-pi0/bin/python \
  --sparse-pid "$SPARSE_PID" --old-coordinator-pid "$COORDINATOR_PID"
```

Each SAC decision steers a 20-step π₀ chunk. The sparse arm receives the simulator's 0/1 terminal reward for that chunk. The Robometer arm receives the mean of corrected progress predictions across the chunk, using only observed prefix frames and no simulator-success bonus. Both arms terminate episodes on simulator success. SAC starts updating after 250 collected chunks, corresponding to roughly 5,000 physical transitions. The budget is 100,000 physical transitions per arm; evaluation uses the same 25 held-out states at step zero and about every 5,000 transitions. All checkpoints report simulator success, independently of the training reward.

The baseline is intentionally measured first. If it starts near 100%, subsequent success curves can reveal loss of skill but have little room to show improvement. The three-seed scratch SAC study is a separate comparison; this π₀ experiment is one paired seed.

## Five-seed LIBERO-90 task 55 comparison

The task-55 study uses `LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray`. Seed 100 has completed both reward arms. `multiseed.py` adds seeds 101–104 using copies of the seed-100 configurations with only the seed and physical GPU assignment changed. Each arm trains for 100,000 physical transitions, evaluates on the same 25 held-out initial states every ~5,000 transitions, and saves locally. The two GPU lanes each run two sparse-reward seeds followed by two Robometer-reward seeds. A Robometer arm uses one entire 24 GiB GPU for π₀, SAC, DINO, rendering, and reward inference.

The five-seed output is `/data/rohan/Reward-Modeling/pi0_libero90_task55_five_seeds_20261011/`. Its `index.html`, `success_rate_five_seeds.png`, PDF, and `curves.json` are updated by the manager while it runs. The bold curves align checkpoints by evaluation number because physical step counts differ slightly; translucent lines show individual seeds. Shaded 95% t intervals and mean values use the available seeds at each checkpoint, so their sample count changes until all jobs finish. The manager and lane status files record process IDs and job state. If the manager stops unexpectedly, GPU lanes run in separate sessions and can be inspected independently.

```bash
.venv-policy/bin/python -m experiments.pi0_stove.multiseed launch \
  --root /data/rohan/Reward-Modeling/pi0_libero90_task55_five_seeds_20261011 \
  --seed100-run /data/rohan/Reward-Modeling/pi0_libero90_task55_20261010/dsrl_LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_seed100
```
