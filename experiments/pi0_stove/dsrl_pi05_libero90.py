"""Paired direct-DSRL runs from pi0.5 on LIBERO-90 tasks 02 and 28."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.pi0_stove.dsrl_libero10_task06 import KINDS, STEPS, config as base_config
from experiments.pi0_stove.dsrl_smoke import CACHE
from experiments.pi0_stove.dsrl_multitask import now
from experiments.reward_audit.config import REPO, task_map, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath

DATA_ROOT = Path('/data/rohan/Reward-Modeling')
TASK_IDS = (2, 28)
SEEDS = (100, 101, 102)
CHECKPOINT = DATA_ROOT / 'pi0_checkpoints/pi05_libero'


def root(task_id: int) -> Path:
    return DATA_ROOT / f'pi05_libero90_task{task_id:02d}_dsrl_5seeds_20261020'


def run_dir(task_id: int, seed: int, kind: str) -> Path:
    return root(task_id) / f'seed{seed}' / f'{kind}_seed{seed}'


def config(task_id: int, seed: int, kind: str, gpu: int) -> dict:
    cfg = base_config(seed, kind, gpu, task_id=6)
    cfg.update(suite='libero_90', task_id=task_id,
               task=task_map()['libero_90'][task_id],
               name=f'pi05_libero90_task{task_id:02d}_dsrl_{kind}',
               comparison=f'pi05_libero90_task{task_id:02d}_direct_dsrl_3seeds',
               ood_claim='held_out_task',
               pi0_checkpoint=str(CHECKPOINT),
               pi0_python=str(DATA_ROOT / '.venv-pi0/bin/python'),
               policy_config_name='pi05_libero',
               horizon=400, steps=STEPS, eval_every=5000,
               eval_episodes=50 if seed in (100, 101) else 25,
               eval_unique_states=25,
               train_init_states=25, save_every=10000,
               rolling_checkpoint_only=True)
    cfg['dsrl']['action_exec_len'] = 10
    assert cfg['dsrl']['noise_mode'] == 'direct'
    assert cfg['sac']['learning_starts'] == 500
    assert cfg['sac']['num_critic_updates_per_actor_update'] == 25
    return cfg


def initialize() -> None:
    if not (CHECKPOINT / 'params').is_dir() or not (CHECKPOINT / 'assets').is_dir():
        raise RuntimeError(f'Missing pi0.5 checkpoint: {CHECKPOINT}')
    for task_id in TASK_IDS:
        folder = root(task_id)
        folder.mkdir(parents=True, exist_ok=True)
        cfg = config(task_id, SEEDS[0], 'robometer', 0)
        expected = {'suite': 'libero_90', 'task_id': task_id,
                    'task': cfg['task'], 'seeds': list(SEEDS),
                    'rewards': list(KINDS), 'steps_per_run': STEPS,
                    'algorithm': 'direct DSRL', 'pi0_checkpoint': str(CHECKPOINT),
                    'policy_config_name': 'pi05_libero',
                    'reward_checkpoint': cfg['reward_checkpoint'],
                    'eval_episodes_by_seed': {'100': 50, '101': 50, '102': 25},
                    'eval_unique_states': 25,
                    'eval_state_ids_by_seed': {'100': list(range(25, 50)) * 2,
                                               '101': list(range(25, 50)) * 2,
                                               '102': list(range(25, 50))},
                    'eval_every': 5000, 'horizon': 400,
                    'baseline_source': 'seed100/sparse_seed100/evaluation.csv at 0 transitions'}
        manifest = folder / 'manifest.json'
        if manifest.exists():
            current = json.loads(manifest.read_text())
            if (current.get('seeds') == [100, 101, 102, 103, 104] and
                    current.get('eval_episodes') == 50 and
                    current.get('task_id') == task_id):
                archive = folder / 'manifest.initial_5seed_plan.json'
                if archive.exists():
                    raise RuntimeError(f'Manifest archive already exists: {archive}')
                write_json(archive, current)
                write_json(manifest, {**expected, 'created': current.get('created'),
                                      'plan_revised': now(),
                                      'original_manifest': archive.name})
                current = json.loads(manifest.read_text())
            if any(current.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f'Unexpected manifest: {manifest}')
        else:
            write_json(manifest, {**expected, 'created': now()})


def complete(task_id: int, seed: int, kind: str) -> bool:
    folder = run_dir(task_id, seed, kind)
    try:
        status = json.loads((folder / 'status.json').read_text())
        return (status.get('status') == 'complete' and status.get('env_steps') == STEPS
                and (folder / 'checkpoints/final.pt').is_file())
    except (OSError, ValueError):
        return False


def launch(task_id: int, seed: int, kind: str, gpu: int):
    folder = run_dir(task_id, seed, kind)
    folder.mkdir(parents=True, exist_ok=False)
    cfg = config(task_id, seed, kind, gpu)
    write_json(folder / 'config.json', cfg)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
           'MUJOCO_GL': 'egl', 'MUJOCO_EGL_DEVICE_ID': str(gpu),
           'EGL_DEVICE_ID': str(gpu), 'AUDIT_LAUNCHER_GUARDED': '1',
           'PYTHONPATH': source_pythonpath(), 'WANDB_MODE': 'disabled',
           'WANDB_DISABLED': 'true', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'XDG_CACHE_HOME': str(CACHE), 'HF_HOME': str(CACHE / 'hf'),
           'TORCHINDUCTOR_CACHE_DIR': str(CACHE / 'torchinductor'),
           'TRITON_CACHE_DIR': str(CACHE / 'triton'),
           'TMPDIR': '/data/rohan/rbm-tmp'}
    log = (folder / 'console.log').open('w')
    try:
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'experiments.pi0_stove.rl',
             '--config', str(folder / 'config.json'), '--out', str(folder)],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return process, log


def update_status(active: dict, pending: list, errors: list[str]) -> None:
    for task_id in TASK_IDS:
        write_json(root(task_id) / 'orchestrator.json', {
            'status': 'failed' if errors else ('running' if active or pending else 'complete'),
            'pid': os.getpid(), 'updated': now(), 'errors': errors,
            'active': {str(gpu): {'task_id': task, 'seed': seed, 'kind': kind,
                                  'pid': process.pid, 'run': str(run_dir(task, seed, kind))}
                       for gpu, (task, seed, kind, process, _, _) in active.items()
                       if task == task_id},
            'pending': [{'seed': seed, 'kind': kind} for task, seed, kind in pending
                        if task == task_id]})


def run() -> None:
    initialize()
    # Each run evaluates its frozen policy before taking a training transition.
    jobs = [(task, seed, kind) for seed in SEEDS for task in TASK_IDS for kind in KINDS]
    pending = [job for job in jobs if not complete(*job)]
    for job in pending:
        if run_dir(*job).exists():
            raise RuntimeError(f'Incomplete run needs inspection: {run_dir(*job)}')
    active: dict = {}
    errors: list[str] = []
    while pending or active:
        for gpu, (task, seed, kind, process, log, reservation) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            reservation.__exit__(None, None, None)
            del active[gpu]
            if code or not complete(task, seed, kind):
                errors.append(f'task {task} {kind} seed {seed} exited {code}: {run_dir(task, seed, kind)}')
            else:
                print(f'Completed task {task} {kind} seed {seed} on GPU {gpu}', flush=True)
        if not errors:
            for gpu in (0, 1):
                if gpu in active or not pending:
                    continue
                if shutil.disk_usage(DATA_ROOT).free < 30 * 1024**3:
                    print('Waiting for 30 GiB free on /data', flush=True)
                    break
                try:
                    reservation = reserve_gpus([gpu])
                    reservation.__enter__()
                except RuntimeError:
                    continue
                task, seed, kind = pending.pop(0)
                try:
                    process, log = launch(task, seed, kind, gpu)
                except BaseException:
                    reservation.__exit__(*sys.exc_info())
                    raise
                active[gpu] = (task, seed, kind, process, log, reservation)
                print(f'Started task {task} {kind} seed {seed} on GPU {gpu}: PID {process.pid}', flush=True)
        update_status(active, pending, errors)
        if errors and not active:
            raise RuntimeError('; '.join(errors))
        time.sleep(30)
    assert all(complete(*job) for job in jobs)


def wait_for_active(pids: list[int]) -> None:
    """Do not disturb runs orphaned when the previous queue coordinator stopped."""
    while True:
        active = []
        for pid in pids:
            try:
                command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ')
            except OSError:
                continue
            if b'experiments.pi0_stove.rl' in command:
                active.append(pid)
        if not active:
            return
        print(f'Waiting for existing training runs {active} to complete', flush=True)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with (DATA_ROOT / 'pi05_libero90_dsrl_20261020.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Coordinator already running') from None
        initialize()
        wait_for_active(args.wait_pid)
        run()


if __name__ == '__main__':
    main()
