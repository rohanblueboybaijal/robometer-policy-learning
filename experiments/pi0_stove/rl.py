"""SAC steering of frozen pi0: paired sparse/Robometer LIBERO stove runs."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import traceback
from pathlib import Path

import numpy as np

from experiments.pi0_stove.baseline import Pi0Client
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.local_log import LocalLog
from experiments.reward_audit.metrics import prefix_frames, wilson
from experiments.reward_audit.rpc import RewardClient
from experiments.reward_audit.sac import Dino, Replay, actor_actions, make_sac, save_checkpoint


def select_reward(kind, sparse_steps, progress_steps):
    """One scalar per pi0 action chunk; both reward arms stay in [0, 1]."""
    if kind == "sparse":
        return float(max(sparse_steps))
    if kind == "robometer":
        if progress_steps is None or not len(progress_steps):
            raise ValueError("Missing Robometer scores for action chunk")
        values = np.asarray(progress_steps, dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid Robometer progress")
        return float(values.mean())
    raise ValueError(kind)


def execute_chunk(policy, env, observation, prompt, steering, length, history, reward_camera, max_frames, need_model,
                  noise_mode="additive"):
    actions = policy.infer(observation, prompt, noise=steering, noise_mode=noise_mode)
    prefixes, sparse_steps = [], []
    for action in actions[:length]:
        result = env.step([action])[0]
        observation = result["obs"]
        sparse_steps.append(result["sparse"])
        history.append(observation[reward_camera])
        if need_model:
            prefixes.append(prefix_frames(history, max_frames))
        if result["terminated"] or result["truncated"]:
            break
    return observation, result, prefixes, sparse_steps


def evaluate(cfg, policy, actor, encoder, env, device, log, step):
    """Identical reset states and pi0 sampling seeds in both reward arms."""
    import torch

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    records = []
    try:
        prompt = env.metadata[0]["language"]
        for episode in range(cfg["eval_episodes"]):
            state_id = cfg["train_init_states"] + episode % cfg.get("eval_unique_states", cfg["eval_episodes"])
            policy.reset_seed(cfg["seed"] * 100000 + cfg["train_init_states"] + episode)
            observation = env.reset({0: state_id})[0]["obs"]
            history = [observation[cfg["reward_camera"]]]
            while True:
                noise_mode = cfg["dsrl"].get("noise_mode", "additive")
                if step == 0:
                    steering = None if noise_mode == "direct" else np.zeros(cfg["dsrl"]["noise_dim"], dtype=np.float32)
                else:
                    features = encoder.encode([observation])
                    steering = actor_actions(actor, features, device, deterministic=True)[0]
                observation, result, _, _ = execute_chunk(
                    policy, env, observation, prompt, steering, cfg["dsrl"]["action_exec_len"], history,
                    cfg["reward_camera"], cfg["max_frames"], False, noise_mode,
                )
                if result["terminated"] or result["truncated"]:
                    break
            record = {"phase": "eval", "env_steps": step, "episode": episode, "init_state_id": state_id,
                      "success": bool(result["terminated"]), "length": result["elapsed"]}
            log.episode(record)
            records.append(record)
        successes = sum(e["success"] for e in records)
        low, high = wilson(successes, len(records))
        log.evaluation(step, successes, len(records), (low, high))
        metrics = {"successes": successes, "episodes": len(records), "success_rate": successes / len(records),
                   "ci_low": low, "ci_high": high}
        log.log(metrics, step, "eval")
        print(f"Evaluation at {step} transitions: {successes}/{len(records)} simulator successes", flush=True)
        return metrics
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def run(cfg, out):
    import torch

    if os.environ.get("AUDIT_LAUNCHER_GUARDED") != "1":
        raise RuntimeError("Launch through the guarded pi0 DSRL coordinator")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    prepare_libero_paths(out / "libero_paths", REPO)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(cfg["torch_threads"])
    device = torch.device("cuda:0")
    log = LocalLog(out)
    policy = train_env = eval_env = reward = None
    write_json(out / "status.json", {"status": "running"})
    try:
        policy = Pi0Client(Path(cfg["pi0_python"]), Path(cfg["pi0_checkpoint"]), cfg["seed"],
                           cfg["policy_gpu"], out, config_name=cfg.get("policy_config_name", "pi0_libero"))
        if cfg["reward"] == "robometer":
            reward = RewardClient(cfg, out, cfg["reward_gpu"])
        encoder = Dino(cfg["dino_checkpoint"], device)
        train_env = EnvPool(cfg, 1)
        eval_env = EnvPool(cfg, 1, rank_offset=100000)
        prompt = train_env.metadata[0]["language"]
        replay = Replay(cfg["buffer_capacity"],
                        {"dino_embedding": encoder.dimension, "observation/state": 8, "language": 384},
                        cfg["seed"], action_dim=cfg["dsrl"]["noise_dim"])
        algorithm = make_sac(cfg, replay, device, action_dim=cfg["dsrl"]["noise_dim"])
        observation = train_env.reset()[0]["obs"]
        history = [observation[cfg["reward_camera"]]]
        episode_return = 0.0
        episode_id = 0
        step, decision, last_eval, last_save, last_log = 0, 0, 0, 0, 0
        noise_mode = cfg["dsrl"].get("noise_mode", "additive")
        if noise_mode not in ("additive", "direct"):
            raise ValueError(f"Unknown noise mode: {noise_mode}")
        final_metrics = evaluate(cfg, policy, algorithm.actor, encoder, eval_env, device, log, 0)
        while step < cfg["steps"]:
            features = encoder.encode([observation])
            can_train = ((step >= cfg["sac"]["learning_starts"]) if noise_mode == "direct"
                         else (decision >= cfg["sac"]["learning_starts"]))
            if can_train:
                steering = actor_actions(algorithm.actor, features, device)[0]
            elif noise_mode == "direct":
                steering = np.random.uniform(-1, 1, cfg["dsrl"]["noise_dim"]).astype(np.float32)
            else:
                steering = np.zeros(cfg["dsrl"]["noise_dim"], dtype=np.float32)
            policy.reset_seed(10_000_000 + cfg["seed"] * 1_000_000 + decision)
            next_obs, result, prefixes, sparse_steps = execute_chunk(
                policy, train_env, observation, prompt, steering,
                min(cfg["dsrl"]["action_exec_len"], cfg["steps"] - step), history,
                cfg["reward_camera"], cfg["max_frames"], reward is not None, noise_mode,
            )
            progress = reward.predict(prefixes, prompt)[:, 0] if reward is not None else None
            scalar_reward = select_reward(cfg["reward"], sparse_steps, progress)
            next_features = encoder.encode([next_obs])
            terminated, truncated = bool(result["terminated"]), bool(result["truncated"])
            replay.add(features, steering[None, :], np.array([scalar_reward], dtype=np.float32),
                       next_features, np.array([terminated]), np.array([truncated]))
            step += len(sparse_steps)
            decision += 1
            episode_return += scalar_reward
            if can_train and (noise_mode == "direct" or len(replay) >= cfg["sac"]["batch_size"]):
                metrics = algorithm.train_step()
                if step - last_log >= cfg["log_every"]:
                    log.log({key: float(value) for key, value in metrics.items()}, step, "sac")
                    last_log = step
            if terminated or truncated:
                record = {"phase": "train", "env_steps": step, "episode": episode_id,
                          "success": terminated, "length": result["elapsed"],
                          "training_return": episode_return}
                if progress is not None:
                    record["final_progress"] = float(progress[-1])
                log.episode(record)
                episode_id += 1
                episode_return = 0.0
                next_obs = train_env.reset()[0]["obs"]
                history = [next_obs[cfg["reward_camera"]]]
            observation = next_obs
            if step - last_eval >= cfg["eval_every"] or step == cfg["steps"]:
                final_metrics = evaluate(cfg, policy, algorithm.actor, encoder, eval_env, device, log, step)
                last_eval = step
            if step - last_save >= cfg["save_every"]:
                save_checkpoint(algorithm, out / "checkpoints/latest.pt", cfg, step)
                last_save = step
        latest = out / "checkpoints/latest.pt"
        final = out / "checkpoints/final.pt"
        if cfg.get("rolling_checkpoint_only") and latest.exists() and last_save == step:
            latest.replace(final)
        else:
            save_checkpoint(algorithm, final, cfg, step)
            if cfg.get("rolling_checkpoint_only"):
                latest.unlink(missing_ok=True)
        write_json(out / "status.json", {"status": "complete", "env_steps": step,
                                          "decisions": decision, "final_metrics": final_metrics})
    except BaseException:
        write_json(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        for resource in (train_env, eval_env, reward, policy):
            if resource is not None:
                resource.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("AUDIT_LAUNCHER_GUARDED") != "1":
        parser.error("Run via the guarded pi0 DSRL coordinator")

    def stop(signum, frame):
        raise KeyboardInterrupt("Experiment interrupted")

    signal.signal(signal.SIGTERM, stop)
    run(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
