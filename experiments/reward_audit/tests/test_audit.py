"""CPU fixtures only. These are not LIBERO performance measurements."""

import copy
import json
import sys

import numpy as np
import pytest

from experiments.reward_audit.config import HERE, load_suite, materialize, resolve_task, write_json
from experiments.reward_audit.envs import observation
from experiments.reward_audit.gpu_guard import assert_idle
from experiments.reward_audit.local_log import LocalLog
from experiments.reward_audit.metrics import binary_auc, prefix_frames, training_reward, wilson
from experiments.reward_audit.queue import assert_unreserved, prerequisites, required_gpus
from experiments.reward_audit.report import generate_report, paired_difference
from experiments.reward_audit.reward_worker import (
    endpoints,
    frame_batches,
    preserve_final_hidden_state,
    verify_training_split,
)
from experiments.reward_audit.sac import Replay, load_actor, make_sac, save_checkpoint


def test_queue_only_requires_gpus_used_by_selected_runs():
    suite = HERE / "configs/libero.json"
    assert required_gpus(suite, ["sanity_sparse"], 1, 0) == [1]
    assert required_gpus(suite, ["sanity_robometer"], 1, 0) == [1, 0]
    with pytest.raises(ValueError, match="Unknown"):
        required_gpus(suite, ["missing"], 1, 0)


def test_queue_waits_for_launcher_reservation_even_without_gpu_allocation(tmp_path, monkeypatch):
    import fcntl

    from experiments.reward_audit import queue

    monkeypatch.setattr(queue.tempfile, "gettempdir", lambda: str(tmp_path))
    with (tmp_path / "reward-audit-gpu-1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="reserved GPU 1"):
            assert_unreserved([1])
    assert_unreserved([1])


def test_qwen3_hidden_state_adapter_preserves_actual_final_layer():
    from types import SimpleNamespace

    final = np.arange(12).reshape(1, 3, 4)
    output = SimpleNamespace(hidden_states=None, last_hidden_state=final)
    assert preserve_final_hidden_state(None, (), output) is output
    assert output.hidden_states[-1] is final
    existing = (final.copy(), final.copy())
    output.hidden_states = existing
    preserve_final_hidden_state(None, (), output)
    assert output.hidden_states is existing
    with pytest.raises(ValueError, match="neither"):
        preserve_final_hidden_state(None, (), SimpleNamespace(hidden_states=None))


def test_queue_requires_completed_prerequisites_and_aborts_on_failure(tmp_path):
    assert prerequisites(tmp_path, ["sparse"]) == ["sparse"]
    write_json(tmp_path / "sparse/status.json", {"status": "running"})
    assert prerequisites(tmp_path, ["sparse"]) == ["sparse"]
    write_json(tmp_path / "sparse/status.json", {"status": "complete"})
    assert prerequisites(tmp_path, ["sparse"]) == []
    write_json(tmp_path / "sparse/status.json", {"status": "failed"})
    with pytest.raises(RuntimeError, match="failed"):
        prerequisites(tmp_path, ["sparse"])


def test_queue_does_not_launch_while_gpu_is_busy(tmp_path, monkeypatch):
    from experiments.reward_audit import queue

    write_json(tmp_path / "sparse/status.json", {"status": "complete"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "--out",
            str(tmp_path),
            "--after",
            "sparse",
            "--runs",
            "sanity_robometer",
            "--policy-gpu",
            "1",
            "--reward-gpu",
            "0",
        ],
    )

    def busy(indices):
        assert indices == [1, 0]
        raise RuntimeError("GPU 0 is occupied")

    def cancel(seconds):
        raise KeyboardInterrupt

    def must_not_launch(*args, **kwargs):
        pytest.fail("The queue tried to start a process on an occupied GPU")

    monkeypatch.setattr(queue, "assert_idle", busy)
    monkeypatch.setattr(queue, "refresh_curves", lambda out: None)
    monkeypatch.setattr(queue.time, "sleep", cancel)
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(queue.subprocess, "Popen", must_not_launch)
    with pytest.raises(KeyboardInterrupt):
        queue.main()
    assert json.loads((tmp_path / "reward_queue.json").read_text())["status"] == "cancelled"


@pytest.fixture
def cfg():
    suite = load_suite(HERE / "configs/libero.json")
    cfg = materialize(suite, suite["runs"][0])
    cfg["actor_hidden"] = [16, 16]
    cfg["critic_hidden"] = [16, 16]
    cfg["sac"]["batch_size"] = 4
    return cfg


def test_paper_task_and_ood_name_resolve():
    assert resolve_task("libero_90", "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet") == 28
    assert resolve_task("libero_90", "KITCHEN_SCENE6_close_the_microwave") == 33
    suite = load_suite(HERE / "configs/libero.json")
    runs = [materialize(suite, r) for r in suite["runs"]]
    assert next(r for r in runs if r["name"] == "ood_robometer")["task_id"] == 4
    for comparison in ("sanity", "ood"):
        selected = [r for r in runs if r["comparison"] == comparison]
        gt = next(r for r in selected if r["reward"] == "sparse")
        for r in selected:
            for key in ("seed", "steps", "num_envs", "sac", "initial_actor", "task", "dino_checkpoint"):
                assert r[key] == gt[key]


def test_prevent_training_split_leakage(cfg):
    cfg.update(ood_claim="held_out_task", reward_checkpoint="jesbu1/robometer-4b-fft-libero")
    with pytest.raises(ValueError, match="includes LIBERO-90"):
        materialize({"defaults": {}}, cfg)


@pytest.mark.parametrize("formula", ["progress", "progress_minus_one", "delta"])
def test_model_only_reward_cannot_use_simulator_labels(formula):
    args = {"kind": "robometer", "formula": formula, "progress": [0.2, 0.8], "previous": [0.1, 0.4]}
    np.testing.assert_equal(training_reward(sparse=[0, 0], **args), training_reward(sparse=[1, 1], **args))


def test_sparse_and_explicit_hybrid_rewards():
    np.testing.assert_allclose(training_reward("sparse", "progress", [-1, 0], [0.9, 0.1], [0, 0]), [-1, 0])
    np.testing.assert_allclose(
        training_reward("robometer", "sparse_plus_progress", [-1, 0], [0.2, 0.8], [0, 0]), [-0.8, 0.8]
    )


def test_prefix_is_causal_and_includes_endpoint():
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(20)]
    selected = prefix_frames(frames[:9], 4)
    assert selected[:, 0, 0, 0].tolist() == [0, 2, 5, 8]
    assert prefix_frames(frames[:1], 8).shape == (1, 2, 2, 3)


@pytest.mark.parametrize(
    "device",
    [
        {"used_mb": 6500, "utilization": 0, "processes": []},
        {"used_mb": 100, "utilization": 90, "processes": []},
        {"used_mb": 0, "utilization": 0, "processes": [{"pid": 99}]},
    ],
)
def test_busy_gpu_never_starts(device):
    with pytest.raises(RuntimeError, match="occupied"):
        assert_idle([0], [{"index": 0, **device}])


def test_idle_gpu_and_missing_device():
    device = {"index": 0, "used_mb": 20, "utilization": 0, "processes": []}
    assert assert_idle([0], [device]) == [device]
    with pytest.raises(RuntimeError, match="does not exist"):
        assert_idle([1], [device])


def test_missing_or_nonfinite_predictions_abort():
    with pytest.raises(ValueError):
        endpoints({}, 1)
    with pytest.raises(ValueError):
        endpoints({"progress_pred": [[float("nan")]], "outputs_success": {"success_probs": [[0.8]]}}, 1)
    result = endpoints({"progress_pred": [[0.2, 0.7]], "outputs_success": {"success_probs": [[[0.1], [0.8]]]}}, 1)
    np.testing.assert_allclose(result, [[0.7, 0.8]])


def test_variable_length_reward_batches_never_use_padding():
    frames = [np.zeros((n, 2, 2, 3), np.uint8) for n in [8, 1, 8, 2, 1, 8]]
    batches = list(frame_batches(frames, 2))
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    assert all(len({len(frames[i]) for i in batch}) == 1 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def test_aliases_cannot_hide_libero_training_leakage(tmp_path):
    categories = tmp_path / "categories.py"
    categories.write_text("DATASET_MAP = {'safe': {'train': ['libero_10']}, 'leaky': {'train': ['libero_90']}}")
    assert verify_training_split(["safe"], categories) == ["libero_10"]
    with pytest.raises(ValueError, match="trained on LIBERO-90"):
        verify_training_split(["leaky"], categories)


def test_quaternion_and_image_convention():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[0, 0] = 255
    raw = {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_eef_pos": np.zeros(3),
        "robot0_gripper_qpos": np.zeros(2),
    }
    obs = observation(raw)
    assert obs["state"].shape == (8,)
    assert (obs["agentview_image"][-1, -1] == 255).all()
    np.testing.assert_equal(raw["robot0_eef_quat"], [0, 0, 0, 1])


def test_replay_preserves_terminal_obs_and_distinguishes_timeout():
    replay = Replay(2, {"state": 1}, 0)
    current, final = {"state": np.array([[1]], dtype=np.float32)}, {"state": np.array([[9]], dtype=np.float32)}
    replay.add(current, np.zeros((1, 7)), [1], final, [False], [True])
    final["state"][0, 0] = 42  # Simulate resetting a live batch after adding.
    sample = replay.sample(4, convert_to_tensors=False)
    assert (sample["next_obs"]["state"] == 9).all()
    assert sample["truncated"].all() and not sample["done"].any()
    replay.add(current, np.zeros((1, 7)), [2], final, [True], [False])
    replay.add(current, np.zeros((1, 7)), [3], final, [True], [False])
    assert len(replay) == 2
    assert sorted(replay.rewards.tolist()) == [2, 3]


def populated_replay():
    replay = Replay(32, {"dino_embedding": 4, "observation/state": 8, "language": 384}, 3)
    obs = {k: np.zeros((8, v.shape[1]), dtype=np.float32) for k, v in replay.obs.items()}
    replay.add(obs, np.zeros((8, 7)), np.zeros(8), obs, np.zeros(8, bool), np.ones(8, bool))
    return replay


def test_native_sac_update_and_checkpoint_roundtrip(cfg, tmp_path):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(123)
    algorithm = make_sac(cfg, populated_replay(), "cpu")
    before = copy.deepcopy(algorithm.actor.state_dict())
    metrics = algorithm.train_step()
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(before[k], v) for k, v in algorithm.actor.state_dict().items())
    save_checkpoint(algorithm, tmp_path / "policy.pt", cfg, 5000)
    other = make_sac(cfg, populated_replay(), "cpu")
    load_actor(other.actor, tmp_path / "policy.pt", cfg, "cpu")
    for k, v in algorithm.actor.state_dict().items():
        torch.testing.assert_close(v, other.actor.state_dict()[k])


def test_native_sac_bootstraps_timeouts(cfg, monkeypatch):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    algorithm = make_sac(cfg, populated_replay(), "cpu")
    for critic in algorithm.critic_target.critics:
        critic.value_head.weight.data.zero_()
        critic.value_head.bias.data.fill_(2.0)
    captured = []
    original = F.mse_loss

    def capture(current, target, **kwargs):
        captured.append(target.detach().clone())
        return original(current, target, **kwargs)

    monkeypatch.setattr(F, "mse_loss", capture)
    algorithm.train_step()
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], cfg["sac"]["gamma"] * 2))


def test_auc_ties_and_undefined_cases():
    assert binary_auc([1, 0.5, 0.5, 0], [1, 1, 0, 0]) == 0.875
    assert binary_auc([0.2, 0.9], [0, 0]) is None
    low, high = wilson(0, 25)
    assert abs(low) < 1e-10 and high > 0
    result = paired_difference(
        [{"init_state_id": i, "success": False} for i in range(10)],
        [{"init_state_id": i, "success": True} for i in range(10)],
    )
    assert result["difference"] == -1 and result["ci_high"] == -1


def test_logger_local_only_and_report_without_results(tmp_path):
    before = set(sys.modules)
    log = LocalLog(tmp_path)
    log.log({"success": 0.5}, 1000, "eval")
    log.episode({"phase": "eval", "success": True})
    log.evaluation(1000, 1, 2, wilson(1, 2))
    log.close()
    assert "wandb" not in set(sys.modules) - before
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["env_steps"] == 1000
    generate_report(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "no_measurements"


def test_report_draws_curves_and_does_not_force_harm(cfg, tmp_path):
    cfg = {**cfg, "num_envs": 8}
    for name, reward in (("sparse", "sparse"), ("robometer", "robometer")):
        folder = tmp_path / name
        run = {**cfg, "name": name, "reward": reward, "comparison": "ood", "ood_claim": "held_out_task"}
        write_json(folder / "config.json", run)
        write_json(folder / "status.json", {"status": "complete"})
        log = LocalLog(folder)
        for step in (0, 100):
            success = step == 100
            for i in range(2):
                log.episode({"phase": "eval", "env_steps": step, "init_state_id": i, "success": success})
            log.evaluation(step, 2 * success, 2, wilson(2 * success, 2))
        log.close()
    generate_report(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["comparisons"][0]["conclusion"] == "no_evidence_of_reward_harm_in_this_seed"
    assert (tmp_path / "performance.png").stat().st_size > 1000
    assert (tmp_path / "performance.pdf").stat().st_size > 1000
    assert (tmp_path / "success_rate_vs_iterations.png").stat().st_size > 1000
    iterations = json.loads((tmp_path / "iteration_curves.json").read_text())
    assert iterations["sparse"][-1]["env_iterations"] == 100 / cfg["num_envs"]


@pytest.mark.parametrize("copy_name", ["runtime", "workspace"])
def test_progress_decoder_always_softmax_and_preserves_target_distributions(copy_name):
    import importlib.util

    import torch

    from experiments.reward_audit.config import REPO
    from experiments.reward_audit.reward_worker import progress_decode_checks

    root = REPO if copy_name == "runtime" else REPO.parent
    path = root / "robometer/robometer/models/utils.py"
    spec = importlib.util.spec_from_file_location(f"decoder_regression_{copy_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Both logit vectors sum to one. Neither may bypass softmax.
    values = torch.tensor([[-6.0, 7.0], [0.25, 0.75]], requires_grad=True)
    actual = module.convert_bins_to_continuous(values)
    expected = torch.tensor([1 / (1 + np.exp(-13)), 1 / (1 + np.exp(-0.5))], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    checks = progress_decode_checks(values.unsqueeze(1), [[v] for v in actual.tolist()])
    assert all(r["legacy_skip_condition_would_trigger"] for r in checks)
    assert not any(r["mismatch"] or r["invalid_output"] for r in checks)
    assert progress_decode_checks(values, [[v] for v in actual.tolist()]) == checks
    numpy_logits = np.array([[-6.0, 7.0], [1000.0, -1000.0]])
    decoded = module.convert_bins_to_continuous(numpy_logits)
    np.testing.assert_allclose(decoded, [1 / (1 + np.exp(-13)), 0.0])
    # Targets are explicitly probabilities, and must not receive softmax twice.
    targets = torch.tensor([[[0.0, 1.0], [0.25, 0.75]]])
    torch.testing.assert_close(module.convert_discrete_target_to_continuous(targets, 2), torch.tensor([[1.0, 0.75]]))
    np.testing.assert_allclose(module.convert_discrete_target_to_continuous(targets.numpy(), 2), [[1.0, 0.75]])
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    torch.testing.assert_close(
        module.convert_discrete_target_to_continuous(indices, 3), torch.tensor([[0.0, 0.5, 1.0]])
    )
    np.testing.assert_allclose(module.convert_discrete_target_to_continuous(indices.numpy(), 3), [[0.0, 0.5, 1.0]])
    # Actual ten-bin logit vector captured from the failed seed 101 rollout.
    failing = torch.tensor(
        [
            3.171875,
            3.234375,
            1.84375,
            0.67578125,
            -0.81640625,
            -1.140625,
            -1.2109375,
            -1.7890625,
            -1.7265625,
            -1.2421875,
        ]
    )
    assert failing.sum().item() == 1.0
    assert module.convert_bins_to_continuous(failing).item() == pytest.approx(0.10182635130285417, abs=2e-7)


def test_progress_decoder_diagnostics_accept_normal_decoding():
    import torch

    from experiments.reward_audit.reward_worker import progress_decode_checks

    logits = torch.tensor([[[2.0, -3.0], [0.0, 0.0]]])
    decoded = torch.softmax(logits, dim=-1)[..., 1].tolist()
    checks = progress_decode_checks(logits, decoded)
    assert not any(record["legacy_skip_condition_would_trigger"] or record["mismatch"] for record in checks)


def test_stove_pair_uses_raw_simulator_reward_and_matching_budgets():
    suite = load_suite(HERE / "configs/stove.json")
    left, right = [materialize(suite, run) for run in suite["runs"]]
    assert left["suite"] == right["suite"] == "libero_goal"
    assert left["task_id"] == right["task_id"] == 7
    assert left["sparse_offset"] == right["sparse_offset"] == 0
    assert right["reward_formula"] == "progress"
    for key in ("steps", "num_envs", "seed", "sac", "initial_actor", "eval_episodes", "horizon"):
        assert left[key] == right[key]


def test_evaluation_uses_simulator_success_and_restores_rng(cfg, tmp_path):
    import random

    import torch

    from experiments.reward_audit.train import evaluate

    cfg = {**cfg, "eval_episodes": 2, "eval_videos": 0}
    image = np.zeros((2, 2, 3), np.uint8)
    obs = {"state": np.zeros(8), "agentview_image": image, "robot0_eye_in_hand_image": image}

    class FakeEnv:
        def __init__(self):
            self.states = []

        def reset(self, indices):
            self.states.append(indices[0])
            return {0: {"obs": obs}}

        def step(self, actions):
            np.random.rand()
            random.random()
            return {0: {"obs": obs, "terminated": False, "truncated": True}}

    class FakeEncoder:
        def encode(self, observations):
            return {"state": np.zeros((len(observations), 8), dtype=np.float32)}

    class FakeActor:
        def act(self, tensors, deterministic):
            assert deterministic
            torch.rand(1)
            return torch.zeros((1, 7)), None

    class FakeReward:
        def predict(self, frames, task):
            return np.ones((len(frames), 2), np.float32)  # Deliberate false positive.

    torch.manual_seed(17)
    np.random.seed(18)
    random.seed(19)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    log, env = LocalLog(tmp_path), FakeEnv()
    result = evaluate(cfg, FakeActor(), FakeEncoder(), env, FakeReward(), "task", log, 0, "cpu", tmp_path)
    log.close()
    assert result["success_rate"] == 0
    assert result["reward_alignment"]["false_positive_rate"] == 1
    assert env.states == [25, 26]
    torch.testing.assert_close(torch.get_rng_state(), torch_before)
    np.testing.assert_equal(np.random.get_state()[1], numpy_before[1])
    assert random.getstate() == python_before


def test_training_reward_windows_use_collected_episodes_and_exact_boundaries():
    from experiments.reward_audit.html_report import training_reward_windows

    records = [
        {"phase": "eval", "env_steps": 5000, "success": True},
        {"phase": "train", "env_steps": 5000, "success": True, "training_return": 1},
        {"phase": "train", "env_steps": 5000, "success": False, "training_return": 0},
        {"phase": "train", "env_steps": 5008, "success": True, "training_return": 1},
        {"phase": "train", "env_steps": 16000, "success": True, "training_return": 1},
    ]
    points = training_reward_windows(records, 12000)
    assert [p["env_steps"] for p in points] == [5000, 10000, 12000]
    assert [p["episodes"] for p in points] == [2, 1, 0]
    assert [p["mean_episode_return"] for p in points] == [.5, 1, None]
    assert [p["success_rewards_per_1000_transitions"] for p in points] == [.2, .2, 0]
    assert [p["partial"] for p in points] == [False, False, True]
    assert training_reward_windows([], 0) == []


def test_training_reward_density_accounts_for_partial_window_exposure():
    from experiments.reward_audit.html_report import training_reward_windows

    records = [{"phase": "train", "env_steps": 5008, "success": True, "training_return": 1}]
    points = training_reward_windows(records, 6000)
    assert points[-1]["mean_episode_return"] == 1
    assert points[-1]["success_rewards_per_1000_transitions"] == 1
    assert points[-1]["partial"]


def test_running_success_auc_weights_irregular_checkpoint_gaps():
    from experiments.reward_audit.html_report import running_auc

    curve = [{"env_steps": 0, "success_rate": 0}, {"env_steps": 10, "success_rate": 1},
             {"env_steps": 40, "success_rate": 0}]
    result = running_auc(curve)
    assert [p["auc"] for p in result] == [0, 5, 20]
    assert [p["normalized_auc"] for p in result] == [None, .5, .5]
    # Worse subsequent performance can lower normalized AUC without losing area.
    result = running_auc(curve + [{"env_steps": 80, "success_rate": 0}])
    assert result[-1]["auc"] == 20
    assert result[-1]["normalized_auc"] == .25


def test_running_success_auc_rejects_undefined_integration_grids():
    from experiments.reward_audit.html_report import running_auc

    assert running_auc([]) == []
    assert running_auc([{"env_steps": 0, "success_rate": 1}])[0]["normalized_auc"] is None
    with pytest.raises(ValueError, match="zero training transitions"):
        running_auc([{"env_steps": 10, "success_rate": .5}])
    with pytest.raises(ValueError, match="strictly increasing"):
        running_auc([{"env_steps": 0, "success_rate": 0}, {"env_steps": 0, "success_rate": .5}])


def _seed_runs():
    from experiments.reward_audit.html_report import LABELS

    suite = load_suite(HERE / "configs/stove.json")
    result = []
    for seed, sparse, model in [(100, 0.2, 0.1), (101, 0.4, 0.2), (102, 0.6, 0.3)]:
        for base in suite["runs"]:
            cfg = materialize(suite, dict(base, seed=seed))
            cfg["eval_every"] = cfg["steps"]
            value = sparse if cfg["reward"] == "sparse" else model
            curve = [{"env_steps": 0, "success_rate": 0.0}, {"env_steps": cfg["steps"], "success_rate": value}]
            result.append(
                {
                    "name": f"{cfg['name']}_seed{seed}",
                    "config": cfg,
                    "complete": True,
                    "final": curve[-1],
                    "curve": curve,
                    "label": LABELS[cfg["reward"]],
                }
            )
    return result


def test_seed_statistics_use_paired_training_seeds_not_pooled_episodes():
    from experiments.reward_audit.html_report import aggregate

    result = aggregate(_seed_runs())
    assert result["completed_paired_seeds"] == [100, 101, 102]
    assert result["final"]["sparse"]["n"] == 3
    assert result["final"]["sparse"]["mean"] == pytest.approx(0.4)
    assert result["final"]["sparse"]["sd"] == pytest.approx(0.2)
    difference = result["paired_final_difference"]
    assert difference["mean"] == pytest.approx(-0.2)
    assert difference["sd"] == pytest.approx(0.1)
    half_width = 4.302652729696142 * 0.1 / np.sqrt(3)
    assert difference["t_ci95"] == pytest.approx([-0.2 - half_width, -0.2 + half_width])
    assert result["paired_auc_difference"]["mean"] == pytest.approx(-0.1)


def test_seed_statistics_exclude_entire_incomplete_pair_and_do_not_invent_sd():
    from experiments.reward_audit.html_report import aggregate

    runs = _seed_runs()
    # One reward arm finishing early must not change the population compared.
    for r in runs:
        if r["config"]["seed"] in [101, 102] and r["config"]["reward"] == "robometer":
            r["complete"] = False
    result = aggregate(runs)
    assert result["completed_paired_seeds"] == [100]
    assert result["final"]["sparse"]["mean"] == 0.2
    assert result["final"]["robometer"]["mean"] == 0.1
    assert result["paired_final_difference"]["sd"] is None
    assert result["paired_final_difference"]["t_ci95"] is None
    assert not result["all_complete"]
    assert all(p["n"] == 1 for p in result["curves"]["sparse"])


def test_seed_statistics_reject_mismatched_settings_and_duplicate_seeds():
    from experiments.reward_audit.html_report import aggregate

    runs = _seed_runs()
    runs[-1]["config"]["horizon"] += 1
    with pytest.raises(ValueError, match="Unmatched"):
        aggregate(runs)
    runs = _seed_runs()
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([*runs, runs[0]])
    runs = _seed_runs()
    runs[0]["curve"] = [runs[0]["curve"][-1]]
    with pytest.raises(ValueError, match="Missing or duplicated"):
        aggregate(runs)
    runs = _seed_runs()
    runs[1]["decoder_mode"] = "upstream_unchanged_with_independent_diagnostics"
    runs[1]["decoder"] = {"branch_hits": 1, "mismatches": 1, "invalid_outputs": 0}
    with pytest.raises(ValueError, match="anomalous pre-fix"):
        aggregate(runs)


def test_direct_coordinator_waits_for_wine_completion(tmp_path, monkeypatch):
    from experiments.reward_audit import parallel

    suite = load_suite(HERE / "configs/stove.json")
    plan = parallel.make_plan(suite, [101, 102])
    wine = tmp_path / "wine"
    write_json(wine / "status.json", {"status": "running"})
    out = tmp_path / "new"
    out.mkdir()
    monkeypatch.setattr(parallel, "generate_html", lambda out: None)
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)

    def forbidden(*args):
        pytest.fail("Coordinator touched GPUs or launched training before wine completed")

    def cancel(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel, "reserve_gpus", forbidden)
    monkeypatch.setattr(parallel, "start_run", forbidden)
    monkeypatch.setattr(parallel.time, "sleep", cancel)
    with pytest.raises(KeyboardInterrupt):
        parallel.execute(plan, out, 1, 0, wine)
    assert json.loads((out / "experiment_status.json").read_text())["status"] == "cancelled"


def test_direct_coordinator_runs_each_seed_once_and_advances_gpu_lanes_independently(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from experiments.reward_audit import parallel

    suite = load_suite(HERE / "configs/stove.json")
    plan = parallel.make_plan(suite, [101, 102])
    started = []

    class Process:
        def __init__(self, cfg, out):
            self.cfg, self.out = cfg, out
            self.pid = 123
            self.returncode = None
            self.remaining = 4 if cfg["reward"] == "robometer" else 1

        def poll(self):
            if self.returncode is None:
                self.remaining -= 1
                if self.remaining <= 0:
                    self.returncode = 0
                    write_json(self.out / "status.json", {"status": "complete"})
            return self.returncode

    def start(cfg, out, gpu, reward_gpu, python):
        assert gpu == reward_gpu == (1 if cfg["reward"] == "sparse" else 0)
        started.append((cfg["reward"], cfg["seed"]))
        return Process(cfg, out)

    monkeypatch.setattr(parallel, "reserve_gpus", lambda indices: nullcontext())
    monkeypatch.setattr(parallel, "preflight", lambda *args: {"ready": True})
    monkeypatch.setattr(parallel, "generate_html", lambda out: None)
    monkeypatch.setattr(parallel, "start_run", start)
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    parallel.execute(plan, tmp_path, 1, 0)
    assert started == [("sparse", 101), ("robometer", 101), ("sparse", 102), ("robometer", 102)]
    status = json.loads((tmp_path / "experiment_status.json").read_text())
    assert status["status"] == "complete"
    assert len(status["runs"]) == 4
    assert all(run["status"] == "complete" for run in status["runs"].values())


def test_direct_coordinator_continues_independent_seed_after_failure(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from experiments.reward_audit import parallel

    suite = load_suite(HERE / "configs/stove.json")
    plan = [cfg for cfg in parallel.make_plan(suite, [101, 102]) if cfg["reward"] == "robometer"]
    started = []

    class Process:
        pid = 123

        def __init__(self, seed):
            self.returncode = 1 if seed == 101 else 0

        def poll(self):
            return self.returncode

    def start(cfg, out, gpu, reward_gpu, python):
        started.append(cfg["seed"])
        if cfg["seed"] == 102:
            write_json(out / "status.json", {"status": "complete"})
        return Process(cfg["seed"])

    def reserve(indices):
        assert indices == [0, 0]
        return nullcontext()

    monkeypatch.setattr(parallel, "reserve_gpus", reserve)
    monkeypatch.setattr(parallel, "preflight", lambda *args: {"ready": True})
    monkeypatch.setattr(parallel, "generate_html", lambda out: None)
    monkeypatch.setattr(parallel, "start_run", start)
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    parallel.execute(plan, tmp_path, 1, 0)
    assert started == [101, 102]
    status = json.loads((tmp_path / "experiment_status.json").read_text())
    assert status["status"] == "failed"  # Preserve the failed seed; do not hide it.
    assert status["runs"]["stove_robometer_seed102"]["status"] == "complete"
