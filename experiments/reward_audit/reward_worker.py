"""Uses the real Robometer loader and collator, isolated from the policy runtime."""

from __future__ import annotations

import argparse
import ast
import json
import os
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

from .config import write_json


def progress_decode_checks(logits, decoded):
    """Observe decoding, including cases where the removed bypass would trigger."""

    logits = logits.detach().cpu().float()
    records = []
    for values, observed in zip(logits, decoded, strict=False):
        # Upstream squeezes the frame dimension for single-frame inputs.
        values = values.reshape(-1, values.shape[-1])
        sums = values.sum(dim=-1)
        bypass = bool((sums == 1).all())
        array = values.numpy().astype(np.float64)
        exponentials = np.exp(array - array.max(axis=-1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
        reference = (probabilities * np.linspace(0, 1, array.shape[-1])).sum(axis=-1)
        observed = np.asarray(observed, dtype=float).reshape(-1)
        shape_matches = observed.shape == reference.shape
        finite = bool(np.isfinite(array).all() and np.isfinite(observed).all())
        difference = float(np.max(np.abs(observed - reference))) if shape_matches and finite else None
        invalid = not finite or bool(((observed < 0) | (observed > 1)).any())
        mismatch = not shape_matches or difference is None or difference > 2e-6

        def numbers(vector):
            return [float(value) if np.isfinite(value) else None for value in vector]

        record = {
            "legacy_skip_condition_would_trigger": bypass,
            "logit_sums": numbers(sums.numpy()),
            "observed": numbers(observed),
            "independent_softmax": numbers(reference),
            "max_abs_difference": difference,
            "mismatch": mismatch,
            "invalid_output": invalid,
        }
        if bypass or mismatch or invalid:
            record["raw_logits"] = [numbers(row) for row in array]
        records.append(record)
    return records


def preserve_final_hidden_state(module, inputs, output):
    """Unsloth Qwen3 omits the hidden_states tuple but returns its final layer.

    Robometer consumes only the last entry of that tuple. Restore that entry
    from the identical last_hidden_state tensor, without changing model math.
    """
    if getattr(output, "hidden_states", None) is None:
        final = getattr(output, "last_hidden_state", None)
        if final is None:
            raise ValueError("Qwen3 returned neither hidden_states nor last_hidden_state")
        output.hidden_states = (final,)
    return output


def frame_batches(frame_sets, batch_size):
    # Robometer pads per-frame heads within a batch. Equal frame counts ensure
    # the last prediction is the true endpoint, rather than a padded token.
    buckets = {}
    for index, frames in enumerate(frame_sets):
        buckets.setdefault(len(frames), []).append(index)
    for indices in buckets.values():
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]


def verify_training_split(training_datasets, category_path):
    tree = ast.parse(Path(category_path).read_text())
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DATASET_MAP" for t in node.targets
        ):
            aliases = ast.literal_eval(node.value)
    expanded = []
    for name in training_datasets:
        expanded.extend(aliases[name]["train"] if name in aliases else [name])
    seen = [name for name in expanded if "libero" in name.lower() and ("90" in name or "all" in name)]
    if seen:
        raise ValueError(f"Reward checkpoint was trained on LIBERO-90: {seen}; held-out task claim invalid")
    return expanded


def endpoints(output, expected):
    # Success logits may have an extra singleton dimension. Missing/malformed
    # predictions are fatal: never silently replace a model failure by zero.
    progress = output.get("progress_pred")
    success = output.get("outputs_success", {}).get("success_probs")
    if progress is None or success is None or len(progress) != expected or len(success) != expected:
        raise ValueError("Incorrect number of Robometer predictions")
    result = []
    for p, s in zip(progress, success, strict=False):
        p, s = np.asarray(p).reshape(-1), np.asarray(s).reshape(-1)
        if not len(p) or not len(s):
            raise ValueError("Empty Robometer head output")
        values = np.array([p[-1], s[-1]], dtype=float)
        if not np.isfinite(values).all() or (values < -1e-5).any() or (values > 1 + 1e-5).any():
            raise ValueError(f"Invalid progress/success prediction: {values}")
        result.append(values.clip(0, 1).tolist())
    return np.asarray(result, dtype=np.float32)


class Robometer:
    def __init__(self, cfg, out=None):
        import torch
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        torch.set_num_threads(cfg["torch_threads"])
        self.device = torch.device("cuda:0")
        self.exp, self.tokenizer, processor, self.model = load_model_from_hf(cfg["reward_checkpoint"], self.device)
        self.model.eval().requires_grad_(False)
        self.out = Path(out) if out is not None else None
        self.decode_counts = {
            "decoder": "always_softmax_v1",
            "batches": 0,
            "samples": 0,
            "legacy_condition_hits": 0,
            "mismatches": 0,
            "invalid_outputs": 0,
        }
        self._progress_logits = None

        def capture_progress(module, inputs, output):
            self._progress_logits = output[0].progress_logits["A"].detach()

        if self.exp.loss.progress_loss_type == "discrete":
            self.model.register_forward_hook(capture_progress)
        self.inference_dtype = getattr(torch, self.exp.model.torch_dtype)
        if "Qwen3" in self.exp.model.base_model_id:
            self.model.model.register_forward_hook(preserve_final_hidden_state)
        self.collator = setup_batch_collator(processor, self.tokenizer, self.exp, is_eval=True)
        self.batch_size = cfg["reward_batch_size"]
        self.metadata = {
            "checkpoint": cfg["reward_checkpoint"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "training_datasets": self.exp.data.train_datasets,
            "model_type": self.exp.model.model_type,
            "progress_loss_type": self.exp.loss.progress_loss_type,
            "inference_dtype": str(self.inference_dtype),
            "autocast": self.inference_dtype in (torch.bfloat16, torch.float16),
            "qwen3_final_hidden_state_adapter": "Qwen3" in self.exp.model.base_model_id,
            "progress_decoder": "always_softmax_v1_with_independent_diagnostics",
        }
        if cfg.get("ood_claim") == "held_out_task":
            import robometer

            category_path = Path(robometer.__file__).parent / "data/dataset_category.py"
            self.metadata["expanded_training_datasets"] = verify_training_split(
                self.exp.data.train_datasets, category_path
            )

    def predict(self, frame_sets, task):
        import torch
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs

        results = np.empty((len(frame_sets), 2), dtype=np.float32)
        for indices in frame_batches(frame_sets, self.batch_size):
            samples = []
            for i in indices:
                frames = frame_sets[i]
                trajectory = Trajectory(
                    frames=frames,
                    frames_shape=tuple(frames.shape),
                    task=task,
                    id=str(i),
                    metadata={"subsequence_length": len(frames)},
                    video_embeddings=None,
                )
                samples.append(ProgressSample(trajectory=trajectory, sample_type="progress"))
            batch = self.collator(samples)["progress_inputs"]
            batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
            # The processor emits float32 pixels and Unsloth may promote
            # intermediate activations. Match the checkpoint precision across
            # floating-point operations, as in mixed-precision training.
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.inference_dtype,
                    enabled=self.inference_dtype in (torch.bfloat16, torch.float16),
                ),
            ):
                outputs = compute_batch_outputs(
                    self.model,
                    self.tokenizer,
                    batch,
                    "progress",
                    is_discrete_mode=self.exp.loss.progress_loss_type == "discrete",
                    num_bins=self.exp.loss.progress_discrete_bins,
                )
            if self._progress_logits is not None and self.out is not None:
                checks = progress_decode_checks(self._progress_logits, outputs["progress_pred"])
                self.decode_counts["batches"] += 1
                self.decode_counts["samples"] += len(checks)
                for record in checks:
                    self.decode_counts["legacy_condition_hits"] += record["legacy_skip_condition_would_trigger"]
                    self.decode_counts["mismatches"] += record["mismatch"]
                    self.decode_counts["invalid_outputs"] += record["invalid_output"]
                anomaly = any(
                    record["legacy_skip_condition_would_trigger"] or record["mismatch"] or record["invalid_output"]
                    for record in checks
                )
                if anomaly or self.decode_counts["batches"] <= 5 or self.decode_counts["batches"] % 100 == 0:
                    write_json(self.out / "decoder_checks.json", self.decode_counts)
                    with (self.out / "decoder_checks.jsonl").open("a") as log:
                        log.write(
                            json.dumps(
                                {
                                    "batch": self.decode_counts["batches"],
                                    "task": task,
                                    "sample_indices": indices,
                                    "checks": checks,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                if anomaly:
                    archive = self.out / f"decoder_anomaly_{self.decode_counts['batches']}.npz"
                    np.savez_compressed(
                        archive,
                        frames=np.stack([frame_sets[i] for i in indices]),
                        logits=self._progress_logits.cpu().float().numpy(),
                        task=np.asarray(task),
                    )
            # Preserve upstream scores. Invalid scores still abort, after the
            # raw logits and independent reference have been saved locally.
            results[indices] = endpoints(outputs, len(samples))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    # Listen first: client can enforce a timeout during model load as well.
    with Listener(args.socket, family="AF_UNIX", authkey=bytes.fromhex(os.environ["AUDIT_RPC_KEY"])) as listener:
        with listener.accept() as connection:
            backend = None
            try:
                backend = Robometer(cfg, args.out)
                write_json(Path(args.out) / "reward_provenance.json", backend.metadata)
                connection.send({"ok": True, "result": backend.metadata})
                while True:
                    request = connection.recv()
                    if request is None:
                        break
                    connection.send({"ok": True, "result": backend.predict(request["frames"], request["task"])})
            except (EOFError, BrokenPipeError):
                pass
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
            finally:
                if backend is not None:
                    write_json(Path(args.out) / "decoder_checks.json", backend.decode_counts)


if __name__ == "__main__":
    main()
