"""Small integration layer around this repository's SAC and MLP networks."""

from __future__ import annotations

import numpy as np


class Replay:
    """Store features only; never retain full videos in replay."""

    def __init__(self, capacity, dimensions, seed, action_dim=7):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.obs = {k: np.empty((capacity, d), dtype=np.float32) for k, d in dimensions.items()}
        self.next_obs = {k: np.empty_like(v) for k, v in self.obs.items()}
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)

    def __len__(self):
        return self.size

    def add(self, obs, actions, rewards, next_obs, done, truncated):
        for i in range(len(actions)):
            index = self.position
            for key in self.obs:
                self.obs[key][index] = obs[key][i]
                self.next_obs[key][index] = next_obs[key][i]
            self.actions[index], self.rewards[index] = actions[i], rewards[i]
            self.done[index], self.truncated[index] = done[i], truncated[i]
            self.position = (index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, convert_to_tensors=True, device="cpu"):
        import torch

        if not self.size:
            raise RuntimeError("Empty replay")
        index = self.rng.integers(self.size, size=batch_size)

        def tensor(array):
            return torch.as_tensor(array[index], device=device) if convert_to_tensors else array[index]

        return {
            "obs": {k: tensor(v) for k, v in self.obs.items()},
            "next_obs": {k: tensor(v) for k, v in self.next_obs.items()},
            "action": tensor(self.actions),
            "reward": tensor(self.rewards),
            "done": tensor(self.done),
            "truncated": tensor(self.truncated),
        }


def make_sac(cfg, replay, device, action_dim=7):
    import gymnasium as gym

    from robometer_policy_learning.algorithms.sac import SAC, SACConfig
    from robometer_policy_learning.modules.mlp import MLPActor, MLPActorConfig, MLPCritic, MLPCriticConfig

    observation_space = gym.spaces.Dict(
        {k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape[1:], dtype=np.float32) for k, v in replay.obs.items()}
    )
    action_space = gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32)
    featurizer = {k: [384] for k in replay.obs}
    actor = MLPActor(
        MLPActorConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=tuple(cfg["actor_hidden"]),
            use_tanh_output=True,
            log_std_init=0.0,
            featurizer=featurizer,
            remove_obs_keys=[],
        )
    ).to(device)
    critic = MLPCritic(
        MLPCriticConfig(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=tuple(cfg["critic_hidden"]),
            use_layer_norm=True,
            featurizer=featurizer,
            remove_obs_keys=[],
        )
    ).to(device)
    params = dict(cfg["sac"])
    # Disable KL-to-old-actor: this is the paper's SAC, not diffusion steering.
    params.update(
        actor=actor,
        critic=critic,
        buffer=replay,
        logger=None,
        action_space=action_space,
        train_actor_with_kl_divergence=False,
    )
    return SAC(SACConfig(**params))


class Dino:
    def __init__(self, checkpoint, device):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval().requires_grad_(False)
        self.dimension = self.model.config.hidden_size
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, observations):
        import torch

        inputs = self.processor(images=[o["agentview_image"] for o in observations], return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        return {
            "dino_embedding": outputs.pooler_output.float().cpu().numpy(),
            "observation/state": np.stack([o["state"] for o in observations]),
            "language": np.zeros((len(observations), 384), dtype=np.float32),
        }


def actor_actions(actor, features, device, deterministic=False):
    import torch

    tensors = {k: torch.as_tensor(v, device=device) for k, v in features.items()}
    with torch.no_grad():
        actions, _ = actor.act(tensors, deterministic=deterministic)
    return actions.cpu().numpy()


def save_checkpoint(algorithm, path, cfg, env_steps):
    from pathlib import Path

    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config": cfg,
        "env_steps": env_steps,
        "actor": algorithm.actor.state_dict(),
        "critic": algorithm.critic.state_dict(),
        "critic_target": algorithm.critic_target.state_dict(),
        "actor_optimizer": algorithm.actor_optimizer.state_dict(),
        "critic_optimizer": algorithm.critic_optimizer.state_dict(),
        "ent_coef_optimizer": algorithm.ent_coef_optimizer.state_dict(),
        "log_ent_coef": algorithm.log_ent_coef.detach().cpu(),
        "updates": algorithm._n_updates,
    }
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def load_actor(actor, path, cfg, device):
    """Warm-start only; fresh replay/optimizers. Loading isn't an exact resume."""
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "config" in checkpoint:
        prior = checkpoint["config"]
        for key in ("task", "dino_checkpoint", "actor_hidden"):
            if prior[key] != cfg[key]:
                raise ValueError(f"Warm-start {key} differs; pass a checkpoint for this task and encoder")
    actor.load_state_dict(checkpoint.get("actor", checkpoint))
