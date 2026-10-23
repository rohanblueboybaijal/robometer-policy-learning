#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

command -v uv >/dev/null || { echo "Install uv and add it to PATH first." >&2; exit 1; }
# HTTPS overrides avoid requiring a GitHub SSH key; preserve pinned revisions.
git -c submodule.LIBERO.url=https://github.com/nakamotoo/LIBERO.git \
    -c submodule.dsrl_openpi.url=https://github.com/jesbu1/dsrl_openpi.git \
    -c submodule.robometer.url=https://github.com/robometer/robometer.git \
    submodule update --init --depth 1

# The parent package's Python 3.11 metadata and Robometer's Python 3.10
# metadata conflict. SAC runs from source in a lean Python 3.10 environment;
# the model lives in a separate Python 3.10 uv environment. No uv sync of the
# parent project's OpenPI / JAX / TensorFlow dependency graph is necessary.
uv venv --python 3.10 .venv-policy
uv venv --python 3.10 .venv-reward
for interpreter in .venv-policy/bin/python .venv-reward/bin/python; do
    uv pip install --python "$interpreter" --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0
done
uv pip install --python .venv-policy/bin/python -r experiments/reward_audit/requirements-policy.lock
uv pip install --python .venv-policy/bin/python --no-deps -e ./LIBERO
uv pip install --python .venv-reward/bin/python -r experiments/reward_audit/requirements-reward.lock
uv pip install --python .venv-reward/bin/python --no-deps -e ./robometer
mkdir -p outputs/reward_audit_setup
uv pip freeze --python .venv-policy/bin/python > outputs/reward_audit_setup/policy-freeze.txt
uv pip freeze --python .venv-reward/bin/python > outputs/reward_audit_setup/reward-freeze.txt
echo "Environments installed. No model weights downloaded and no experiments launched."
