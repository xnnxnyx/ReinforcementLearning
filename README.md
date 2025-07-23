# Frozen Lake

## Environment setup

Install the _exact_ package versions used by the starter code:

```bash
pip install numpy
pip install gymnasium
pip install pygame
```

## Project Files

| File                | Purpose                                                             |
| ------------------- | ------------------------------------------------------------------- |
| **`q_learning.py`** | Q‑learning agent with _linear_ function approximation.              |
| **`reinforce.py`**  | REINFORCE agent with a linear _soft‑max_ policy.                    |
| **`features.py`**   | Common feature extractor for both agents (tabular & custom linear). |

- `env.py` – Frozen‑Lake environment
- `main.py` – Training/​evaluation driver
- `test_q_learning.py`, `test_reinforce.py` – Unit tests

---

## Training & testing

### Unit tests (no randomness permitted)

```bash
pytest test_q_learning.py
pytest test_reinforce.py
```

### Quick start (tabular features)

```bash
# Q‑learning
python main.py --algorithm q_learning --feature_type tabular

# REINFORCE
python main.py --algorithm reinforce --feature_type tabular
```

During training, average return is printed every 100 trajectories.  
Disable rendering to speed things up: `--disable_render`.

---

## Saving & loading agents

A trained agent is automatically saved as:

```
params-<algo>-<feature>-<timestamp>.pkl
```

Reload it with:

```bash
python main.py --algorithm q_learning  --feature_type tabular                --evaluate --load_params path/to/params.pkl
```

(Swap `q_learning`/`reinforce` as required.)

---

## Designing **linear** features

Add new feature extractors in `features.py`.  
Use them via:

```bash
--feature_type linear          # your custom features
--feature_type linear_ref      # provided baseline
```

Evaluate generalisation on unseen maps by varying `--env_seed` and `--map_size`.

