# Contrastive

Training code for the frozen **MixVPR** and **SALAD** uncertainty heads in **KappaPlace**.

The retrieval model stays frozen. Only the kappa head is trained, with SGD at `1e-3`, on SF-XL place batches of 24 places × 20 images. The anchor for each image is the normalized mean of the other 19 retrieval descriptors from the same place. Early stopping watches validation ECE@1 with patience 15. Training-time ECE uses percentile bins.

## 🛠 Setup

Tested with **Python 3.9**, **PyTorch 2.8.0**, **torchvision 0.23.0**, and **CUDA 12.8** (cu128 wheel). Paper results used an NVIDIA driver reporting **CUDA 13.2**. A GPU is strongly recommended.

```bash
git clone <contrastive-repository-url>
cd contrastive

python3.9 -m venv .venv
source .venv/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Contrastive training

Runnable copies are `scripts/train_mixvpr_frozen.sh` and `scripts/train_salad_frozen.sh`.

| Row | Command |
| --- | --- |
| MixVPR-512, aggregator head | `MIXVPR_CKPT=path/to/mixvpr_512.pth bash scripts/train_mixvpr_frozen.sh` |
| SALAD-544, descriptor head | `SALAD_CKPT=path/to/dino_salad_512_32.ckpt bash scripts/train_salad_frozen.sh` |

SALAD training uses the descriptor head: `Linear(544, 1)` and Softplus on the pre-L2 token and cluster sum. `USE_WANDB=1` turns on Weights & Biases. Checkpoints are written under `logs/MixVPR` or `logs/SALAD`.

## Evaluation

Evaluate the saved checkpoint in KappaPlace.

## ⚙️ Configuration

`configs/datasets_mixvpr.json` is MixVPR-512 at 320×320 with the aggregator head. `configs/datasets_salad.json` is SALAD-544 at 322×322 with the descriptor head. JSON defaults (`--config`) merge with command-line overrides (`configs/parser.py`).

Run `python train.py --help` for the full list.

## 📁 Project layout

| Path | Role |
| --- | --- |
| `train.py` | Frozen MixVPR and SALAD kappa-head training |
| `eval.py` | Retrieval metrics and ECE for a saved checkpoint |
| `configs/` | MixVPR-512 and SALAD-544 configs |
| `models/` | MixVPR, SALAD, kappa head |
| `data/` | SF-XL place batches and benchmark loaders |
| `losses/` | Batch leave-one-out vMF |
| `eval_metrics/` | ECE, retrieval, uncertainty |
| `utils/` | Runtime, early stopping, W&B |
| `scripts/` | Frozen MixVPR and SALAD training commands |
| `validation.py` | Recall tables during training validation |

## ❤️ Acknowledgements

- [CosPlace](https://github.com/gmberton/CosPlace) — SF-XL place groups used for training.
- [MixVPR](https://github.com/amaralibey/MixVPR) — feature-mixing aggregator (these runs use the 512-d ResNet-50 model).
- [SALAD](https://github.com/serizba/salad) — DINOv2 optimal-transport aggregator (these runs use 544-d `dino_salad_512_32`).

## 📖 Citation

```bibtex
@article{anonymous2026kappaplace,
  title={KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision},
  author={Anonymous},
  year={2026}
}
```

Also cite [CosPlace](https://github.com/gmberton/CosPlace), [MixVPR](https://github.com/amaralibey/MixVPR), and [SALAD](https://github.com/serizba/salad).
