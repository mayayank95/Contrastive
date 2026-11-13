# MixVPR and SALAD uncertainty heads

Training and evaluation for the frozen **MixVPR** and **SALAD** rows of the KappaPlace paper tables. CosPlace training and the rest of the paper code live in the KappaPlace repository.

The backbone and the retrieval aggregator stay frozen. Only the kappa head is trained, with batch leave-one-out vMF on SF-XL place batches (24 places × 20 images).

## Runs that produce the tables

| Row | Encoder | Kappa head | Init |
| --- | --- | --- | --- |
| MixVPR-512 | Frozen ResNet-50 MixVPR, 512-d, 320×320 | Aggregator copy, then `Linear(512, 1)` and Softplus | Build-time weights. The loaded retrieval aggregator is not copied in. |
| SALAD-544, aggregator | Frozen DINOv2 ViT-B/14 + SALAD, 544-d (`dino_salad_512_32`), 322×322 | Aggregator copy. Its forward returns the descriptor **before** the two L2 normalizations. Retrieval still returns the intra-normalized descriptor. | Same random init as MixVPR. |
| SALAD-544, no aggregator | Same frozen SALAD | No aggregator. `Linear(544, 1)` on the pre-L2 token and cluster sum. | The linear layer's default init. |

There is no flat random-image sampling, medoid target, or joint backbone training in this repo.

## Setup

Tested with **Python 3.9**, **PyTorch 2.8.0**, **torchvision 0.23.0**, and **CUDA 12.8**.

```bash
git clone <repository-url>
cd mixvprqsalad-cvpr

python3.9 -m venv .venv
source .venv/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data

Point `data_folder` in the config at the parent of the VPR benchmarks, and `places_root` / `sfxl_train_root` at SF-XL. Layout matches KappaPlace: each benchmark has `train/`, `val/`, and `test/` with `database/` and `queries/`.

SF-XL training reads `Dataframes/{city}.csv` under `places_root` (cities `SF3770`–`SF3781`). Image paths come from the `sfxl_rel_path` column, resolved under `sfxl_train_root`.

| Dataset | Config name | Path under `data_folder` |
| --- | --- | --- |
| SF-XL | `sf_xl` | `SF-XL/processed` |
| Pittsburgh30k | `pitts30k` | `pitts30k/images` |
| MSLS | `msls-val` | `msls` |
| AmsterTime | `amstertime` | `amstertime` |
| Nordland | `nordland` | `nordland` |
| SPED | `spedtest` | `SPEDTEST` |
| SVOX | `svox` | `svox/images` |

## Train

Released checkpoints: [MixVPR](https://github.com/amaralibey/MixVPR) ResNet-50 512-d, and [SALAD](https://github.com/serizba/salad) `dino_salad_512_32`.

```bash
MIXVPR_CKPT=path/to/mixvpr_512.pth ./scripts/train_mixvpr_frozen.sh

SALAD_CKPT=path/to/dino_salad_512_32.ckpt ./scripts/train_salad_frozen.sh
VAR_HEAD=desc SALAD_CKPT=path/to/dino_salad_512_32.ckpt ./scripts/train_salad_frozen.sh
```

`VAR_HEAD=agg` is the aggregator row. `VAR_HEAD=desc` is the SALAD row without an aggregator. Both use place batches only.

## Evaluate

`--var_head_type` must match the checkpoint. Add `--use_labels` for UTM recalls on benchmarks that encode pose in the path (already set in the configs).

```bash
python eval.py --config configs/datasets_mixvpr.json \
  --resume_model path/to/mixvpr_kappa.pth \
  --var_head_type agg

python eval.py --config configs/datasets_salad.json \
  --resume_model path/to/salad_agg_kappa.pth \
  --var_head_type agg

python eval.py --config configs/datasets_salad.json \
  --resume_model path/to/salad_desc_kappa.pth \
  --var_head_type descriptor
```

`eval.py` reports recall@K, ECE, and the rejection numbers used in the tables.

## Layout

| Path | Role |
| --- | --- |
| `train.py` | Frozen training, batch leave-one-out vMF |
| `eval.py` | Recall, ECE, rejection |
| `configs/` | MixVPR-512 and SALAD-544 configs |
| `models/` | MixVPR, SALAD, kappa head |
| `data/` | SF-XL place batches and benchmark loaders |
| `losses/` | vMF NLL and batch place-centroid targets |
| `scripts/` | The three paper training commands |

## Citation

```bibtex
@article{anonymous2026kappaplace,
  title={KappaPlace: Learning Hyperspherical Uncertainty for Visual Place Recognition via Prototype-Anchored Supervision},
  author={Anonymous},
  year={2026}
}
```

Also cite [MixVPR](https://github.com/amaralibey/MixVPR) and [SALAD](https://github.com/serizba/salad).
