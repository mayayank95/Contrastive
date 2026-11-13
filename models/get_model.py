import logging
from typing import Any, Dict

import torch

from models.model_mode import build_model_mode

logger = logging.getLogger(__name__)


def _load_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    state_dict_key: str = "model_state_dict",
):
    """Load model weights (strict=False).

    A retrieval checkpoint fills the backbone and the retrieval aggregator.
    The uncertainty aggregator's weights are not in that checkpoint, so that
    head stays at the initialization it had when the model was built.
    A checkpoint that already stores ``var_head`` keeps those weights.
    """
    logger.info("Loading model weights from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state_dict_key in checkpoint and isinstance(checkpoint[state_dict_key], dict):
        state_dict = checkpoint[state_dict_key]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if any(k.startswith("core.") for k in state_dict):
        state_dict = {k[len("core.") :]: v for k, v in state_dict.items() if k.startswith("core.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("State dict missing keys (left at default init): %s", missing)
    if unexpected:
        logger.debug("State dict unexpected keys (ignored): %s", unexpected)


def get_model(args: Dict[str, Any]) -> torch.nn.Module:
    """Build a frozen MixVPR or SALAD uncertainty model and load a checkpoint."""
    method = str(args["method"]).lower()
    if method not in ("mixvpr", "salad"):
        raise ValueError(f"Unknown method: {method}. Expected mixvpr or salad.")

    model = build_model_mode(args)
    resume_path = args.get("resume_model")
    if resume_path is not None:
        state_dict_key = args.get("ckpt_state_dict_key", "model_state_dict")
        _load_weights(model, resume_path, state_dict_key=state_dict_key)
    else:
        logger.info("No --resume_model provided. Model uses its build-time weights.")

    if args.get("freeze_model"):
        trainable = model.freeze_base()
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        logger.debug("Base model frozen, training %d params: %s", len(trainable), trainable_names)

    return model
