from losses.vmf_loss import VMFLikelihood


def get_loss(loss_name, active_losses, uncertainty_loss, descriptors_dimension=512):
    """Batch leave-one-out vMF. The frozen paper runs do not use a basic metric loss."""
    del loss_name
    if "uncertainty" not in active_losses:
        raise ValueError("Training requires --losses uncertainty")
    if str(uncertainty_loss).lower() != "vmf":
        raise ValueError(
            f"uncertainty_loss must be vmf, got {uncertainty_loss!r}."
        )
    return None, VMFLikelihood(d=int(descriptors_dimension))
