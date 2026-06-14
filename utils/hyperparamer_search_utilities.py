import torch
import numpy as np
import random
import logging
from torch.utils.data import IterableDataset, get_worker_info
from copy import deepcopy

class SubsampledShardedDataset(IterableDataset):
    """
    A wrapper to limit the number of shards seen in a single 'epoch'.
    This allows for much faster iterations during HPO.

    Designed for use with ShardedNPZDataset (IterableDataset). Each worker
    fork gets its own copy, so temporarily swapping shard_paths inside
    __iter__ is safe: the main-process dataset is never mutated.
    """
    def __init__(self, base_dataset, max_shards=10, seed=None):
        super().__init__()
        self.base = base_dataset
        self.max_shards = max_shards
        self.seed = seed
        self._epoch = 0  # bumped each __iter__ for varied subsets

        # Forward key attributes so downstream code (conditional wrapper,
        # task filter, dataloader setup) works transparently.
        self.n_tasks = base_dataset.n_tasks
        self.celltype_mapping = getattr(base_dataset, "celltype_mapping", None)
        self.shard_paths = base_dataset.shard_paths  # read-only reference for inspection

    def __iter__(self):
        full_shard_list = list(self.base.shard_paths)

        # Deterministic-per-epoch shuffle so results are reproducible
        rng = random.Random(self.seed + self._epoch if self.seed is not None else None)
        rng.shuffle(full_shard_list)
        subset = full_shard_list[:self.max_shards]
        self._epoch += 1

        # Temporarily swap shard_paths on the *forked* copy of base
        original_paths = self.base.shard_paths
        self.base.shard_paths = subset

        try:
            for item in self.base:
                yield item
        finally:
            # Restore even if the iterator is abandoned early
            self.base.shard_paths = original_paths


def freeze_trunk(model, freeze_lora=True):
    """
    Freeze everything except the head for head-only warmup.

    After calling this, only parameters whose name contains 'head' will
    have requires_grad=True.  Call ``unfreeze_trunk`` to restore.

    Args:
        model: LightningModel (possibly PEFT-wrapped).
        freeze_lora: If True, also freeze LoRA adapter weights.
    """
    frozen, kept = 0, 0
    for name, param in model.named_parameters():
        if "head" in name and "original_module" not in name:
            param.requires_grad = True
            kept += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    logging.info(f"[freeze_trunk] Frozen {frozen:,} params, head kept trainable ({kept:,} params)")


def unfreeze_trunk(model, layers_to_keep_frozen=None):
    """
    Re-enable gradients for the trunk (reverses ``freeze_trunk``).

    Args:
        model: LightningModel.
        layers_to_keep_frozen: Optional list of name fragments that should
            stay frozen (e.g. ["conv_tower"]).
    """
    layers_to_keep_frozen = layers_to_keep_frozen or []
    unfrozen = 0
    for name, param in model.named_parameters():
        if any(frag in name for frag in layers_to_keep_frozen):
            continue
        param.requires_grad = True
        unfrozen += param.numel()
    logging.info(f"[unfreeze_trunk] Re-enabled {unfrozen:,} params")


def get_lr_finder_logic(trainer, model, dataloader):
    """
    Run the PyTorch Lightning LR finder and return the suggested LR.

    Requires that ``model.configure_optimizers`` and ``model.training_step``
    are already set up (which ``main_npz.py`` does before calling this).
    """
    from pytorch_lightning.tuner import Tuner
    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(model, train_dataloaders=dataloader)

    new_lr = lr_finder.suggestion()
    logging.info(f"[LR Finder] Suggested Learning Rate: {new_lr}")

    # Save the plot if matplotlib is available (non-blocking)
    try:
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder_plot.png", dpi=150)
        logging.info("[LR Finder] Saved plot to lr_finder_plot.png")
    except Exception:
        pass

    return new_lr


def check_loss_balance(model, dataloader, alphas=None, max_batches=32):
    """
    Probing tool: evaluate how different ``alpha_shape`` values affect loss
    magnitude on a small validation mini-batch.

    Works for both standard multi-task and conditional-head models.

    Args:
        model: LightningModel with a CompoundPSILoss attached as ``model.loss``.
        dataloader: A DataLoader yielding batches in the same format as training.
        alphas: List of alpha_shape values to test.
        max_batches: Number of batches to average over.

    Returns:
        Dict[float, float] mapping alpha_shape -> mean loss.
    """
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]

    if not hasattr(model.loss, "alpha_shape"):
        logging.warning("[check_loss_balance] model.loss has no alpha_shape attribute; skipping")
        return {}

    original_alpha = model.loss.alpha_shape
    device = next(model.parameters()).device
    model.eval()

    results = {a: [] for a in alphas}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            # Flexible unpacking (mirrors custom_step logic)
            x = batch[0].to(device)
            y = batch[1].to(device)
            mask, confidence, cell_type_id = None, None, None

            is_conditional = hasattr(model.loss, "alpha_shape")  # proxy; real check below
            if len(batch) >= 5:
                mask = batch[2].to(device)
                confidence = batch[3].to(device)
                cell_type_id = batch[4].to(device)
            elif len(batch) == 4:
                mask = batch[2].to(device)
                if batch[3].dtype == torch.long:
                    cell_type_id = batch[3].to(device)
                else:
                    confidence = batch[3].to(device)
            elif len(batch) == 3:
                if batch[2].dtype == torch.long:
                    cell_type_id = batch[2].to(device)
                else:
                    mask = batch[2].to(device)

            if y.ndim == 2:
                y = y.unsqueeze(0)
            if mask is not None and mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if confidence is not None and confidence.ndim == 2:
                confidence = confidence.unsqueeze(0)

            # Forward
            if cell_type_id is not None:
                logits = model((x, cell_type_id), logits=True)
            else:
                logits = model(x, logits=True)

            # Evaluate each alpha
            for a in alphas:
                model.loss.alpha_shape = a
                try:
                    loss = model.loss(logits, y, mask=mask, confidence=confidence)
                except TypeError:
                    loss = model.loss(logits, y)
                results[a].append(loss.item())

    # Restore
    model.loss.alpha_shape = original_alpha
    model.train()

    avg_results = {a: np.mean(vals) if vals else float("nan") for a, vals in results.items()}
    logging.info("[check_loss_balance] alpha_shape -> mean loss:")
    for a, v in sorted(avg_results.items()):
        logging.info(f"  alpha_shape={a:.3f}  ->  loss={v:.6f}")
    return avg_results