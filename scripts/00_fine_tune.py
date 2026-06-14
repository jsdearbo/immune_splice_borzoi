import os
import sys
import logging
import argparse
import yaml
import torch

import grelu.resources
from types import SimpleNamespace, MethodType
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Monkeypatch for EarlyStoppingReason (fix for PL 2.4/2.5 mismatch)
import pytorch_lightning.callbacks.early_stopping as es
if not hasattr(es, "EarlyStoppingReason"):
    class EarlyStoppingReason:
        NOT_STOPPED = "NOT_STOPPED"
    es.EarlyStoppingReason = EarlyStoppingReason

# Path setup: add repo utils/ to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
_utils_dir = os.path.join(_repo_root, "utils")
for _p in [_repo_root, _utils_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sharded_dataset import ShardedNPZDataset, ConditionalCellTypeWrapper
from lora_patch import attach_lora_methods, apply_custom_head, configure_optimizers_lora
from hyperparamer_search_utilities import SubsampledShardedDataset, freeze_trunk, unfreeze_trunk
from losses import FocalLoss, WeightedPSILoss, PSILoss, BhattacharyyaLoss, CompoundPSILoss, TaskDiversityLoss, StrongPSILoss

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_weights_only_from_checkpoint(model, ckpt_path):
    """Load only matching model weights and skip optimizer/epoch state."""
    logging.info(f"Weight-only init from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        source_state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        source_state = ckpt
    else:
        raise ValueError("Checkpoint format not understood for weight-only loading")

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = 0
    skipped_missing = 0

    for key, value in source_state.items():
        if key not in model_state:
            skipped_missing += 1
            continue
        if model_state[key].shape != value.shape:
            skipped_shape += 1
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logging.info(
        "Weight-only load summary: matched=%d, skipped_missing=%d, skipped_shape=%d, "
        "missing_after_load=%d, unexpected_after_load=%d",
        len(filtered_state), skipped_missing, skipped_shape, len(missing), len(unexpected)
    )

def conditional_wrapper_len(self):
    if self.mode == 'expand':
        # In 'expand' mode, each original item becomes n_tasks items
        return len(self.dataset) * self.dataset.n_tasks
    else:
        # In 'random' mode, the length is the same as the original dataset
        return len(self.dataset)

def custom_step_impl(model, batch, batch_idx, is_val=False):
    x, y, mask, confidence, cell_type_id = None, None, None, None, None
    is_conditional = model.model_params.get("conditional_head", False)
    
    # Infer load_mask from model parameters
    train_params = model.train_params
    load_mask = True
    if "loss_params" in train_params:
        lp = train_params["loss_params"]
        load_mask = lp.get("use_mask", lp.get("psi_params", {}).get("use_mask", True))

    if is_conditional and len(batch) >= 2:
        x = batch[0]
        y = batch[1]
        cell_type_id = batch[-1]
        if len(batch) == 4:
            if load_mask:
                mask = batch[2]
            else:
                confidence = batch[2]
        elif len(batch) == 5:
            mask = batch[2]
            confidence = batch[3]
    else:
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            if load_mask:
                x, y, mask = batch
            else:
                x, y, confidence = batch
        elif len(batch) == 4:
            x, y, mask, confidence = batch

    if isinstance(mask, list):
        mask = None
    if isinstance(confidence, list):
        confidence = None
    if mask is not None and mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if confidence is not None and confidence.ndim == 2:
        confidence = confidence.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)

    if is_conditional:
        logits = model((x, cell_type_id), logits=True)
    else:
        logits = model(x, logits=True)

    prefix = "val" if is_val else "train"

    try:
        loss_out = model.loss(logits, y, mask=mask, confidence=confidence)
    except TypeError:
        loss_out = model.loss(logits, y)

    if isinstance(model.loss, StrongPSILoss):
        primary_loss, bce_l, cell_l, prof_l = loss_out
        model.log(f"{prefix}_bce_loss", bce_l, prog_bar=False, on_step=not is_val, on_epoch=True)
        model.log(f"{prefix}_cell_loss", cell_l, prog_bar=False, on_step=not is_val, on_epoch=True)
        model.log(f"{prefix}_prof_loss", prof_l, prog_bar=False, on_step=not is_val, on_epoch=True)
    else:
        primary_loss = loss_out

    total_loss = primary_loss
    model.log(f"{prefix}_loss", total_loss, prog_bar=True, on_step=not is_val, on_epoch=True)

    if is_val:
        model.val_losses.append(total_loss)

        with torch.no_grad():
            preds = torch.sigmoid(logits)
            if preds.ndim == 3:
                preds_flat = preds.permute(0, 2, 1).reshape(-1, preds.shape[1])
                y_flat = y.permute(0, 2, 1).reshape(-1, y.shape[1])
                if mask is not None:
                    mask_flat = mask.permute(0, 2, 1).reshape(-1, mask.shape[1])
                    valid = mask_flat.sum(dim=1) > 0
                    preds_flat = preds_flat[valid]
                    y_flat = y_flat[valid]
            else:
                preds_flat = preds.reshape(-1, 1)
                y_flat = y.reshape(-1, 1)

            if preds_flat.numel() > 1:
                n_t = preds_flat.shape[1] if preds_flat.ndim > 1 else 1
                pearson_sum = 0.0
                for t_idx in range(n_t):
                    p = preds_flat[:, t_idx] if n_t > 1 else preds_flat.squeeze()
                    tgt = y_flat[:, t_idx] if n_t > 1 else y_flat.squeeze()
                    if p.std() > 1e-8 and tgt.std() > 1e-8:
                        pearson_sum += torch.corrcoef(torch.stack([p, tgt]))[0, 1].item()
                val_pearson = pearson_sum / max(n_t, 1)
                val_mae = (preds_flat - y_flat).abs().mean().item()

                model.log("val_pearson", val_pearson, prog_bar=True, on_step=False, on_epoch=True)
                model.log("val_mae", val_mae, prog_bar=False, on_step=False, on_epoch=True)

    return total_loss

def custom_training_step(self, batch, batch_idx):
    return custom_step_impl(self, batch, batch_idx, is_val=False)

def custom_validation_step(self, batch, batch_idx):
    return custom_step_impl(self, batch, batch_idx, is_val=True)

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate")
    parser.add_argument('--head-lr', type=float, default=None, help="Override head learning rate")
    parser.add_argument('--name', type=str, default=None, help="Override run name for logging")
    parser.add_argument('--save-dir', type=str, default=None, help="Override train_params.save_dir")
    parser.add_argument('--dirpath', type=str, default=None, help="Override checkpoint.dirpath")
    parser.add_argument('--target-tsv-base-path', type=str, default=None, help="Override target_tsv_base_path")
    parser.add_argument('--resume-ckpt', type=str, default=None, help="Path to checkpoint to resume training from (e.g. last.ckpt)")
    parser.add_argument('--resume-weights-only', action='store_true', default=False,
                        help="With --resume-ckpt, load only model weights and start fresh optimizer/scheduler/epoch state")
    parser.add_argument('--wandb-id', type=str, default=None, help="W&B run ID to resume logging into (find in the run URL or wandb dir)")
    parser.add_argument('--max-shards', type=int, default=None, help="Limit shards per epoch (micro-epoch for HPO). None = use all shards.")
    parser.add_argument('--freeze-trunk', action='store_true', default=False, help="Freeze everything except the head (head-only warmup for HPO)")
    args = parser.parse_args()

    if args.resume_weights_only and not args.resume_ckpt:
        parser.error("--resume-weights-only requires --resume-ckpt")

    cfg_dict = load_config(args.config)

    # Apply CLI overrides before creating the namespace
    if args.target_tsv_base_path: cfg_dict["target_tsv_base_path"] = args.target_tsv_base_path
    if args.name: cfg_dict["name"] = args.name
    if args.lr is not None:
        cfg_dict.setdefault("train_params", {})["lr"] = args.lr
    if args.head_lr is not None:
        cfg_dict.setdefault("train_params", {})["head_lr"] = args.head_lr
    if args.save_dir:
        cfg_dict.setdefault("train_params", {})["save_dir"] = args.save_dir
    if args.dirpath:
        cfg_dict.setdefault("train_params", {})
        if "checkpoint" not in cfg_dict["train_params"] or not isinstance(cfg_dict["train_params"]["checkpoint"], dict):
             cfg_dict["train_params"]["checkpoint"] = {}
        cfg_dict["train_params"]["checkpoint"]["dirpath"] = args.dirpath

    config = SimpleNamespace(**cfg_dict)
    # Copy mutable config blocks so we can safely override defaults
    model_params = dict(getattr(config, "model_params", {}))
    train_params = dict(getattr(config, "train_params", {}))

    # Setting Default Early Stopping / LR monitor to Pearson Max!
    if "checkpoint" not in train_params:
        train_params["checkpoint"] = {}
    train_params["checkpoint"].setdefault("monitor", "val_pearson")
    train_params["checkpoint"].setdefault("mode", "max")

    # Task formatting
    if isinstance(train_params.get("task"), str) and train_params["task"].lower() == "multilabel":
        train_params["task"] = "binary"
    else:
        train_params["task"] = train_params.get("task", "binary").lower()

    train_params["logger"] = None # Disable auto-wandb attach
    
    # Data Loading
    n_tasks = len(config.cell_types)
    crop_len_bins = model_params.get("crop_len", 0)

    # Create explicit cell type mapping (from YAML config, not sorted!)
    celltype_to_task_idx = {ct: i for i, ct in enumerate(config.cell_types)}
    idx_to_celltype = {i: ct for ct, i in celltype_to_task_idx.items()}
    
    # cell type mapping verification logging (important sanity check for correct task ordering and conditional head logic)
    logging.info(f"\n" + "="*70)
    logging.info(f"CELL TYPE MAPPING (FROM YAML CONFIG):")
    for ct in config.cell_types:
        logging.info(f"  Index {celltype_to_task_idx[ct]:d}: {ct}")
    logging.info(f"="*70 + "\n")

    # Convert LR to float if it came in as string
    if "lr" in train_params and isinstance(train_params["lr"], str):
        try: train_params["lr"] = float(train_params["lr"])
        except ValueError: pass # Let it fail later if invalid

    # DDP can see unused params with LoRA/custom heads; enable find_unused_parameters for multi-GPU
    devices = train_params.get("devices")
    is_multi_gpu = isinstance(devices, (list, tuple)) and len(devices) > 1
    if is_multi_gpu and "strategy" not in train_params:
        train_params["strategy"] = "ddp_find_unused_parameters_true"
    
    # Loss Params
    load_mask, load_conf, unify_mask = True, True, True
    if "loss_params" in train_params:
        lp = train_params["loss_params"]
        load_mask = lp.get("use_mask", lp.get("psi_params", {}).get("use_mask", True))
        load_conf = lp.get("use_confidence", lp.get("psi_params", {}).get("use_confidence", True))
        unify_mask = lp.get("unify_mask", lp.get("psi_params", {}).get("unify_mask", True))

    train_dataset = ShardedNPZDataset(
        config.target_tsv_base_path, "train", shuffle_shards=True, n_tasks=None,  # Load all tasks first
        crop_len_bins=crop_len_bins, load_mask=load_mask, load_confidence=load_conf, unify_mask=unify_mask
    )
    val_dataset = ShardedNPZDataset(
        config.target_tsv_base_path, "val", shuffle_shards=False, n_tasks=None,  # Load all tasks first
        crop_len_bins=crop_len_bins, load_mask=load_mask, load_confidence=load_conf, unify_mask=unify_mask
    )
    
    # If config specifies fewer cell types than available in data, filter to relevant tasks
    actual_n_tasks = train_dataset.n_tasks
    if n_tasks < actual_n_tasks:
        # Map config cell types to data task indices
        celltype_alphabetical = ["B_Cell", "Macrophage", "T_Cell"]
        task_indices = []
        
        logging.info(f"\n" + "="*70)
        logging.info(f"TASK FILTERING: Data has {actual_n_tasks} tasks, config specifies {n_tasks}")
        for ct in config.cell_types:
            if ct in celltype_alphabetical:
                idx = celltype_alphabetical.index(ct)
                task_indices.append(idx)
                logging.info(f"  Config cell type '{ct}' -> Data task index {idx}")
            else:
                logging.warning(f"  Warning: Config cell type '{ct}' not found in alphabetical mapping")
        logging.info(f"Filtering targets to task indices: {task_indices}")
        logging.info("="*70 + "\n")
        
        # # Apply task filtering
        # from sharded_dataset import TaskFilterWrapper
        # train_dataset = TaskFilterWrapper(train_dataset, task_indices)
        # val_dataset = TaskFilterWrapper(val_dataset, task_indices)
        # n_tasks = len(task_indices)  # Update n_tasks to filtered count
    
    # HPO: Subsample shards for micro-epochs
    if args.max_shards is not None:
        logging.info(f"HPO mode: Limiting to {args.max_shards} shards per epoch (micro-epoch)")
        train_dataset = SubsampledShardedDataset(train_dataset, max_shards=args.max_shards, seed=42)
        val_dataset = SubsampledShardedDataset(val_dataset, max_shards=max(1, args.max_shards // 5), seed=43)

    # Conditional Wrapper Logic
    if model_params.get("conditional_head", False):
        logging.info("Wrapping datasets for Conditional Training")
        train_dataset = ConditionalCellTypeWrapper(train_dataset, mode="random", seed=42)
        
        # CHANGED: Ensure Validation is deterministic so val_loss is comparable between epochs
        val_dataset = ConditionalCellTypeWrapper(val_dataset, mode="expand", seed=43)
        
        # Bind the __len__ method to the wrapper class instances
        train_dataset.__len__ = MethodType(conditional_wrapper_len, train_dataset)
        val_dataset.__len__ = MethodType(conditional_wrapper_len, val_dataset)
        
        # Inject explicit n_celltypes because we're about to mask n_tasks=1 for the model head
        model_params["n_celltypes"] = n_tasks
        model_params["idx_to_celltype"] = idx_to_celltype
        model_params["celltype_to_task_idx"] = celltype_to_task_idx
        n_tasks_to_model = 1 # Conditional model outputs 1 track
    else:
        model_params["idx_to_celltype"] = idx_to_celltype
        model_params["celltype_to_task_idx"] = celltype_to_task_idx
        n_tasks_to_model = n_tasks
    
    # Model Init
    logging.info(f"Loading model: {model_params.get('model_type', 'Borzoi')}")
    model = grelu.resources.load_model(project="borzoi", model_name="mouse_rep0")
    model.model_params.update(model_params)
    model.train_params.update(train_params)
    
    # model.initialize_loss()  # Removed to avoid TypeError from default LightningModel logic
    model.initialize_activation()
    model.initialize_metrics()

    # Assign custom loss from local losses.py if specified
    custom_loss_name = train_params.get("loss", "bce").lower()
    if custom_loss_name == "strongpsiloss":
        from losses import StrongPSILoss
        loss_params = train_params.get("loss_params", {})
        flat_params = {}
        # Flatten psi_params
        if "psi_params" in loss_params:
            for k, v in loss_params["psi_params"].items():
                if k == "alpha_shape":
                    flat_params["beta_profile"] = v
                else:
                    flat_params[k] = v
        # Flatten strongpsiloss_params
        if "strongpsiloss_params" in loss_params:
            for k, v in loss_params["strongpsiloss_params"].items():
                flat_params[k] = v
        # Add any other top-level keys (except psi_params and strongpsiloss_params)
        for k, v in loss_params.items():
            if k not in ("psi_params", "strongpsiloss_params"):
                flat_params[k] = v
        model.loss = StrongPSILoss(**flat_params)
    elif custom_loss_name == "celltypebhatt":
        from losses import CellTypeBhattacharyyaLoss
        loss_params = train_params.get("loss_params", {})
        if "psi_params" in loss_params:
            psi_params = loss_params.pop("psi_params")
            for k, v in psi_params.items():
                loss_params[k] = v
        model.loss = CellTypeBhattacharyyaLoss(**loss_params)

    model.training_step = MethodType(custom_training_step, model)
    model.validation_step = MethodType(custom_validation_step, model)

    # Apply Custom Loss Class
    if "loss_params" in train_params:
        lp = train_params["loss_params"]
        if "focal_loss" in lp:
            fl = lp["focal_loss"]
            model.loss = FocalLoss(alpha=float(fl.get('alpha', 0.25)) if fl.get('alpha') != 'None' else None, gamma=float(fl.get('gamma', 2.0)))
        elif "psi_params" in lp:
            pp = lp["psi_params"]
            if "alpha_cell" in pp or "beta_profile" in pp:
                model.loss = StrongPSILoss(
                    lambda_mid=float(pp.get('lambda_mid', 25.0)),
                    alpha_cell=float(pp.get('alpha_cell', 0.05)),
                    beta_profile=float(pp.get('beta_profile', 0.02)),
                    gamma_pointwise=float(pp.get('gamma_pointwise', 1.0)),
                    eps=float(pp.get('eps', 1e-6)),
                    use_mask=bool(pp.get('use_mask', True)),
                    pointwise_mode=pp.get('pointwise_mode', 'logit')
                )
                logging.info(f"Using StrongPSILoss (pointwise_mode={model.loss.pointwise_mode})")
            elif "alpha_shape" in pp:
                model.loss = StrongPSILoss(
                    lambda_mid=float(pp.get('lambda_mid', 25.0)),
                    alpha_cell=0.0,
                    beta_profile=float(pp['alpha_shape']),
                    gamma_pointwise=float(pp.get('gamma_pointwise', 1.0)),
                    eps=float(pp.get('eps', 1e-6)),
                    use_mask=bool(pp.get('use_mask', True)),
                    pointwise_mode=pp.get('pointwise_mode', 'logit')
                )
                logging.info(f"Using StrongPSILoss via fallback (pointwise_mode={model.loss.pointwise_mode})")
            else:
                model.loss = PSILoss(lambda_mid=float(pp.get('lambda_mid', 25.0)), eps=float(pp.get('eps', 1e-6)), use_mask=bool(pp.get('use_mask', True)))

    # Apply Head Architecture (Using Centralized Function)
    apply_custom_head(model, n_tasks_to_model, model_params)

    # Apply LoRA or Custom Optimizer
    if model_params.get("use_lora", False):
        logging.info("Attaching LoRA methods...")
        model = attach_lora_methods(model)
        
    elif "head_lr" in train_params:
        logging.info("Attaching custom optimizer for differential Learning Rates...")
        model.configure_optimizers = MethodType(configure_optimizers_lora, model)
        model._is_lora_tuning = True

    # Freezing logic
    freeze_kwargs = {}
    if model_params.get("freeze_embedding", False):
        for param in model.model.embedding.parameters():
            param.requires_grad = False
        freeze_kwargs["freeze_embedding"] = True

    if model_params.get("layers_to_freeze"):
        freeze_kwargs["layers_to_freeze"] = model_params["layers_to_freeze"]

    # HPO: Head-only warmup (freeze everything except the head)
    freeze_trunk_cfg = bool(train_params.get("freeze_trunk", False))
    if args.freeze_trunk or freeze_trunk_cfg:
        logging.info("Head-only warmup — freezing trunk, only head is trainable")
        freeze_trunk(model)

    # Logger
    wandb_name = config.name if hasattr(config, 'name') else "borzoi_finetune"
    wandb_kwargs = dict(project=config.project, name=wandb_name, config=cfg_dict, log_model=False)
    if args.wandb_id:
        wandb_kwargs["id"] = args.wandb_id
        wandb_kwargs["resume"] = "must"  # Fail loudly if the run ID doesn't exist
        logging.info(f"Resuming W&B run: {args.wandb_id}")
    logger = WandbLogger(**wandb_kwargs)
    model.train_params["logger"] = logger

    # Force Early Stopping if defined in config
    if "early_stopping_patience" in train_params:
        ckpt_cfg = train_params.get("checkpoint", {})
        es_monitor = ckpt_cfg.get("monitor", "val_pearson")
        es_mode = ckpt_cfg.get("mode", "max")
        logging.info(f"Adding EarlyStopping callback: monitor={es_monitor}, mode={es_mode}, patience={train_params['early_stopping_patience']}")
        early_stop_callback = EarlyStopping(
            monitor=es_monitor,
            patience=train_params["early_stopping_patience"],
            mode=es_mode
        )
        if "callbacks" not in model.train_params:
            model.train_params["callbacks"] = []
        model.train_params["callbacks"].append(early_stop_callback)

    # ---------------------------------
    # Add gReLU Diagnostics Callback
    # ---------------------------------
    try:
        from grelu_diagnostics_callback import GreluSplicingDiagnostics
        save_dir_base = train_params.get("save_dir", ".")
        diag_out_dir = os.path.join(save_dir_base, wandb_name + "_diagnostics")
        logging.info(f"Adding GreluSplicingDiagnostics callback: saving to {diag_out_dir}")
        diag_callback = GreluSplicingDiagnostics(
            out_dir=diag_out_dir,
            every_n_epochs=1,
            max_save=64,
            trunk_prefix="embedding",
            head_prefix="head"
        )
        if "callbacks" not in model.train_params:
            model.train_params["callbacks"] = []
        model.train_params["callbacks"].append(diag_callback)
    except ImportError as e:
        logging.warning(f"Could not import GreluSplicingDiagnostics: {e}")

    # Verify cell type mapping consistency
    logging.info("\n" + "="*70)
    logging.info("CELL TYPE MAPPING VERIFICATION:")
    logging.info(f"  YAML config cell_types: {config.cell_types}")
    if hasattr(train_dataset, 'celltype_mapping') and train_dataset.celltype_mapping:
        logging.info(f"  NPZ file mapping: {train_dataset.celltype_mapping}")
        npz_mapping_matches = all(
            train_dataset.celltype_mapping.get(idx) == config.cell_types[idx]
            for idx in range(len(config.cell_types))
        )
        if npz_mapping_matches:
            logging.info("MAPPING VERIFIED: NPZ matches YAML config")
        else:
            logging.warning("MAPPING MISMATCH: NPZ does not match YAML config!")
    else:
        logging.info("NPZ does not contain explicit cell type mapping (expected for older datasets)")
    logging.info("="*70 + "\n")

    model.train() 
    eval_modules = []
    for name, m in model.named_modules():
        if not m.training:
            eval_modules.append(name)
    logging.info(f"Modules in eval mode (after model.train()): {eval_modules}")
    logging.info("Starting Training...")

    # Weight-only initialization mimics transfer from a prior stage while keeping
    # optimizer/scheduler/epoch counters fresh for the new stage.
    trainer_checkpoint_path = args.resume_ckpt
    if args.resume_ckpt and args.resume_weights_only:
        load_weights_only_from_checkpoint(model, args.resume_ckpt)
        trainer_checkpoint_path = None
    
    # --- PYTORCH LIGHTNING TRAINER MONKEYPATCH ---
    # grelu completely ignores our custom callbacks when it instantiates pl.Trainer.
    # We monkeypatch the init to manually inject everything stored in model.train_params["callbacks"].
    import pytorch_lightning as pl
    orig_trainer_init = pl.Trainer.__init__
    
    def patched_trainer_init(self, *args, **kwargs):
        if "callbacks" in model.train_params and len(model.train_params["callbacks"]) > 0:
            # Merge grelu's callbacks (passed in kwargs) with ours
            existing_callbacks = kwargs.get("callbacks", [])
            if not isinstance(existing_callbacks, list):
                existing_callbacks = [existing_callbacks]
            
            # Combine them, making sure not to duplicate EarlyStopping if grelu already added one
            final_callbacks = list(existing_callbacks)
            for custom_cb in model.train_params["callbacks"]:
                # Simple deduplication by class name
                if not any(type(c).__name__ == type(custom_cb).__name__ for c in final_callbacks):
                    final_callbacks.append(custom_cb)
            
            kwargs["callbacks"] = final_callbacks
            logging.info(f"Monkeypatched Trainer! Injected {len(final_callbacks)} callbacks.")
        orig_trainer_init(self, *args, **kwargs)
        
    pl.Trainer.__init__ = patched_trainer_init
    # ---------------------------------------------
    
    if model_params.get("use_lora", False):
        # LoRA path: tune_on_dataset_lora correctly handles apply_custom_head + wrap_with_lora
        trainer = model.tune_on_dataset(train_dataset, val_dataset, checkpoint_path=trainer_checkpoint_path, **freeze_kwargs)
    else:
        # Non-LoRA path: bypass grelu's tune_on_dataset which calls change_head and
        # overwrites our custom head (SplitHead/NonlinearHead) with a generic ConvHead.
        # apply_custom_head was already called above, so do the housekeeping that
        # tune_on_dataset normally performs and go straight to train_on_dataset.
        model.base_data_params = model.data_params.copy()
        model.data_params = {}

        # Handle layers_to_freeze (normally done inside tune_on_dataset)
        layers_to_freeze = freeze_kwargs.get("layers_to_freeze", [])
        if layers_to_freeze:
            logging.info(f"Freezing layers matching: {layers_to_freeze}")
            for name, param in model.model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    if "lora_" in name or "modules_to_save" in name:
                        continue
                    param.requires_grad = False

        trainer = model.train_on_dataset(train_dataset, val_dataset, checkpoint_path=trainer_checkpoint_path)

if __name__ == "__main__":
    main()