import torch
import torch.nn.functional as F
from pathlib import Path
from pytorch_lightning.callbacks import Callback

# ----------------------------
# Core metrics
# ----------------------------

def grad_norm(named_params, prefix=None):
    total = 0.0
    for name, p in named_params:
        # Changed to 'in' to handle PEFT wrappers cleanly
        if prefix and prefix not in name:
            continue
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def collapse_score(probs, targets, mask=None, eps=1e-8):
    pred_var = probs.var(dim=1, unbiased=False)
    targ_var = targets.var(dim=1, unbiased=False)

    if mask is not None:
        valid = mask.any(dim=1).float()
        pred_var *= valid
        targ_var *= valid

    num = (targ_var - pred_var).sum(dim=-1)
    den = targ_var.sum(dim=-1).clamp_min(eps)
    return num / den  # [B]


def com_error(probs, targets, mask=None, eps=1e-8):
    if mask is not None:
        probs = probs * mask
        targets = targets * mask

    p = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = targets / targets.sum(dim=-1, keepdim=True).clamp_min(eps)

    pos = torch.arange(probs.shape[-1], device=probs.device).float()
    pos = pos.view(1, 1, -1)

    p_com = (p * pos).sum(dim=-1)
    q_com = (q * pos).sum(dim=-1)

    return (p_com - q_com).abs().mean(dim=1)


# ----------------------------
# Callback
# ----------------------------

class GreluSplicingDiagnostics(Callback):
    def __init__(
        self,
        out_dir="diagnostics",
        every_n_epochs=1,
        max_save=64,
        trunk_prefix="embedding", 
        head_prefix="head",
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.every_n_epochs = every_n_epochs
        self.max_save = max_save
        self.trunk_prefix = trunk_prefix
        self.head_prefix = head_prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self._buffer = []
        self._tracked_grads = False

    def forward_logits(self, pl_module, x, cell_type_id=None):
        if cell_type_id is not None:
            return pl_module((x, cell_type_id), logits=True)
        else:
            return pl_module(x, logits=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._buffer = []
        self._tracked_grads = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.current_epoch % self.every_n_epochs != 0: return
        if trainer.global_rank != 0: return

        # ---------------------------------
        # unpack batch
        # ---------------------------------
        x, targets, mask, confidence, cell_type_id = None, None, None, None, None
        
        if isinstance(batch, dict):
            x = batch["sequence"]
            targets = batch["targets"]
            mask = batch.get("mask", None)
            confidence = batch.get("confidence", None)
        else:
            is_conditional = hasattr(pl_module.model, "head") and type(pl_module.model.head).__name__ == "ConditionalHead"
            if is_conditional and len(batch) >= 2:
                x, targets, cell_type_id = batch[0], batch[1], batch[-1]
                if len(batch) >= 4: mask = batch[2]
                if len(batch) == 5: confidence = batch[3]
            else:
                if len(batch) >= 2: x, targets = batch[0], batch[1]
                if len(batch) >= 3: mask = batch[2]
                if len(batch) == 4: confidence = batch[3]
                    
        if isinstance(mask, list): mask = None
        if isinstance(confidence, list): confidence = None
        
        if targets.ndim == 2: targets = targets.unsqueeze(0)
        if mask is not None and mask.ndim == 2: mask = mask.unsqueeze(0)
        if confidence is not None and confidence.ndim == 2: confidence = confidence.unsqueeze(0)
            
        # Skip empty regions
        if targets.sum() <= 0: return
        
        # ---------------------------------
        # gradient computation (once per epoch on first active batch)
        # ---------------------------------
        if not self._tracked_grads:
            with torch.enable_grad():
                pl_module.eval()
                logits_grad = self.forward_logits(pl_module, x, cell_type_id)
                loss_out = pl_module.loss(logits_grad, targets, mask=mask, confidence=confidence)
                
                if isinstance(loss_out, tuple) and len(loss_out) == 4:
                    total, pointwise, cell, profile = loss_out
                else:
                    total = loss_out
                    pointwise, cell, profile = loss_out, torch.tensor(0.0, device=logits_grad.device), torch.tensor(0.0, device=logits_grad.device)

                alpha = getattr(pl_module.loss, "alpha_cell", 0.0)
                beta = getattr(pl_module.loss, "beta_profile", 0.0)
                named_params = [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]

                def measure(loss, retain):
                    if not isinstance(loss, torch.Tensor) or loss.requires_grad == False:
                        return {"all": 0, "trunk": 0, "head": 0}
                    pl_module.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=retain)
                    return {
                        "all": grad_norm(named_params),
                        "trunk": grad_norm(named_params, self.trunk_prefix),
                        "head": grad_norm(named_params, self.head_prefix),
                    }

                grad_point = measure(pointwise, retain=True)
                grad_cell = measure(alpha * cell, retain=True) if alpha > 0 else {"all": 0, "trunk": 0, "head": 0}
                grad_prof = measure(beta * profile, retain=False) if beta > 0 else {"all": 0, "trunk": 0, "head": 0}
                pl_module.zero_grad(set_to_none=True)

            self._tracked_grads = True
            
            self._epoch_losses = {
                "total": total.detach().cpu().item() if isinstance(total, torch.Tensor) else total,
                "pointwise": pointwise.detach().cpu().item() if isinstance(pointwise, torch.Tensor) else pointwise,
                "cell": cell.detach().cpu().item() if isinstance(cell, torch.Tensor) else cell,
                "profile": profile.detach().cpu().item() if isinstance(profile, torch.Tensor) else profile,
            }
            self._epoch_grads = {
                "pointwise": grad_point,
                "cell": grad_cell,
                "profile": grad_prof,
            }
            
            # Log metrics
            pl_module.log_dict({
                "diag/loss_pointwise": pointwise,
                "diag/loss_cell": cell,
                "diag/loss_profile": profile,
                "diag/grad_pointwise": grad_point["all"],
                "diag/grad_cell": grad_cell["all"],
                "diag/grad_profile": grad_prof["all"],
                "diag/grad_pointwise_head": grad_point["head"],
                "diag/grad_cell_head": grad_cell.get("head", 0),
                "diag/grad_profile_head": grad_prof.get("head", 0),
            }, prog_bar=False, logger=True)

        # ---------------------------------
        # sequence buffering for plotting
        # ---------------------------------
        if len(self._buffer) < self.max_save * 3: # Buffer margin
            with torch.no_grad():
                pl_module.eval()
                logits = self.forward_logits(pl_module, x, cell_type_id)
                probs = torch.sigmoid(logits)
                collapse = collapse_score(probs, targets, mask)
                com = com_error(probs, targets, mask)
                
                # Global metric logging
                pl_module.log("diag/collapse", collapse.mean(), prog_bar=False, logger=True)
                pl_module.log("diag/com_error", com.mean(), prog_bar=False, logger=True)
                pl_module.log("diag/collapse_ratio", (probs.var(dim=1).mean() / targets.var(dim=1).mean().clamp_min(1e-8)), prog_bar=False, logger=True)

                self._buffer.append({
                    "probs": probs.cpu(),
                    "targets": targets.cpu(),
                    "mask": mask.cpu() if mask is not None else None,
                    "collapse": collapse.cpu(),
                    "com": com.cpu()
                })

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or trainer.current_epoch % self.every_n_epochs != 0: return
        if not self._buffer: return

        probs = torch.cat([b["probs"] for b in self._buffer], dim=0)
        targets = torch.cat([b["targets"] for b in self._buffer], dim=0)
        mask = torch.cat([b["mask"] for b in self._buffer], dim=0) if self._buffer[0]["mask"] is not None else None
        collapse = torch.cat([b["collapse"] for b in self._buffer], dim=0)
        com = torch.cat([b["com"] for b in self._buffer], dim=0)

        n = min(self.max_save, probs.shape[0])
        _, idx = torch.sort(collapse, descending=True)
        idx = idx[:n]

        torch.save({
            "epoch": trainer.current_epoch,
            "probs": probs[idx],
            "targets": targets[idx],
            "mask": mask[idx] if mask is not None else None,
            "collapse": collapse[idx],
            "com": com[idx],
            "losses": getattr(self, "_epoch_losses", {}),
            "grad_norms": getattr(self, "_epoch_grads", {}),
        }, self.out_dir / f"epoch_{trainer.current_epoch:04d}.pt")
