import time

import numpy as np
import torch
import torch.nn.functional as F


class SegMeter:
    """Accumulates per-class overlap counts per volume and per slice."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.volumes = {}  # (case, week) -> array (C, 3): intersection, pred, target
        self.slices = []   # one row per slice: case, week, slice, then (C, 3) counts

    def update(self, preds, targets, cases, weeks, slice_ids):
        """preds, targets: (B, H, W) integer class maps."""
        p = F.one_hot(preds, self.num_classes).flatten(1, 2)    # (B, HW, C)
        t = F.one_hot(targets, self.num_classes).flatten(1, 2)
        counts = torch.stack([(p * t).sum(1), p.sum(1), t.sum(1)], dim=2)  # (B, C, 3)
        counts = counts.cpu().numpy().astype(np.int64)
        for i in range(counts.shape[0]):
            key = (int(cases[i]), int(weeks[i]))
            if key not in self.volumes:
                self.volumes[key] = np.zeros((self.num_classes, 3), dtype=np.int64)
            self.volumes[key] += counts[i]
            self.slices.append((key[0], key[1], int(slice_ids[i]), counts[i]))

    @staticmethod
    def _dice_iou(counts):
        inter, pred, target = counts[..., 0], counts[..., 1], counts[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = 2.0 * inter / (pred + target)       # NaN if class absent from both
            iou = inter / (pred + target - inter)
        return dice, iou

    def per_volume(self):
        """Return (keys, dice, iou); dice and iou have shape (n_volumes, C)."""
        keys = sorted(self.volumes)
        counts = np.stack([self.volumes[k] for k in keys])
        dice, iou = self._dice_iou(counts)
        return keys, dice, iou

    def per_slice(self, class_idx):
        """Return a list of dicts with the Dice of one class on every slice.
        'index' is the position in the (unshuffled) dataset."""
        rows = []
        for index, (case, week, slice_id, counts) in enumerate(self.slices):
            dice, _ = self._dice_iou(counts[class_idx])
            rows.append({"index": index, "case": case, "week": week, "slice": slice_id,
                         "dice": float(dice), "target_px": int(counts[class_idx, 2]),
                         "pred_px": int(counts[class_idx, 1])})
        return rows

    def mean_dice(self):
        """Mean Dice per class over volumes, shape (C,)."""
        return np.nanmean(self.per_volume()[1], axis=0)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None):
    """Run the model over a loader.

    Returns (meter, mean_loss, forward_seconds). forward_seconds is the time
    spent in the model only, for the per-case inference time."""
    model.eval()
    meter = SegMeter(num_classes)
    use_cuda = device.type == "cuda"
    total_loss, n_batches, forward_time = 0.0, 0, 0.0
    for images, masks, cases, weeks, slice_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            logits = model(images)
        if use_cuda:
            torch.cuda.synchronize()
        forward_time += time.perf_counter() - start
        if criterion is not None:
            total_loss += criterion(logits, masks).item()
        meter.update(logits.argmax(dim=1), masks, cases, weeks, slice_ids)
        n_batches += 1
    return meter, total_loss / max(n_batches, 1), forward_time

def format_results(meter, class_names):
    """Per-class Dice and IoU (mean +/- std over volumes) as printable text."""
    keys, dice, iou = meter.per_volume()
    lines = [f"Volumes evaluated: {len(keys)}",
            f"{'class':12s} {'DSC mean':>9s} {'DSC std':>8s} {'DSC min':>8s} {'IoU mean':>9s}"]
    for c, name in enumerate(class_names):
        lines.append(f"{name:12s} {np.nanmean(dice[:, c]):9.4f} {np.nanstd(dice[:, c]):8.4f} "
                    f"{np.nanmin(dice[:, c]):8.4f} {np.nanmean(iou[:, c]):9.4f}")
    lines.append(f"{'mean':12s} {np.nanmean(dice):9.4f} {'':8s} {'':8s} {np.nanmean(iou):9.4f}")
    return "\n".join(lines)
