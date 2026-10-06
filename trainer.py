"""Training and evaluation share one batch contract and metric implementation."""
import time
from contextlib import nullcontext

import numpy as np
import torch

from metrics import classification_metrics


def run_epoch(model, loader, criterion, device, classes, optimizer=None, scaler=None, amp_enabled=False, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    labels_all, probabilities_all, rows = [], [], []
    total_loss, count = 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with nullcontext() if training else torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            inputs = {key: batch[key].to(device, non_blocking=True) for key in ("image", "tokens") if key in batch}
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, aux = model(images=inputs.get("image"), tokens=inputs.get("tokens"))
                loss = criterion(logits, labels, aux)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check inputs, AMP settings, and learning rates")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            probabilities = logits.detach().float().softmax(dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            labels_all.extend(true.tolist())
            probabilities_all.extend(probabilities.tolist())
            total_loss += loss.item() * len(labels)
            count += len(labels)
            if not training:
                selected = aux["selected_indices"].detach().cpu().tolist()
                for index, (actual, probs) in enumerate(zip(true, probabilities)):
                    prediction = int(probs.argmax())
                    rows.append({"sample_id": batch["sample_id"][index], "path": batch["path"][index],
                                 "true_index": int(actual), "pred_index": prediction, "true_class": classes[int(actual)],
                                 "pred_class": classes[prediction], "confidence": float(probs[prediction]),
                                 "correct": bool(prediction == actual), "probabilities": probs.tolist(),
                                 "selected_indices": selected[index]})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = classification_metrics(labels_all, probabilities_all, classes, total_loss / max(count, 1))
    metrics["seconds"] = time.perf_counter() - started
    metrics["samples_per_second"] = count / max(metrics["seconds"], 1e-9)
    return metrics, rows
