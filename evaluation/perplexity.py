import torch
import torch.nn.functional as F


def compute_perplexity(model, data_loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["input_ids"].to(
                device
            )  # Use input_ids as labels for causal LM
            outputs = model(input_ids)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens))
