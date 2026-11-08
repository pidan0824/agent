"""Model clients, provider adapters, request construction and chat templates."""


def resolve_device(preference="auto"):
    """Select a local Torch device, honoring an explicit override."""
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Expected auto, CPU, CUDA or MPS")
    return device
