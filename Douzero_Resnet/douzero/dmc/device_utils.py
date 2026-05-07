"""
Utility module for unified CPU/GPU/TPU device handling.
Provides helpers to detect available hardware and create torch devices.
"""
import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


def is_tpu_available():
    """Check if TPU (via torch_xla) is available."""
    if not TPU_AVAILABLE:
        return False
    try:
        devices = xm.get_xla_supported_devices()
        return devices is not None and len(devices) > 0
    except Exception:
        return False


def resolve_training_device(training_device):
    """
    Resolve the training_device argument to a unified string format.
    Supports:
        - 'cpu' -> 'cpu'
        - 'tpu' / 'xla' -> 'tpu'
        - '0', '1', ... -> GPU index string (passed through)
    Returns one of: 'cpu', 'tpu', or a GPU index string.
    """
    if training_device is None:
        if is_tpu_available():
            return 'tpu'
        elif torch.cuda.is_available():
            return '0'
        else:
            return 'cpu'

    td = str(training_device).lower().strip()
    if td == 'cpu':
        return 'cpu'
    if td in ('tpu', 'xla'):
        if not is_tpu_available():
            raise RuntimeError(
                "TPU not available. Please install torch_xla and run on a TPU-enabled environment, "
                "or use CPU/GPU for training."
            )
        return 'tpu'

    # Treat as GPU index
    return str(training_device)


def get_torch_device(device_spec):
    """
    Return a torch.device (or XLA device) from a specification string.
    device_spec: 'cpu', 'tpu', or GPU index like '0', '1'.
    """
    if device_spec == 'cpu':
        return torch.device('cpu')
    elif device_spec == 'tpu':
        if not TPU_AVAILABLE:
            raise RuntimeError("torch_xla is not installed. Cannot use TPU.")
        return xm.xla_device()
    else:
        return torch.device(f'cuda:{device_spec}')


def get_map_location(device_spec):
    """
    Return a map_location string/object suitable for torch.load.
    For TPU we load to CPU first, then move to TPU.
    """
    if device_spec == 'tpu':
        return 'cpu'
    elif device_spec == 'cpu':
        return 'cpu'
    else:
        return f'cuda:{device_spec}'


def save_checkpoint(state_dict, path):
    """
    Save a checkpoint, handling TPU tensors by moving them to CPU first.
    """
    if TPU_AVAILABLE:
        xm.save(state_dict, path)
    else:
        torch.save(state_dict, path)
