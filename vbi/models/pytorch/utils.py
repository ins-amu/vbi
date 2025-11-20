from typing import Optional, Union
import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for this module. Please install it to proceed.")


def check_and_expand_param(param: Union[float, np.ndarray, torch.Tensor],
                           name: str,
                           n_nodes: int,
                           n_sets: int,
                           device: str = 'cuda',
                           param_type: Optional[str] = None) -> torch.Tensor:
    """
    Check parameter and expand to correct shape [n_nodes x n_sets]
    
    This function safely handles parameter expansion by avoiding ambiguity when
    n_nodes == n_sets. For 1D arrays with ambiguous length, param_type must be specified.
    
    Args:
        param: Parameter value (scalar, array, or tensor)
        name: Parameter name (for error messages)
        n_nodes: Number of nodes
        n_sets: Number of parameter sets
        device: Target device
        param_type: For 1D arrays when len == n_nodes == n_sets, specify:
                   'nodes' for node-specific parameter (shape [n_nodes, 1])
                   'sets' for set-specific parameter (shape [1, n_sets])
        
    Returns:
        torch.Tensor of shape [n_nodes, n_sets] or [1, 1], [1, n_sets], [n_nodes, 1]
        
    Raises:
        ValueError: If parameter shape is incompatible or ambiguous
    """
    # Convert to tensor if needed
    if isinstance(param, (int, float)):
        # Scalar - keep as [1, 1]
        return torch.tensor([[param]], device=device, dtype=torch.float32)
    
    if isinstance(param, np.ndarray):
        param = torch.from_numpy(param).float().to(device)
    elif isinstance(param, torch.Tensor):
        param = param.float().to(device)
    else:
        raise ValueError(f"{name}: must be scalar, numpy array, or torch tensor")
    
    # Check dimensionality
    if param.ndim == 0:
        # Scalar tensor
        return param.view(1, 1)
    elif param.ndim == 1:
        # 1D array - handle carefully to avoid ambiguity
        param_len = len(param)
        
        if param_len == 1:
            # Single value
            return param.view(1, 1)
        elif param_len == n_nodes and param_len == n_sets:
            # Ambiguous case: length matches both n_nodes and n_sets
            if param_type is None:
                raise ValueError(
                    f"{name}: 1D array length ({param_len}) matches both "
                    f"n_nodes ({n_nodes}) and n_sets ({n_sets}). "
                    f"Please specify param_type='nodes' or param_type='sets' "
                    f"to resolve ambiguity."
                )
            elif param_type == 'nodes':
                return param.view(-1, 1)  # [n_nodes, 1]
            elif param_type == 'sets':
                return param.view(1, -1)  # [1, n_sets]
            else:
                raise ValueError(
                    f"{name}: param_type must be 'nodes' or 'sets', got '{param_type}'"
                )
        elif param_len == n_nodes:
            # Node-specific, single set
            return param.view(-1, 1)
        elif param_len == n_sets:
            # Set-specific, broadcast to all nodes
            return param.view(1, -1)
        else:
            raise ValueError(
                f"{name}: 1D array length ({param_len}) must match "
                f"n_nodes ({n_nodes}), n_sets ({n_sets}), or be 1. "
                f"Got length {param_len}."
            )
    elif param.ndim == 2:
        # 2D array - check shape compatibility
        if param.shape == (n_nodes, n_sets):
            return param
        elif param.shape == (1, 1):
            return param
        elif param.shape == (1, n_sets):
            return param
        elif param.shape == (n_nodes, 1):
            return param
        else:
            raise ValueError(
                f"{name}: 2D shape {param.shape} incompatible with "
                f"required dimensions [n_nodes={n_nodes}, n_sets={n_sets}]. "
                f"Allowed shapes: [{n_nodes}, {n_sets}], [1, 1], "
                f"[1, {n_sets}], or [{n_nodes}, 1]"
            )
    else:
        raise ValueError(f"{name}: must be 0D, 1D, or 2D, got {param.ndim}D")
