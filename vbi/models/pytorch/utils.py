from typing import Optional, Union
import numpy as np
import torch
import math
import time

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


def torch_corr(A: torch.Tensor) -> torch.Tensor:
    """
    Compute correlation matrix for GPU tensors.
    
    Args:
        A: Input tensor of shape [n_features, n_samples]
        
    Returns:
        Correlation matrix of shape [n_features, n_features]
    """
    Amean = torch.mean(A, 1, keepdim=True)
    Ax = A - Amean
    Astd = torch.mean(Ax**2, 1)
    Amm = torch.mm(Ax, torch.transpose(Ax, 0, 1)) / A.shape[1]
    Aout = torch.sqrt(torch.ger(Astd, Astd))
    Acor = Amm / Aout
    return Acor


def torch_corr2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute correlation between two sets of variables.
    
    Args:
        A: First tensor of shape [n_features_A, n_samples]
        B: Second tensor of shape [n_features_B, n_samples]
        
    Returns:
        Correlation matrix of shape [n_features_A, n_features_B]
    """
    Amean = torch.mean(A, 1, keepdim=True)
    Ax = A - Amean
    Astd = torch.mean(Ax**2, 1)
    
    Bmean = torch.mean(B, 1, keepdim=True)
    Bx = B - Bmean
    Bstd = torch.mean(Bx**2, 1)
    
    numerator = torch.mm(Ax, torch.transpose(Bx, 0, 1)) / A.shape[1]
    denominator = torch.sqrt(torch.ger(Astd, Bstd))
    torch_cor = numerator / denominator
    return torch_cor


def torch_arctanh(A: torch.Tensor) -> torch.Tensor:
    """
    Compute arctanh (inverse hyperbolic tangent) for Fisher z-transformation.
    
    Args:
        A: Input tensor
        
    Returns:
        arctanh(A)
    """
    return 0.5 * torch.log((1 + A) / (1 - A))


def fc_correlation_cost(
    bold_empirical: Union[np.ndarray, torch.Tensor],
    bold_simulated: Union[np.ndarray, torch.Tensor],
    n_dup: int,
    device: str = 'cuda',
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate FC correlation cost comparing empirical and simulated BOLD signals.
    
    This function computes functional connectivity (FC) matrices from empirical and
    simulated BOLD signals, then calculates the correlation between them. For simulated
    data, multiple duplicates of each parameter set are averaged. NaN values are handled
    by excluding affected sets from the calculation.
    
    Args:
        bold_empirical: Empirical BOLD signal of shape [n_nodes, n_timesamples]
        bold_simulated: Simulated BOLD signal of shape [n_nodes, n_sets, n_timesamples]
                       where n_sets = num_sets * n_dup (each set repeated n_dup times)
                       Data must be arranged as (matching rww.py repeat_interleave):
                       [set0_dup0, set0_dup1, ..., set0_dup(n_dup-1),
                        set1_dup0, set1_dup1, ..., set1_dup(n_dup-1), ...]
                       where each unique parameter set has n_dup consecutive duplications
        n_dup: Number of duplicates for each parameter set
        device: Device to run computations on ('cuda' or 'cpu')
        verbose: Whether to print timing information
        
    Returns:
        corr_cost: FC correlation cost for each unique parameter set, shape [num_sets,]
                  Values range from 0 (perfect match) to ~2 (worst case).
                  Sets with all NaN values are assigned cost of 10.
                  
    Example:
        >>> bold_emp = np.random.randn(68, 1200)  # 68 nodes, 1200 timepoints
        >>> bold_sim = np.random.randn(68, 500, 1200)  # 500 = 100 sets * 5 duplicates
        >>> # bold_sim arrangement: [set0_dup0, set0_dup1, ..., set0_dup4, 
        >>> #                        set1_dup0, set1_dup1, ..., set1_dup4, ...]
        >>> cost = fc_correlation_cost(bold_emp, bold_sim, n_dup=5)
        >>> cost.shape
        (100,)
    """
    fc_timestart = time.time()
    
    # Convert to torch tensors and move to device
    if isinstance(bold_empirical, np.ndarray):
        bold_empirical = torch.from_numpy(bold_empirical).float().to(device)
    else:
        bold_empirical = bold_empirical.float().to(device)
        
    if isinstance(bold_simulated, np.ndarray):
        bold_simulated = torch.from_numpy(bold_simulated).float().to(device)
    else:
        bold_simulated = bold_simulated.float().to(device)
    
    # Check shapes
    if bold_empirical.ndim != 2:
        raise ValueError(f"bold_empirical must be 2D [n_nodes, n_timesamples], got shape {bold_empirical.shape}")
    if bold_simulated.ndim != 3:
        raise ValueError(f"bold_simulated must be 3D [n_nodes, n_sets, n_timesamples], got shape {bold_simulated.shape}")
    
    n_nodes = bold_empirical.shape[0]
    n_timesamples = bold_empirical.shape[1]
    
    if bold_simulated.shape[0] != n_nodes:
        raise ValueError(f"Node dimension mismatch: empirical has {n_nodes}, simulated has {bold_simulated.shape[0]}")
    if bold_simulated.shape[2] != n_timesamples:
        raise ValueError(f"Time dimension mismatch: empirical has {n_timesamples}, simulated has {bold_simulated.shape[2]}")
    
    n_sets = bold_simulated.shape[1]
    n_num = n_sets // n_dup  # Number of unique parameter sets
    
    if n_sets % n_dup != 0:
        raise ValueError(f"n_sets ({n_sets}) must be divisible by n_dup ({n_dup})")
    
    # Compute empirical FC
    emp_fc = torch_corr(bold_empirical)
    
    # Create upper triangular mask for vectorizing FC matrices
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1) == 1
    vect_len = int(n_nodes * (n_nodes - 1) / 2)
    
    # Vectorize empirical FC
    emp_fc_vector = emp_fc[fc_mask]
    
    # Calculate vectored simulated FC for each set
    sim_fc_vector = torch.zeros(n_sets, vect_len, device=device)
    for i in range(n_sets):
        sim_fc = torch_corr(bold_simulated[:, i, :])
        sim_fc_vector[i, :] = sim_fc[fc_mask]
    
    # Handle NaN values and average the simulated FCs with same parameter set
    # Data arrangement (from rww.py repeat_interleave): 
    #   [set0_dup0, set0_dup1, ..., set0_dup(n_dup-1), 
    #    set1_dup0, set1_dup1, ..., set1_dup(n_dup-1), ...]
    # Set NaN to 0 for summation, but track which entries are valid
    valid_mask = ~torch.isnan(sim_fc_vector)  # [n_sets, vect_len]
    sim_fc_vector_clean = sim_fc_vector.clone()
    sim_fc_vector_clean[~valid_mask] = 0
    
    # Reshape to [n_num, n_dup, vect_len] for easier averaging across duplicates
    sim_fc_reshaped = sim_fc_vector_clean.view(n_num, n_dup, vect_len)
    valid_mask_reshaped = valid_mask.view(n_num, n_dup, vect_len)
    
    # Sum across duplicates (dim=1) and count valid entries
    sim_fc_num = sim_fc_reshaped.sum(dim=1)  # [n_num, vect_len]
    sim_fc_den = valid_mask_reshaped.sum(dim=1).float()  # [n_num, vect_len]
    
    # Avoid division by zero: set denominator to NaN where count is 0
    sim_fc_den[sim_fc_den == 0] = float('nan')
    
    # Compute average, will be NaN where all values were NaN
    sim_fc_ave = sim_fc_num / sim_fc_den  # [n_num, vect_len]
    
    # Prepare empirical FC for correlation computation
    emp_fc_repeated = emp_fc_vector.unsqueeze(0).repeat(n_num, 1)  # [n_num, vect_len]
    
    # Apply Fisher z-transformation
    sim_fc_z = torch_arctanh(sim_fc_ave)
    emp_fc_z = torch_arctanh(emp_fc_repeated)
    
    # Calculate correlation between empirical and simulated FCs
    corr_mass = torch_corr2(sim_fc_z, emp_fc_z)
    corr_cost = torch.diag(corr_mass)
    
    # Convert to numpy
    corr_cost = corr_cost.cpu().numpy()
    
    # Final cost is 1 - correlation (so 0 = perfect, 2 = worst case)
    corr_cost = 1 - corr_cost
    
    # Assign high cost (10) to sets where all duplicates had NaN
    corr_cost[np.isnan(corr_cost)] = 10
    
    if verbose:
        fc_elapsed = time.time() - fc_timestart
        print(f'Time for calculating FC correlation cost: {fc_elapsed:.3f}s')
    
    return corr_cost


def fcd_ks_cost(
    bold_empirical: Union[np.ndarray, torch.Tensor],
    bold_simulated: Union[np.ndarray, torch.Tensor],
    n_dup: int,
    window_size: int = 83,
    n_bins: int = 10000,
    fcd_range: tuple = (-1.0, 1.0),
    device: str = 'cuda',
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate FCD KS statistics cost comparing empirical and simulated BOLD signals.
    
    This function computes functional connectivity dynamics (FCD) matrices using sliding
    windows, then calculates the Kolmogorov-Smirnov statistic to compare the distributions
    of FCD values between empirical and simulated data. NaN values in simulated data are
    handled by excluding affected sets from the calculation.
    
    Args:
        bold_empirical: Empirical BOLD signal of shape [n_nodes, n_timesamples]
        bold_simulated: Simulated BOLD signal of shape [n_nodes, n_sets, n_timesamples]
                       where n_sets = num_sets * n_dup (each set repeated n_dup times)
                       Data must be arranged as (matching rww.py repeat_interleave):
                       [set0_dup0, set0_dup1, ..., set0_dup(n_dup-1),
                        set1_dup0, set1_dup1, ..., set1_dup(n_dup-1), ...]
        n_dup: Number of duplicates for each parameter set
        window_size: Size of sliding window for FCD calculation (default: 83)
        n_bins: Number of bins for FCD histogram (default: 10000)
        fcd_range: Range for FCD histogram as (min, max) tuple (default: (-1.0, 1.0))
        device: Device to run computations on ('cuda' or 'cpu')
        verbose: Whether to print timing information
        
    Returns:
        ks_cost: FCD KS statistics cost for each unique parameter set, shape [num_sets,]
                Values are normalized KS distances. Sets with all NaN values are 
                assigned cost of 10.
                
    Example:
        >>> bold_emp = np.random.randn(68, 1200)  # 68 nodes, 1200 timepoints
        >>> bold_sim = np.random.randn(68, 500, 1200)  # 500 = 100 sets * 5 duplicates
        >>> cost = fcd_ks_cost(bold_emp, bold_sim, n_dup=5)
        >>> cost.shape
        (100,)
    """
    fcd_timestart = time.time()
    
    # Convert to torch tensors and move to device
    if isinstance(bold_empirical, np.ndarray):
        bold_empirical = torch.from_numpy(bold_empirical).float().to(device)
    else:
        bold_empirical = bold_empirical.float().to(device)
        
    if isinstance(bold_simulated, np.ndarray):
        bold_simulated = torch.from_numpy(bold_simulated).float().to(device)
    else:
        bold_simulated = bold_simulated.float().to(device)
    
    # Check shapes
    if bold_empirical.ndim != 2:
        raise ValueError(f"bold_empirical must be 2D [n_nodes, n_timesamples], got shape {bold_empirical.shape}")
    if bold_simulated.ndim != 3:
        raise ValueError(f"bold_simulated must be 3D [n_nodes, n_sets, n_timesamples], got shape {bold_simulated.shape}")
    
    n_nodes = bold_empirical.shape[0]
    n_timesamples = bold_empirical.shape[1]
    
    if bold_simulated.shape[0] != n_nodes:
        raise ValueError(f"Node dimension mismatch: empirical has {n_nodes}, simulated has {bold_simulated.shape[0]}")
    if bold_simulated.shape[2] != n_timesamples:
        raise ValueError(f"Time dimension mismatch: empirical has {n_timesamples}, simulated has {bold_simulated.shape[2]}")
    
    n_sets = bold_simulated.shape[1]
    n_num = n_sets // n_dup  # Number of unique parameter sets
    
    if n_sets % n_dup != 0:
        raise ValueError(f"n_sets ({n_sets}) must be divisible by n_dup ({n_dup})")
    
    # Calculate time length for sliding windows
    time_length = n_timesamples - window_size + 1
    
    if time_length <= 0:
        raise ValueError(f"window_size ({window_size}) must be less than n_timesamples ({n_timesamples})")
    
    # Compute empirical FCD
    emp_fcd_hist = _compute_fcd_histogram(bold_empirical.unsqueeze(1), window_size, 
                                          time_length, n_nodes, n_bins, fcd_range, device)
    emp_fcd_cumsum = torch.cumsum(emp_fcd_hist, dim=0).cpu().numpy()
    
    # Compute simulated FCD histograms for all sets
    sim_fcd_hist = _compute_fcd_histogram(bold_simulated, window_size, 
                                          time_length, n_nodes, n_bins, fcd_range, device)  # [n_bins, n_sets]
    
    # Convert to cumulative distribution
    sim_fcd_cumsum = torch.cumsum(sim_fcd_hist, dim=0).cpu().numpy()  # [10000, n_sets]
    
    # Identify valid simulations (where the histogram was properly computed)
    # A valid simulation should have the same final cumsum as empirical
    sim_fcd_cumsum_masked = sim_fcd_cumsum.copy()
    valid_sims = sim_fcd_cumsum[-1, :] == emp_fcd_cumsum[-1, 0]
    sim_fcd_cumsum_masked[:, ~valid_sims] = 0
    
    # Average across duplicates for each unique parameter set
    # Data arrangement: [set0_dup0, set0_dup1, ..., set0_dup(n_dup-1), set1_dup0, ...]
    # Reshape to [n_num, n_dup, n_bins]
    sim_fcd_reshaped = sim_fcd_cumsum_masked.T.reshape(n_num, n_dup, -1)  # [n_num, n_dup, n_bins]
    valid_sims_reshaped = valid_sims.reshape(n_num, n_dup)  # [n_num, n_dup]
    
    # Sum across duplicates and count valid ones
    sim_fcd_sum = sim_fcd_reshaped.sum(axis=1)  # [n_num, n_bins]
    valid_count = valid_sims_reshaped.sum(axis=1, keepdims=True).astype(float)  # [n_num, 1]
    
    # Avoid division by zero
    valid_count[valid_count == 0] = np.nan
    
    # Compute average CDF
    sim_fcd_ave = sim_fcd_sum / valid_count  # [n_num, n_bins]
    
    # Calculate KS statistics: max absolute difference between CDFs
    emp_fcd_repeated = np.tile(emp_fcd_cumsum, (1, n_num))  # [n_bins, n_num]
    ks_diff = np.abs(sim_fcd_ave.T - emp_fcd_repeated)  # [n_bins, n_num]
    ks_cost = ks_diff.max(axis=0) / emp_fcd_cumsum[-1, 0]  # Normalize by total count
    
    # Assign high cost (10) to sets where all duplicates were invalid
    ks_cost[np.isnan(ks_cost)] = 10
    ks_cost[sim_fcd_ave[:, -1] != emp_fcd_cumsum[-1, 0]] = 10
    
    if verbose:
        fcd_elapsed = time.time() - fcd_timestart
        print(f'Time for calculating FCD KS statistics cost: {fcd_elapsed:.3f}s')
    
    return ks_cost


def _compute_fcd_histogram(
    bold: torch.Tensor,
    window_size: int,
    time_length: int,
    n_nodes: int,
    n_bins: int,
    fcd_range: tuple,
    device: str
) -> torch.Tensor:
    """
    Compute FCD histogram for BOLD signal(s).
    
    Args:
        bold: BOLD signal of shape [n_nodes, n_sets, n_timesamples]
        window_size: Size of sliding window
        time_length: Number of sliding windows
        n_nodes: Number of nodes
        n_bins: Number of histogram bins
        fcd_range: Range for histogram as (min, max) tuple
        device: Device for computation
        
    Returns:
        fcd_hist: Histogram of FCD values, shape [n_bins, n_sets]
    """
    n_sets = bold.shape[1]
    fc_edgenum = int(n_nodes * (n_nodes - 1) / 2)
    
    # Process in batches to manage memory
    sub_num = min(10, n_sets)  # Process up to 10 sets at a time
    batch_num = n_sets // sub_num
    resid_num = n_sets % sub_num
    
    # Create FC mask
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1) == 1
    
    # Create batched FC mask
    fc_maskm = torch.zeros(n_nodes * sub_num, n_nodes * sub_num, 
                          dtype=torch.bool, device=device)
    for i in range(sub_num):
        fc_maskm[n_nodes * i:n_nodes * (i + 1), 
                n_nodes * i:n_nodes * (i + 1)] = fc_mask
    
    # Mask for upper triangle of FCD matrix
    fcd_mask = torch.triu(torch.ones(time_length, time_length, device=device), diagonal=1) == 1
    
    # Initialize histogram
    fcd_hist = torch.zeros(n_bins, n_sets, device='cpu')
    
    # Process batches
    fc_mat = torch.zeros(fc_edgenum, sub_num, time_length, device=device)
    
    for b in range(batch_num):
        bold_batch = bold[:, b * sub_num:(b + 1) * sub_num, :]
        bold_batch_reshaped = bold_batch.transpose(0, 1).contiguous().view(-1, bold.shape[2])
        
        # Compute FC for each time window
        for i in range(time_length):
            bold_window = bold_batch_reshaped[:, i:i + window_size]
            
            # Check for NaN in this window
            if torch.isnan(bold_window).any():
                fc_mat[:, :, i] = float('nan')
            else:
                bold_fc = torch_corr(bold_window)
                fc_mat[:, :, i] = bold_fc[fc_maskm].view(sub_num, fc_edgenum).T
        
        # Compute FCD for each set in batch
        for j in range(sub_num):
            fc_series = fc_mat[:, j, :]  # [fc_edgenum, time_length]
            
            # Check for NaN
            if torch.isnan(fc_series).any():
                fcd_hist[:, j + b * sub_num] = 0  # Mark as invalid
            else:
                fcd_temp = torch_corr(fc_series.T)  # Correlate across time windows
                fcd_hist[:, j + b * sub_num] = torch.histc(
                    fcd_temp[fcd_mask].cpu(), bins=n_bins, min=fcd_range[0], max=fcd_range[1])
    
    # Handle residual sets
    if resid_num > 0:
        fc_mask_resid = torch.zeros(n_nodes * resid_num, n_nodes * resid_num,
                                    dtype=torch.bool, device=device)
        for i in range(resid_num):
            fc_mask_resid[n_nodes * i:n_nodes * (i + 1),
                         n_nodes * i:n_nodes * (i + 1)] = fc_mask
        
        fc_resid = torch.zeros(fc_edgenum, resid_num, time_length, device=device)
        bold_resid = bold[:, batch_num * sub_num:n_sets, :]
        bold_resid_reshaped = bold_resid.transpose(0, 1).contiguous().view(-1, bold.shape[2])
        
        for i in range(time_length):
            bold_window = bold_resid_reshaped[:, i:i + window_size]
            
            if torch.isnan(bold_window).any():
                fc_resid[:, :, i] = float('nan')
            else:
                bold_fc = torch_corr(bold_window)
                fc_resid[:, :, i] = bold_fc[fc_mask_resid].view(resid_num, fc_edgenum).T
        
        for j in range(resid_num):
            fc_series = fc_resid[:, j, :]
            
            if torch.isnan(fc_series).any():
                fcd_hist[:, j + sub_num * batch_num] = 0
            else:
                fcd_temp = torch_corr(fc_series.T)
                fcd_hist[:, j + sub_num * batch_num] = torch.histc(
                    fcd_temp[fcd_mask].cpu(), bins=n_bins, min=fcd_range[0], max=fcd_range[1])
    
    return fcd_hist


