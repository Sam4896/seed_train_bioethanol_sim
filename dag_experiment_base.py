# Copyright (c) 2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
# See LICENSE.md for license information.
from abc import abstractmethod, ABC
from typing import Dict, List, Optional, Tuple
import torch
from botorch.utils.transforms import unnormalize
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.posteriors import GPyTorchPosterior
import gpytorch


def convert_tensor_to_dict(
    combined_tensor: torch.Tensor,
    node_indices_dict: Dict[str, List[int]],
    dim: int = -1,
) -> Dict[str, torch.Tensor]:
    """Convert a tensor to a dictionary of tensors based on index lists.

    Args:
        combined_tensor: Tensor of shape (n, d) where n is the number of samples and d is the total dimension of the input nodes.
        node_indices_dict: Dictionary mapping node names to the list of indices in combined_tensor that belong to that node.
        dim: Dimension to index along (should be -1 for last dim).

    Returns:
        Dictionary of tensors with the node names as keys and the corresponding tensor slices as values.

    """
    out = {}
    for node, idx in node_indices_dict.items():
        out[node] = combined_tensor.index_select(
            dim, torch.tensor(idx, device=combined_tensor.device)
        )
    return out


def convert_dict_to_tensor(
    node_data_dict: Dict[str, torch.Tensor],
    node_indices_dict: Dict[str, List[int]],
) -> torch.Tensor:
    """Place each tensor's channels into the combined tensor at the given indices.

    along the last dimension. Works with shapes (..., C_k) sharing the same
    leading dimensions.

    Example: node_data_dict = {'x1': (N,3), 'x2': (N,2)}
             node_indices_dict = {'x1': [0,1,4], 'x2': [2,3]}
             -> output shape (N, 5), ordered [x1_0, x1_1, x2_0, x2_1, x1_2]
    """
    # Use the first tensor to infer leading dims, dtype, device
    first = next(iter(node_data_dict.values()))
    leading = first.shape[:-1]
    total_C = sum(t.shape[-1] for t in node_data_dict.values())

    out = first.new_zeros(*leading, total_C)

    for k, t in node_data_dict.items():
        if t.shape[:-1] != leading:
            raise ValueError(
                f"All tensors must share leading shape {leading}, got {t.shape}."
            )
        idx = torch.as_tensor(node_indices_dict[k], dtype=torch.long, device=out.device)
        if t.shape[-1] != idx.numel():
            raise ValueError(
                f"Mismatched channels for '{k}': got {t.shape[-1]} vs {idx.numel()} indices."
            )
        # Copy channels into their slots along the last dimension
        out.index_copy_(-1, idx, t)

    return out


def consolidate_mvn_mixture(mvn_batch, weights=None, eps=1e-6):
    """Reduce a batch of MVNs (batch dim M) to a single MVN via moment matching.

    dist.mean: shape [M, D]
    dist.covariance_matrix: shape [M, D, D]
    weights: optional [M] (unnormalized OK). If None, uses uniform.
    """
    if isinstance(mvn_batch, GPyTorchPosterior):
        is_posterior = True
        dist = mvn_batch.distribution
    else:
        is_posterior = False
        dist = mvn_batch

    mus = dist.mean  # [M, D]
    covs = dist.covariance_matrix  # [M, D, D]
    M, D = mus.shape

    if weights is None:
        weights = torch.full((M,), 1.0 / M, dtype=mus.dtype, device=mus.device)
    else:
        weights = weights.to(mus) / weights.sum()

    w = weights.view(M, 1)
    mu = (w * mus).sum(dim=0)  # [D]

    # Σ = Σ_i w_i (Σ_i + μ_i μ_i^T) - μ μ^T
    outer = mus.unsqueeze(-1) @ mus.unsqueeze(-2)  # [M, D, D]
    second = (weights.view(M, 1, 1) * (covs + outer)).sum(dim=0)  # [D, D]
    Sigma = second - (mu.unsqueeze(-1) @ mu.unsqueeze(-2))  # [D, D]

    # Stabilize & symmetrize
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    Sigma = Sigma + eps * torch.eye(D, dtype=Sigma.dtype, device=Sigma.device)

    if is_posterior:
        return GPyTorchPosterior(gpytorch.distributions.MultivariateNormal(mu, Sigma))
    else:
        return gpytorch.distributions.MultivariateNormal(mu, Sigma)


def consolidate_mtmvn_mixture(mtmvn_batch, weights=None, eps: float = 1e-6):
    """Reduce a batch of MTMVNs across the leading batch dim via moment matching.

    Preserves cross-task correlations and the original interleaving layout.

    Args:
        mtmvn_batch: A batch of MTMVNs.
        weights: Optional weights for the MTMVNs.
        eps: Epsilon for numerical stability.

    Returns:
        A consolidated MTMVN (or a GPyTorchPosterior over it if that was the input).

    """
    try:
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        is_posterior = isinstance(mtmvn_batch, GPyTorchPosterior)
    except Exception:
        GPyTorchPosterior = None
        is_posterior = False

    dist = mtmvn_batch.distribution if is_posterior else mtmvn_batch
    assert isinstance(dist, gpytorch.distributions.MultitaskMultivariateNormal)

    interleaved = bool(
        getattr(dist, "interleaved", getattr(dist, "_interleaved", True))
    )
    T = dist.num_tasks

    # mean: [M, N, T] (or [N, T] -> add M=1)
    mus = dist.mean
    if mus.dim() == 2:
        mus = mus.unsqueeze(0)
    M, N, TT = mus.shape
    assert TT == T

    # cov: [M, N*T, N*T] (or [N*T, N*T] -> add M=1)
    covs = dist.covariance_matrix
    if covs.dim() == 2:  # type: ignore
        covs = covs.unsqueeze(0)  # type: ignore
    assert covs.shape == (M, N * T, N * T)  # type: ignore

    # flatten means to match covariance ordering
    mus_flat = (
        mus.reshape(M, N * T) if interleaved else mus.permute(0, 2, 1).reshape(M, N * T)
    )

    # weights over batch components
    if weights is None:
        weights = torch.full(
            (M,), 1.0 / M, dtype=mus_flat.dtype, device=mus_flat.device
        )
    else:
        weights = weights.to(mus_flat).reshape(M)
        weights = weights / weights.sum()

    # moment matching in NT-dimensional space
    w = weights.view(M, 1)
    mu_flat = (w * mus_flat).sum(dim=0)  # [NT]
    outer = mus_flat.unsqueeze(-1) @ mus_flat.unsqueeze(-2)  # [M, NT, NT]
    second = (weights.view(M, 1, 1) * (covs + outer)).sum(dim=0)  # [NT, NT]
    Sigma = second - (mu_flat.unsqueeze(-1) @ mu_flat.unsqueeze(-2))  # [NT, NT]

    # stabilize
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    Sigma = Sigma + eps * torch.eye(N * T, dtype=Sigma.dtype, device=Sigma.device)

    # reshape mean to [N, T] consistent with layout; covariance stays [NT, NT]
    if interleaved:
        mu_2d = mu_flat.view(N, T)
    else:
        mu_2d = mu_flat.view(T, N).transpose(-1, -2)  # -> [N, T]

    consolidated = gpytorch.distributions.MultitaskMultivariateNormal(
        mu_2d, Sigma, interleaved=interleaved
    )

    return GPyTorchPosterior(consolidated) if is_posterior else consolidated  # type: ignore


def handle_nans_and_create_mask(data_dict, imputation_value=-999.0):
    """Create a mask for NaNs in the data_dict and impute them.

    Args:
        data_dict: A dictionary of tensors.
        imputation_value: The value to use for imputing NaNs.

    Returns:
        A tuple containing:
            - imputed_data_dict: The data dictionary with NaNs imputed.
            - masks_dict: A dictionary of boolean masks, where True indicates a row
                that originally contained a NaN.

    """
    imputed_data_dict = {}
    masks_dict = {}
    for key, tensor in data_dict.items():
        if isinstance(tensor, torch.Tensor):
            imputed_tensor = tensor.clone()
            if torch.isnan(imputed_tensor).any():
                row_mask = torch.isnan(imputed_tensor).any(dim=-1)
                masks_dict[key] = row_mask
                imputed_tensor[row_mask] = imputation_value
                imputed_data_dict[key] = imputed_tensor
            else:
                imputed_data_dict[key] = imputed_tensor
        else:
            imputed_data_dict[key] = tensor

    return imputed_data_dict, masks_dict


class DAGSyntheticTestFunction(SyntheticTestFunction, ABC):
    """Base class for experiment functions.

    This abstract base class provides the foundation for all experiment functions.
    It handles input and output validation while requiring subclasses to implement
    the actual evaluation logic.

    Args:
        dim: The dimension of the input space.
        negate: Whether to negate the function.
        bounds: The bounds of the input space.
        device: The device to run the function on.
        is_observation_noisy: Whether to add observation noise.
        is_stochastic: Whether to add process noise.
        observed_output_node_names: The names of the observed output nodes.
        root_node_indices: The indices of the root nodes.

    """

    def __init__(
        self,
        dim: int,
        negate: bool,
        bounds: List[Tuple[float, float]],
        observed_output_node_names: List[str],
        root_node_indices_dict: Dict[str, List[int]],
        objective_node_name: str,
        process_stochasticity_std: Optional[float] = None,
        observation_noise_std: Optional[float] = None,
    ):
        self.dim = dim
        self.continuous_inds = list(range(dim))

        super().__init__(negate=negate, bounds=bounds)

        self.observed_output_node_names = observed_output_node_names
        self.root_node_indices_dict = root_node_indices_dict
        self.objective_node_name = objective_node_name

        # Set process stochasticity
        self.process_stochasticity_std = (
            process_stochasticity_std if process_stochasticity_std is not None else 0.0
        )
        self.is_stochastic = (
            process_stochasticity_std is not None and process_stochasticity_std > 0.0
        )

        # Set observation noise
        self.observation_noise_std = (
            observation_noise_std if observation_noise_std is not None else 0.0
        )
        self.is_observation_noisy = (
            observation_noise_std is not None and observation_noise_std > 0.0
        )

        # Find optimum
        if not hasattr(self, "_optimal_value"):
            self.optimizers, self._optimal_value = (
                self.find_optimum_multi_start_high_dim()
            )

        self._validate_root_node_indices()

    def _validate_root_node_indices(self):
        if self.root_node_indices_dict is None:
            raise ValueError("root_node_indices_dict must be set.")
        # root_node_indices is Dict[str, List[int]], all indices must be unique and cover range(dim)
        all_inds = [i for inds in self.root_node_indices_dict.values() for i in inds]
        if sorted(all_inds) != list(range(self.dim)):
            raise ValueError(
                f"root_node_indices must cover all input indices 0..{self.dim - 1} exactly once. Got: {all_inds}"
            )

    def _validate_output_shape(self, output: torch.Tensor) -> torch.Tensor:
        """Validate the output shape of the function."""
        if output.ndim == 1:
            output = output.unsqueeze(-1)
        return output

    def _add_proportional_noise(
        self, signal: torch.Tensor, std: Optional[float]
    ) -> torch.Tensor:
        """Add noise proportional to the signal magnitude.

        Args:
            signal: The tensor to add noise to.
            std: The standard deviation of the noise, scaled by the signal.
                 If None or 0.0, no noise is added.

        Returns:
            The noisy tensor.

        """
        if std is not None and std > 0:
            return signal + torch.abs(signal) * torch.randn_like(signal) * std
        return signal

    @abstractmethod
    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the function without noise."""
        pass

    @abstractmethod
    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the function with system and observation noise."""
        pass

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the function on a batch of points X.

        Args:
            X: A `batch_shape x d`-dim tensor of inputs.

        Returns:
            A dictionary of function evaluations.

        """
        if X.shape[-1] != self.dim:
            raise ValueError(f"Input dimension must be {self.dim}")

        input_dict = convert_tensor_to_dict(
            combined_tensor=X, node_indices_dict=self.root_node_indices_dict
        )

        if self.is_observation_noisy:
            output = self._evaluate_noisy(input_dict)
        else:
            output = self._evaluate_true(input_dict)

        for key in self.observed_output_node_names:
            if self.negate and key in [self.objective_node_name]:
                output[key] = -self._validate_output_shape(output[key])
            else:
                output[key] = self._validate_output_shape(output[key])

        output["inputs"] = X

        return output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  dim={self.dim},\n"
            f"  negate={self.negate},\n"
            f"  device={self.device},\n"
            f"  is_observation_noisy={self.is_observation_noisy},\n"
            f"  is_stochastic={self.is_stochastic},\n"
            f"  observation_noise_std={self.observation_noise_std},\n"
            f"  process_stochasticity_std={self.process_stochasticity_std},\n"
            f"  bounds={self.bounds},\n"
            f"  observed_output_node_names={self.observed_output_node_names},\n"
            f"  root_node_indices_dict={self.root_node_indices_dict},\n"
            f")"
        )

    def get_output_tensor(
        self,
        outputs_dict: Dict[str, torch.Tensor],
        node_order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Convert outputs from dictionary to concatenated tensor.

        Args:
            outputs_dict: Dictionary containing output tensors
            node_order: Optional list of node names to order the output tensor by

        Returns:
            A tensor of all outputs concatenated along the last dimension

        """
        output_tensors = []
        if node_order is None:
            node_order = sorted(self.observed_output_node_names)
        for node_name in node_order:
            if node_name not in outputs_dict:
                raise ValueError(f"Missing output for node {node_name}")
            output_tensors.append(outputs_dict[node_name])

        return torch.cat(output_tensors, dim=-1)

    def get_sobol_samples(self, n_samples: int) -> torch.Tensor:
        """Get Sobol samples from the bounds of the sim env.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            A `n_samples x dim` tensor of Sobol samples.

        """
        device, dtype = self.bounds.device, self.bounds.dtype
        sobol = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True)
        samples = sobol.draw(n_samples).to(device=device, dtype=dtype)  # type: ignore
        samples = unnormalize(samples, self.bounds)  # type: ignore
        return samples

    def find_optimum_multi_start_high_dim(self, n_starts=10, n_steps=500):
        """Find the optimum of the function using multi-start optimization."""
        dim = self.dim  # 100 in this case
        bounds = self.bounds
        device, dtype = bounds.device, bounds.dtype

        best_value = float("-inf")
        best_point = None

        # Generate quasi-random starting points instead of purely random
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
        start_points = sobol.draw(n_starts).to(device=device, dtype=dtype)  # type: ignore
        # start_points = bounds[0] + (bounds[1] - bounds[0]) * start_points
        start_points = unnormalize(start_points, bounds)  # type: ignore

        for i in range(n_starts):
            x = start_points[i, :].unsqueeze(-2)
            x.requires_grad_(True)

            # Use a more aggressive optimization scheme
            optimizer = torch.optim.Adam([x], lr=0.05)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=50, factor=0.5
            )

            prev_loss = float("inf")
            for step in range(n_steps):
                optimizer.zero_grad()
                outputs_dict = self(x)
                obj = -outputs_dict[self.objective_node_name]  # Negate for maximization
                obj.backward()

                # Track progress for scheduler
                if step % 10 == 0:
                    scheduler.step(obj.item())

                    # Early stopping if progress is minimal
                    if abs(prev_loss - obj.item()) < 1e-7:
                        break
                    prev_loss = obj.item()

                optimizer.step()

                # Project back to bounds
                with torch.no_grad():
                    x.data = torch.max(torch.min(x.data, bounds[1]), bounds[0])  # type: ignore

            outputs_dict = self(x)
            value = outputs_dict[self.objective_node_name].item()
            if value > best_value:
                best_value = value
                best_point = x.detach().clone()

        if best_point is None:
            raise ValueError("Optimization failed to find a valid point")
        return best_point.detach().cpu().numpy(), best_value
