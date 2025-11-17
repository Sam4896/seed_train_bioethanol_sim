# Copyright (c) 2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
# See LICENSE.md for license information.
"""Bayesian optimization example using SVGP and the bioethanol multi-stage simulator."""

import torch
from typing import Dict

from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.models import SingleTaskVariationalGP
from botorch.models.utils.inducing_point_allocators import GreedyImprovementReduction
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import PredictiveLogLikelihood

from bioethanol_multi_stage import BioEthanolMultiStageSeedTrain
from dag_experiment_base import DAGSyntheticTestFunction


class LogTransform(OutcomeTransform):
    """Log transform with numerical stability to prevent log(0) or log(negative)."""

    def __init__(self, min_value: float = 1e-6):
        """Initialize log transform.

        Args:
            min_value: Minimum value to clamp inputs to before taking log.
        """
        super().__init__()
        self.min_value = min_value

    def forward(
        self, Y: torch.Tensor, Yvar: torch.Tensor | None = None, X=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply log transform with numerical stability."""
        # Clamp to minimum value to avoid log(0) or log(negative)
        Y_clamped = torch.clamp(Y, min=self.min_value)
        Y_transformed = torch.log(Y_clamped)

        # Transform variance if provided (using delta method approximation)
        if Yvar is not None:
            Yvar_transformed = Yvar / (Y_clamped**2)
            return Y_transformed, Yvar_transformed
        return Y_transformed, None

    def untransform(
        self, Y: torch.Tensor, Yvar: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reverse log transform."""
        Y_untransformed = torch.exp(Y)

        # Transform variance back if provided
        if Yvar is not None:
            Yvar_untransformed = Yvar * (Y_untransformed**2)
            return Y_untransformed, Yvar_untransformed
        return Y_untransformed, None

    def untransform_posterior(self, posterior, X=None):
        """Untransform posterior distribution."""
        return posterior


def initialize_data(
    sim_env: DAGSyntheticTestFunction, n_init: int = 10
) -> Dict[str, torch.Tensor]:
    """Create initial UNTRANSFORMED training data using Sobol samples and simulator outputs.

    Args:
        sim_env: The simulation environment
        n_init: The number of initial data points

    Returns:
        The initial training data
    """
    x = sim_env.get_sobol_samples(n_init)
    outputs = sim_env(x)
    return {
        "inputs": x,
        "Objective": outputs["Objective"],
    }


def transform_data(
    sim_env: DAGSyntheticTestFunction,
    data_dict: Dict[str, torch.Tensor],
    outcome_transform: OutcomeTransform | None = None,
) -> tuple[Dict[str, torch.Tensor], OutcomeTransform]:
    """Normalize inputs to [0,1] and apply outcome transform.

    Args:
        sim_env: The simulation environment
        data_dict: Raw data dictionary
        outcome_transform: Optional pre-fitted outcome transform

    Returns:
        Transformed data dictionary and fitted outcome transform
    """
    transformed = {
        "inputs": normalize(data_dict["inputs"], bounds=sim_env.bounds),
        "Objective": data_dict["Objective"],
    }

    # Create or use provided outcome transform
    if outcome_transform is None:
        # Create ChainedTransform: Log -> Standardize
        log_transform = LogTransform(min_value=1e-6)
        std_transform = Standardize(m=transformed["Objective"].shape[-1])

        # Apply log transform first (no fitting needed)
        transformed["Objective"], _ = log_transform(transformed["Objective"])
        # Then apply standardize transform (fits on data)
        std_transform.train()  # Ensure in training mode for fitting
        transformed["Objective"], _ = std_transform(transformed["Objective"])

        # Store both transforms for later use
        class ChainedTransform(OutcomeTransform):
            def __init__(self, log_tf, std_tf):
                super().__init__()
                self.log_transform = log_tf
                self.standardize_transform = std_tf

            def forward(self, Y, Yvar=None, X=None):
                Y, Yvar = self.log_transform.forward(Y, Yvar)
                return self.standardize_transform.forward(Y, Yvar, X=X)

            def untransform(self, Y, Yvar=None):
                Y, Yvar = self.standardize_transform.untransform(Y, Yvar)
                return self.log_transform.untransform(Y, Yvar)

            def untransform_posterior(self, posterior, X=None):
                return self.log_transform.untransform_posterior(
                    self.standardize_transform.untransform_posterior(posterior, X=X)
                )

        outcome_transform = ChainedTransform(log_transform, std_transform)
    else:
        # Apply existing transform (should already be fitted)
        transformed["Objective"], _ = outcome_transform(transformed["Objective"])

    outcome_transform.eval()

    return transformed, outcome_transform


def build_svgp_model(
    transformed_data: Dict[str, torch.Tensor],
    outcome_transform: OutcomeTransform,
    device: torch.device | None = None,
    num_inducing_points: int | None = None,
) -> SingleTaskVariationalGP:
    """Construct and fit a SingleTaskVariationalGP model with GreedyImprovementReduction.

    Args:
        transformed_data: Training data with inputs normalized to [0,1] and outputs transformed
        outcome_transform: The outcome transform to use
        device: Device to run on
        num_inducing_points: Number of inducing points to use. If None, uses min(100, num_training_points)

    Returns:
        A fitted SVGP model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = transformed_data["inputs"].to(device)
    train_y = transformed_data["Objective"].to(device)

    # Determine number of inducing points
    n_train = train_x.shape[-2]
    if num_inducing_points is None:
        num_inducing_points = min(100, n_train)
    else:
        num_inducing_points = min(num_inducing_points, n_train)

    # Create a temporary model to use with GreedyImprovementReduction
    # This is needed because the allocator requires a model to compute quality
    temp_model = SingleTaskVariationalGP(
        train_X=train_x,
        train_Y=train_y,
        inducing_points=num_inducing_points,  # Use all training points initially
        outcome_transform=outcome_transform,
    )

    # Fit the temporary model briefly to get a reasonable model for allocation
    temp_mll = PredictiveLogLikelihood(
        temp_model.likelihood, temp_model.model, num_data=train_x.shape[0]
    )
    fit_gpytorch_mll(
        temp_mll,
        optimizer_kwargs={"options": {"maxiter": 10}},  # Quick fit
    )

    # Create GreedyImprovementReduction allocator with the temporary model
    # This selects inducing points that maximize improvement in promising regions
    inducing_point_allocator = GreedyImprovementReduction(
        model=temp_model,
        maximize=True,
    )

    # Create final SVGP model with GreedyImprovementReduction
    model = SingleTaskVariationalGP(
        train_X=train_x,
        train_Y=train_y,
        inducing_points=num_inducing_points,  # Number of inducing points
        inducing_point_allocator=inducing_point_allocator,
        learn_inducing_points=False,  # Don't optimize inducing points after allocation
        outcome_transform=outcome_transform,
    )

    # Use PredictiveLogLikelihood for variational models
    mll = PredictiveLogLikelihood(
        model.likelihood, model.model, num_data=train_x.shape[0]
    )

    # Fit the model
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": 200}},
    )

    return model


def propose_candidate_with_qlogei(
    model: SingleTaskVariationalGP,
    sim_env: DAGSyntheticTestFunction,
    best_f: float,
    q: int = 1,
) -> torch.Tensor:
    """Propose next point by optimizing qLogEI over the objective.

    Args:
        model: The SVGP model (with GreedyImprovementReduction for inducing points)
        sim_env: The simulation environment
        best_f: The current best observed objective value (in transformed space)
        q: Number of candidates to propose

    Returns:
        The proposed candidate(s) in normalized [0,1] space
    """
    # Build qLogEI
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

    def objective_transform(samples, X=None):  # noqa: N803
        # Extract objective (last/only output column) from samples
        return samples[..., -1]

    objective = GenericMCObjective(objective_transform)

    acqf = qLogExpectedImprovement(
        model=model, best_f=best_f, sampler=sampler, objective=objective
    )

    # Optimize over unit cube; the model is trained on normalized inputs
    bounds = torch.stack(
        [
            torch.zeros(
                sim_env.dim, dtype=sim_env.bounds.dtype, device=sim_env.bounds.device
            ),
            torch.ones(
                sim_env.dim, dtype=sim_env.bounds.dtype, device=sim_env.bounds.device
            ),
        ]
    )

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=8,
        raw_samples=256,
        options={"batch_limit": 32, "maxiter": 200},
    )

    return candidate.detach()


def bo_loop(n_init: int = 10, n_iter: int = 10, observation_noise_std: float = 0.05):
    """Run a simple BO loop with qLogEI using the bioethanol simulator.

    Args:
        n_init: Number of initial samples
        n_iter: Number of BO iterations
        observation_noise_std: Standard deviation of observation noise
    """
    # Initialize simulator
    sim_env = BioEthanolMultiStageSeedTrain(
        observation_noise_std=observation_noise_std,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data
    data_raw = initialize_data(sim_env, n_init=n_init)
    transformed_data, outcome_transform = transform_data(sim_env, data_raw)

    # Build initial model
    model = build_svgp_model(transformed_data, outcome_transform, device=device)
    model = model.to(device)

    # Track best values (in transformed and untransformed space)
    best_f_transformed = transformed_data["Objective"].max().item()
    best_f_untransformed = data_raw["Objective"].max().item()
    print(f"Initial best (untransformed): {best_f_untransformed:.6f}")

    for t in range(n_iter):
        # Propose candidate
        x_new = propose_candidate_with_qlogei(model, sim_env, best_f_transformed, q=1)

        # Evaluate simulator in the original space
        x_new_un = unnormalize(x_new, bounds=sim_env.bounds)
        out_raw = sim_env(x_new_un)

        # Append to dataset
        data_raw["inputs"] = torch.cat([data_raw["inputs"], x_new_un], dim=-2)
        data_raw["Objective"] = torch.cat(
            [data_raw["Objective"], out_raw["Objective"]], dim=-2
        )

        # Refit model with new data (refit transform on all data)
        transformed_data, outcome_transform = transform_data(
            sim_env,
            data_raw,
            outcome_transform=None,  # Refit transform on all data
        )
        model = build_svgp_model(transformed_data, outcome_transform, device=device)
        model = model.to(device)

        # Update best values
        best_f_transformed = max(
            best_f_transformed, transformed_data["Objective"].max().item()
        )
        best_f_untransformed = max(
            best_f_untransformed, data_raw["Objective"].max().item()
        )
        print(
            f"Iteration {t + 1}/{n_iter}: best (untransformed) = {best_f_untransformed:.6f}"
        )

    print(f"Final best (untransformed): {best_f_untransformed:.6f}")
    return data_raw, model


if __name__ == "__main__":
    bo_loop(n_init=10, n_iter=5, observation_noise_std=0.05)
