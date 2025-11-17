# © 2025 Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
from typing import Dict, List, Optional

import torch
import warnings

from dag_experiment_base import DAGSyntheticTestFunction

# Make sure these relative imports are correct for your project structure
# Import the new 3-stage simulator and feature extractor
from bioethanol_multi_stage_ode import BioEthanolMultiStageODESimulator
from bioethanol_feature_extractor import BioEthanolMultiStageFeatureExtractor

V_MAX_LITERS = 20000.0  # Maximum volume for a single tank


class BioEthanolMultiStageSeedTrain(DAGSyntheticTestFunction):
    """Citable 3-Stage Multi-Stage Process (POGPN DAG) for Bioethanol.

    This class implements the 26-dimensional, 3-stage multi-stage process
    (Flask -> Seed Fermenter -> Production) as a citable benchmark for process-aware
    Bayesian Optimization.

    Input vector (dim=26) order:
    --- Stage 1 (Flask, Batch): 6 inputs [0-5] ---
    0: X0_1 (Initial biomass, g/L)
    1: S0_1 (Initial substrate, g/L)
    2: V0_1 (Initial volume, L)
    3: T_1 (Static Temperature, K)
    4: pH_1 (Static pH)
    5: t_final_1 (Duration, hr)
    --- Stage 2 (Seed, Fed-Batch): 10 inputs [6-15] ---
    6: V0_2 (Initial medium volume, L)
    7: S0_2 (Initial medium substrate, g/L)
    8: S_in_2 (Feed substrate, g/L)
    9: f_frac_2 (Fraction of volume per hour): Later converted to feed rate (L/hr) by multiplying with V0_2 (L)
    10: mu_set_2 (Feed rate parameter, hr^-1)
    11: T_initial_2 (Start Temp, K)
    12: T_final_2 (End Temp, K)
    13: pH_initial_2 (Start pH)
    14: pH_final_2 (End pH)
    15: t_final_2 (Duration, hr)
    --- Stage 3 (Production, Fed-Batch): 10 inputs [16-25] ---
    16: V0_3 (Initial medium volume, L)
    17: S0_3 (Initial medium substrate, g/L)
    18: S_in_3 (Feed substrate, g/L)
    19: f_frac_3 (Fraction of volume per hour): Later converted to feed rate (L/hr) by multiplying with V0_3 (L)
    20: mu_set_3 (Feed rate parameter, hr^-1)
    21: T_initial_3 (Start Temp, K)
    22: T_final_3 (End Temp, K)
    23: pH_initial_3 (Start pH)
    24: pH_final_3 (End pH)
    25: t_final_3 (Duration, hr)


    # Mermaid DAG diagram for BioEthanolMultiStage
    # This shows the flow of information (latent variables) and
    # how controls affect each stage, as modeled by POGPN.

    ```mermaid
    flowchart TD
        subgraph Stage 1: Shake Flask
            P1_controls["P1_controls (6)"] --> P1_obs["P1_obs (Xf, Sf, Vf, Pf)"]
        end
        subgraph Stage 2: Seed Fermenter
            P1_obs --> P2_obs["P2_obs (Xf, Sf, Vf, Pf)"]
            P2_controls["P2_controls (10)"] --> P2_obs
        end
        subgraph Stage 3: Production
            P2_obs --> P3_obs["P3_obs (Pf)"]
            P3_controls["P3_controls (10)"] --> P3_obs
        end
        subgraph Final Objective
            P3_obs --> Objective["Objective (P_STY_gL_hr)"]
        end
    ```
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        step_size: float = 0.01,
        process_stochasticity_std: Optional[float] = None,
        observation_noise_std: Optional[float] = None,
    ) -> None:
        # Accept dim from config for compatibility; enforce 26D
        if dim is not None and dim != 26:
            warnings.warn(
                f"BioEthanolMultiStageSeedTrain expects dim=26, received dim={dim}. Overriding to 26."
            )
        dim = 26  # Total number of input dimensions

        # Citable bounds for the 26-dimensional multi-stage process
        # (Based on original bounds, adapted for a 3-stage process)
        bounds = [
            # --- Stage 1 (Flask) [0-5] ---
            (0.5, 10.0),  # 0: X0_1 (g/L)
            (10.0, 80.0),  # 1: S0_1 (g/L)
            (0.5, 2.0),  # 2: V0_1 (L)
            (298.15, 313.15),  # 3: T_1 (K)
            (3.5, 5.5),  # 4: pH_1
            (6.0, 48.0),  # 5: t_final_1 (hr)
            # --- Stage 2 (Seed) [6-15] ---
            (50.0, 200.0),  # 6: V0_2 (L)
            (10.0, 80.0),  # 7: S0_2 (g/L)
            (100.0, 600.0),  # 8: S_in_2 (g/L)
            (0.05, 5.0),  # 9: F0_2 (L/hr)  ← absolute feed
            (0.01, 0.25),  # 10: mu_set_2 (hr^-1)
            (298.15, 313.15),  # 11: T_initial_2 (K)
            (298.15, 313.15),  # 12: T_final_2 (K)
            (3.5, 5.5),  # 13: pH_initial_2
            (3.5, 5.5),  # 14: pH_final_2
            (12.0, 96.0),  # 15: t_final_2 (hr)
            # --- Stage 3 (Production) [16-25] ---
            (500.0, 5000.0),  # 16: V0_3 (L)
            (10.0, 80.0),  # 17: S0_3 (g/L)
            (100.0, 600.0),  # 18: S_in_3 (g/L)
            (0.5, 50.0),  # 19: F0_3 (L/hr)  ← absolute feed
            (0.01, 0.25),  # 20: mu_set_3 (hr^-1)
            (298.15, 313.15),  # 21: T_initial_3 (K)
            (298.15, 313.15),  # 22: T_final_3 (K)
            (3.5, 5.5),  # 23: pH_initial_3
            (3.5, 5.5),  # 24: pH_final_3
            (48.0, 168.0),  # 25: t_final_3 (hr)
        ]

        # Map input indices to the root nodes of the DAG
        root_node_indices_dict: Dict[str, List[int]] = {
            "P1_controls": list(range(0, 6)),
            "P2_controls": list(range(6, 16)),
            "P3_controls": list(range(16, 26)),
        }

        # Define the names of the *observed* nodes in the DAG
        observed_output_node_names: List[str] = [
            "P1_obs",
            "P2_obs",
            "P3_obs",
            "Objective",
        ]

        super().__init__(
            dim=dim,
            negate=False,  # We want to MAXIMIZE CTWI
            bounds=bounds,
            observed_output_node_names=observed_output_node_names,
            root_node_indices_dict=root_node_indices_dict,
            objective_node_name="Objective",
            process_stochasticity_std=process_stochasticity_std,
            observation_noise_std=observation_noise_std,
        )

        # Choose simulator: ODE by default; SDE if process noise is requested
        if process_stochasticity_std is not None and process_stochasticity_std > 0.0:
            warnings.warn(
                "Process stochasticity is not supported for BioEthanolMultiStage. Using ODE instead."
            )
            self.sim = BioEthanolMultiStageODESimulator(
                v_max_stages=(2.0, 500.0, V_MAX_LITERS)
            )
        else:
            # Use realistic stage-specific reactor capacities (L): Flask, Seed, Production
            self.sim = BioEthanolMultiStageODESimulator(
                v_max_stages=(2.0, 500.0, V_MAX_LITERS)
            )

        self.fx = BioEthanolMultiStageFeatureExtractor()
        self.step_size = float(step_size)

    def _reconstruct_inputs(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct the 26-dim input tensor from the sliced DAG inputs.

        This function reverses the sparse DAG slicing defined in __init__.
        """
        # Get tensors from the sliced input dictionary
        P1_controls = input_dict["P1_controls"]
        P2_controls = input_dict["P2_controls"]
        P3_controls = input_dict["P3_controls"]

        # Reconstruct the standard 26-element order
        inputs = torch.cat([P1_controls, P2_controls, P3_controls], dim=-1)
        return inputs

    def _evaluate_true(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the true, deterministic 3-stage process."""
        # Reconstruct the (batch, 26) input tensor
        inputs = self._reconstruct_inputs(input_dict=input_dict)

        # Run the full 3-stage simulation
        # This returns a nested dict of all series and sim inputs
        sim_output_dict = self.sim.simulate(inputs, step_size=self.step_size)

        # Extract the DAG tensor nodes (P1_obs, P2_obs, P3_obs, Objective)
        output = self.fx.get_dag_tensors(sim_output_dict)

        # Add the root node inputs back into the output dict
        # (This is required by the POGPN framework)
        output["P1_controls"] = input_dict["P1_controls"]
        output["P2_controls"] = input_dict["P2_controls"]
        output["P3_controls"] = input_dict["P3_controls"]
        return output

    def _evaluate_noisy(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the process with observation noise."""
        # Get the true, deterministic outputs
        output = self._evaluate_true(input_dict)

        sigma = self.observation_noise_std or 0.0
        if sigma and sigma > 0.0:
            # Add proportional noise to each *observed* node
            output["P1_obs"] = self._add_proportional_noise(output["P1_obs"], sigma)
            output["P2_obs"] = self._add_proportional_noise(output["P2_obs"], sigma)
            output["P3_obs"] = self._add_proportional_noise(output["P3_obs"], sigma)
            output["Objective"] = self._add_proportional_noise(
                output["Objective"], sigma
            )
        return output

    @property
    def V_max_liters(self) -> float:
        # Return production stage V_max where objective is measured
        if hasattr(self.sim, "tank_sim_P3"):
            return self.sim.tank_sim_P3.V_max_liters
        if hasattr(self.sim, "tank_sim"):
            return self.sim.tank_sim.V_max_liters
        return V_MAX_LITERS
