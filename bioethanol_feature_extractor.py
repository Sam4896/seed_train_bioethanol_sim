# Copyright (c) 2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
# See LICENSE.md for license information.
"""Feature extraction for the 3-stage bioethanol seed train simulator.

This module computes features for each stage of the seed train process
and a final scalar objective from the time-series outputs of the
`BioEthanolMultiStageODESimulator`.

The features are tailored to the specific goal of each stage, reflecting
the POGPN paper's logic[cite: 171].

-   **P1_obs (Flask) & P2_obs (Seed):** These are "transfer stages." Their
    latent output state (X, S, V) is the input to the next stage.
    We therefore observe the final, fundamental state variables that are
    physically transferred:
    1.  Xf_gL: Final biomass concentration (g/L)
    2.  Sf_gL: Final substrate concentration (g/L)
    3.  Vf_L: Final volume (L)

-   **P3_obs (Production):** This is the "production stage" and objective
    node. Its success is not defined by a state to be passed,
    but by its key performance indicators (KPIs). For P3 we expose the
    following observations to the DAG:
    1.  Pf_gL: Final Product Titer (concentration, g/L)
    2.  Vf_L: Final volume (L)

-   **Objective (Overall_Process_STY):** The objective is the Product STY
    from Stage 3, but *amortized over the total time of all three stages*.
    This correctly penalizes inefficient or overly long seed stages.
"""

from typing import Dict

import torch


class BioEthanolMultiStageFeatureExtractor:
    """Compute POGPN-style features and objective from 3-stage time-series data.

    Changes:
    - P3_obs now contains only [Pf_gL, Vf_L].
    - Space-Time Yield (P_STY_gL_hr) is moved to the overall Objective and is
      no longer part of P3_obs.
    """

    @staticmethod
    def _trapz(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Trapezoidal integration for time-series data."""
        return torch.trapz(y, t, dim=-1)

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is at least 2D for batch operations."""
        return x if x.ndim == 2 else x.unsqueeze(0)

    def _calculate_mass_balances(
        self,
        # xf: torch.Tensor,
        pf: torch.Tensor,
        # sf: torch.Tensor,
        vf: torch.Tensor,
        # x0: torch.Tensor,  # from sim_inputs
        pi: torch.Tensor,  # from series
        # s0: torch.Tensor,  # from sim_inputs
        v0: torch.Tensor,  # from sim_inputs
        # s_in: torch.Tensor,  # from sim_inputs
    ) -> Dict[str, torch.Tensor]:
        """Centralized calculation of mass balances."""
        # m_p: Total mass of product formed (g)
        m_p = torch.clamp(pf * vf - pi * v0, min=0.0)

        # fed_vol = torch.clamp(vf - v0, min=0.0)
        # m_s_in: Total mass of substrate added (initial + fed)
        # m_s_in = s0 * v0 + fed_vol * s_in
        # m_s_out: Total mass of substrate remaining
        # m_s_out = sf * vf
        # m_s_consumed: Total mass of substrate consumed
        # m_s_consumed = torch.clamp(m_s_in - m_s_out, min=1e-12)

        return {
            "m_p_g": m_p,
            # "m_s_consumed_g": m_s_consumed,
        }

    def transfer_state_observations(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the final state variables for a transfer stage (P1 or P2).

        This is the vector of "necessary things" that are passed to the
        next stage, representing the latent state f(w).

        Returns:
            Xf_gL (g/L): Final biomass concentration
            Sf_gL (g/L): Final substrate concentration
            Vf_L (L): Final volume
            Pf_gL (g/L): Final product concentration

        """
        x = self._ensure_2d(x)
        s = self._ensure_2d(s)
        v = self._ensure_2d(v)
        p = self._ensure_2d(p)

        return {
            "Xf_gL": x[..., -1].clamp_min(0.0),
            "Sf_gL": s[..., -1].clamp_min(0.0),
            "Vf_L": v[..., -1].clamp_min(0.0),
            "Pf_gL": p[..., -1].clamp_min(0.0),
        }

    def production_state_observations(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute product-centric KPIs for the production stage (P3).

        Metrics computed:
        - Pf_gL (g/L): Final Product Titer (concentration)
        - Vf_L (L): Final volume
        """
        p = self._ensure_2d(p)
        v = self._ensure_2d(v)

        # Final state values
        pf = p[..., -1].clamp_min(0.0)
        vf = v[..., -1].clamp_min(0.0)

        return {"Pf_gL": pf, "Vf_L": vf}

    def overall_objective(
        self,
        series_p3: Dict[str, torch.Tensor],
        production_state_observations: Dict[str, torch.Tensor],
        sim_inputs_p3: torch.Tensor,
    ) -> torch.Tensor:
        """Compute penalized STY (Space–Time Yield) objective with inoculum realism.

        The objective = STY × soft_penalty(inoculum_fraction),
        encouraging realistic inoculum (≈10%) while maintaining smoothness for BO.
        """
        # Extract final and initial volumes/products
        V_final = production_state_observations["Vf_L"]
        V_initial = series_p3["volume_L"][..., 0]
        P_final = production_state_observations["Pf_gL"]
        P_initial = series_p3["product_conc_gL"][..., 0]

        t_final = sim_inputs_p3[..., -1]
        v_avg = 0.5 * (V_final + V_initial)
        m_p = P_final * V_final - P_initial * V_initial  # net product mass

        # Standard STY definition
        p_sty = torch.where(v_avg > 0, m_p / (v_avg * t_final), torch.zeros_like(m_p))

        return p_sty

    def get_dag_tensors(
        self,
        sim_output_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Return tensors suitable as DAG observed nodes for POGPN.

        Extracts intermediate observations from P1 and P2, and
        final metrics from P3; computes the Objective as STY for P3.

        Returns (dict of tensors):
        - P1_obs: (batch, 3) = [Xf_1, Sf_1, Vf_1]
        - P2_obs: (batch, 3) = [Xf_2, Sf_2, Vf_2]
        - P3_obs: (batch, 2) = [Pf_3, Vf_3]
        - Objective: (batch, 1) = [P_STY_3]
        """
        # Unpack the series and simulation inputs from the dictionary
        series_p1 = sim_output_dict["series"]["P1"]
        series_p2 = sim_output_dict["series"]["P2"]
        series_p3 = sim_output_dict["series"]["P3"]

        sim_inputs_p3 = sim_output_dict["sim_inputs"]["P3"]

        # --- P1 Node (Transfer State Metrics) ---
        state_p1 = self.transfer_state_observations(
            series_p1["biomass_conc_gL"],
            series_p1["substrate_conc_gL"],
            series_p1["volume_L"],
            series_p1["product_conc_gL"],
        )
        p1_obs = torch.stack(
            [
                state_p1["Xf_gL"],
                state_p1["Pf_gL"],
            ],
            dim=-1,
        )

        # --- P2 Node (Transfer State Metrics) ---
        state_p2 = self.transfer_state_observations(
            series_p2["biomass_conc_gL"],
            series_p2["substrate_conc_gL"],
            series_p2["volume_L"],
            series_p2["product_conc_gL"],
        )
        p2_obs = torch.stack(
            [
                state_p2["Pf_gL"],
            ],
            dim=-1,
        )

        # --- P3 Node (Production KPI Metrics) ---
        prod_p3 = self.production_state_observations(
            series_p3["product_conc_gL"],
            series_p3["volume_L"],
        )
        p3_obs = prod_p3["Pf_gL"].unsqueeze(-1)

        # --- Objective Node (Overall Process STY) ---
        objective_score = self.overall_objective(
            series_p3=series_p3,
            production_state_observations=prod_p3,
            sim_inputs_p3=sim_inputs_p3,
        )
        objective = objective_score.unsqueeze(-1)

        return {
            "P1_obs": p1_obs,
            "P2_obs": p2_obs,
            "P3_obs": p3_obs,
            "Objective": objective,
        }
