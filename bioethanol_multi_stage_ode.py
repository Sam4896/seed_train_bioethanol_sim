# Copyright (c) 2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
# See LICENSE.md for license information.
"""Bioethanol multi-stage process simulator using SciPy ODE solver (batched output).

This module provides two ODE-based simulators:
1.  **SingleTankODESimulator**: Simulates a single, dynamic fed-batch process.
    It supports linear ramps for Temperature and pH.
2.  **BioEthanolMultiStageODESimulator**: Simulates a 3-stage "multi-stage" process,
    a common and citable industrial biotechnology workflow. This class uses
    the SingleTankODESimulator to simulate each stage (Flask, Seed, Production)
    and handles the logic of "inoculating" the next stage with the output
    of the previous one.

This enables a citable, high-dimensional (26-input) benchmark for
process-aware Bayesian Optimization (e.g., with POGPN).

State order and inputs follow the same conventions as the SDE module:
- States y = [X, P, S, V]
- Single Tank Inputs (12) = [X0, S0, V0, P0, S_in, T_initial, T_final,
                            pH_initial, pH_final, F0, mu_set, t_final]
- Seed Train Inputs (26) = [P1_Inputs (6), P2_Inputs (10), P3_Inputs (10)]
"""

from typing import Dict, Tuple

import numpy as np
import torch
from dataclasses import dataclass
from scipy.integrate import solve_ivp

"""Bioethanol Citable Kinetic Parameters

This module provides a data class containing the citable kinetic and
thermodynamic parameters for a standard bioethanol fermentation process,
typically modeling Saccharomyces cerevisiae (yeast).

This class separates the biological/chemical constants from the
simulation's differential equations, allowing for modularity.

Key Citable Models Used:
1.  **Growth/Substrate:** Andrews/Haldane-type model for substrate
    inhibition.
    [Ref: Andrews, J. F. (1968). Biotechnology and Bioengineering, 10(6)]
2.  **Product Formation:** Luedeking-Piret model (growth and non-growth
    associated).
    [Ref: Luedeking, R., & Piret, E. L. (1959). J. Biochemical and
    Microbiological Technology and Engineering, 1(4)]
3.  **Substrate Consumption:** Pirt model (growth, product, and
    maintenance).
    [Ref: Pirt, S. J. (1965). Proceedings of the Royal Society B, 163(991)]
4.  **Product Inhibition:** Ghose & Tyagi model for linear product inhibition.
    [Ref: Ghose, T. K., & Tyagi, R. D. (1979). Biotechnology and
    Bioengineering, 21(8)]
"""


@dataclass(frozen=True)
class BioEthanolKinetics:
    """Data class holding citable kinetic parameters for bioethanol.

    (frozen=True) makes this class immutable, preventing parameters
    from being accidentally changed during a simulation.
    """

    # --- Growth & Inhibition Parameters ---

    #: Max. specific growth rate (Ref: Monod)
    MU_MAX_REF: float = 0.35  # units: hr^-1

    #: Substrate affinity constant (Ref: Monod)
    KS_SUBSTRATE: float = 1.5  # units: g/L

    #: Substrate inhibition constant (Ref: Andrews/Haldane)
    KI_SUBSTRATE: float = 180.0  # units: g/L

    #: Max. product concentration causing total inhibition
    #: (Ref: Ghose & Tyagi)
    P_MAX_INHIBITION: float = 100.0  # units: g/L

    # --- Yield & Maintenance Parameters (Ref: Luedeking-Piret & Pirt) ---

    #: Yield of biomass from substrate (growth)
    Y_X_S: float = 0.1  # units: g-X / g-S

    #: Yield of product from substrate (growth)
    Y_P_S: float = 0.45  # units: g-P / g-S

    #: Growth-associated product formation (Alpha term)
    ALPHA_Y_P_X: float = 0.2  # units: g-P / g-X

    #: Non-growth-associated product formation (Beta term)
    BETA_M_P: float = 0.01  # units: g-P / (g-X * hr)

    #: Substrate maintenance coefficient
    M_S_MAINTENANCE: float = 0.01  # units: g-S / (g-X * hr)

    #: Endogenous decay/death rate
    KD_DEATH_RATE: float = 0.01  # units: hr^-1

    # --- Thermodynamic Parameters (Temperature & pH) ---

    #: Optimal temperature for growth
    T_OPTIMAL: float = 308.15  # units: K (35°C)

    #: Minimum temperature for growth
    T_MIN: float = 293.15  # units: K (20°C)

    #: Maximum temperature for growth
    T_MAX: float = 313.15  # units: K (40°C)

    #: Optimal pH for growth
    PH_OPTIMAL: float = 4.5  # units: pH

    #: Minimum pH for growth
    PH_MIN: float = 3.5  # units: pH

    #: Maximum pH for growth
    PH_MAX: float = 5.5  # units: pH


class SingleTankODESimulator:
    """Deterministic ODE simulator for a *single tank* with dynamic controls.

    This class simulates a single bio-reactor (batch or fed-batch) and
    has been upgraded from the original to support linear ramps for
    temperature and pH, which are common in real-world processes.

    Notes
    -----
    - Uses the same kinetics as the original file (BioEthanolKinetics).
    - Simulates a single trajectory based on 11 inputs.
    - This class is the "engine" used by the BioEthanolMultiStageODESimulator.
    """

    def __init__(self, v_max: float = 100.0) -> None:
        self.V_max_liters = v_max  # Use the passed-in v_max
        self.kin = BioEthanolKinetics()  # Citable kinetic parameters
        self.V_MIN_EPS = 1e-9

    def _get_feed_rate(self, t: float, V: float, F0: float, mu_set: float) -> float:
        """Exponential feed rate, capped at V_max."""
        F = F0 * np.exp(mu_set * t)
        return 0.0 if V >= self.V_max_liters else F

    def _linear_ramp(
        self, t: float, t_final: float, y_initial: float, y_final: float
    ) -> float:
        """Calculates the current value for a linear ramp."""
        if t_final <= 0.0 or t >= t_final:
            return y_final
        if t <= 0.0:
            return y_initial
        # Linear interpolation
        return y_initial + (y_final - y_initial) * (t / t_final)

    def _temperature_factor(self, T: float) -> float:
        """Citable temperature kinetics factor (from BioEthanolKinetics)."""
        T = max(self.kin.T_MIN, T)
        if not (self.kin.T_MIN < T < self.kin.T_MAX):
            return 0.0
        num = (T - self.kin.T_MAX) * (T - self.kin.T_MIN) ** 2
        den_1 = self.kin.T_OPTIMAL - self.kin.T_MIN
        den_2 = den_1 * (T - self.kin.T_OPTIMAL)
        den_3 = (self.kin.T_OPTIMAL - self.kin.T_MAX) * (
            self.kin.T_OPTIMAL + self.kin.T_MIN - 2.0 * T
        )
        val = num / (den_1 * (den_2 - den_3))
        return float(max(0.0, val))

    def _ph_factor(self, pH: float) -> float:
        """Citable pH kinetics factor (from BioEthanolKinetics)."""
        if pH <= self.kin.PH_MIN or pH >= self.kin.PH_MAX:
            return 0.0
        if pH < self.kin.PH_OPTIMAL:
            f = (
                1.0
                - ((pH - self.kin.PH_OPTIMAL) / (self.kin.PH_MIN - self.kin.PH_OPTIMAL))
                ** 2
            )
        else:
            f = (
                1.0
                - ((pH - self.kin.PH_OPTIMAL) / (self.kin.PH_MAX - self.kin.PH_OPTIMAL))
                ** 2
            )
        return float(max(0.0, f))

    def _ode_rhs(
        self, t: float, y: np.ndarray, control: Tuple[float, ...]
    ) -> np.ndarray:
        """Core ODE right-hand-side function with dynamic controls."""
        # Unpack state, ensuring no modifications to y
        X_state, P_state, S_state, V_state = y

        # Safe state values for rate calculations
        P_calc = max(0.0, P_state)
        S_calc = max(0.0, S_state)
        V_safe = max(self.V_MIN_EPS, V_state)

        # Unpack dynamic control vector
        (
            S_in,
            F0,
            mu_set,
            T_initial,
            T_final,
            pH_initial,
            pH_final,
            t_final,
        ) = control

        # --- Dynamic Control Ramps ---
        # Calculate current T and pH based on linear ramp
        T_current = self._linear_ramp(t, t_final, T_initial, T_final)
        pH_current = self._linear_ramp(t, t_final, pH_initial, pH_final)

        # --- Kinetic limitation factors (citable from BioEthanolKinetics) ---
        f_S = S_calc / (
            self.kin.KS_SUBSTRATE + S_calc + (S_calc**2) / self.kin.KI_SUBSTRATE
        )
        f_P = max(0.0, 1.0 - P_calc / self.kin.P_MAX_INHIBITION)
        f_T = self._temperature_factor(T_current)
        f_pH = self._ph_factor(pH_current)

        mu = self.kin.MU_MAX_REF * f_S * f_P * f_T * f_pH
        q_P = max(0.0, self.kin.ALPHA_Y_P_X * mu + self.kin.BETA_M_P)
        q_S = mu / self.kin.Y_X_S + q_P / self.kin.Y_P_S + self.kin.M_S_MAINTENANCE

        F_in = self._get_feed_rate(t, V_safe, F0, mu_set)

        # --- State Derivatives ---
        dX = (mu - self.kin.KD_DEATH_RATE) * X_state - (F_in / V_safe) * X_state
        dP = q_P * X_state - (F_in / V_safe) * P_state
        dS = (F_in / V_safe) * (S_in - S_state) - q_S * X_state
        dV = F_in

        # Add safety clamps to prevent states from becoming negative
        if y[0] + dX * 0.01 < 0:
            dX = 0.0
        if y[1] + dP * 0.01 < 0:
            dP = 0.0
        if y[2] + dS * 0.01 < 0:
            dS = 0.0

        return np.array([dX, dP, dS, dV], dtype=float)

    @torch.no_grad()
    def simulate(
        self, inputs: torch.Tensor, step_size: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """Run batched deterministic trajectories for a single tank.

        inputs: tensor of shape (batch, 12) ordered as
            [X0, S0, V0, P0, S_in, T_initial, T_final, pH_initial,
             pH_final, F0, mu_set, t_final].
        step_size: The time step for the uniform output grid.

        Returns a dict with time-series tensors of shape (batch, steps).
        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        batch = inputs.shape[0]
        device, dtype = inputs.device, inputs.dtype

        # Unpack all 12 inputs
        np_inputs = inputs.detach().cpu().numpy().astype(float)
        X0 = np_inputs[..., 0]
        S0 = np_inputs[..., 1]
        V0 = np_inputs[..., 2]
        P0 = np_inputs[..., 3]
        S_in = np_inputs[..., 4]
        T_initial = np_inputs[..., 5]
        T_final = np_inputs[..., 6]
        pH_initial = np_inputs[..., 7]
        pH_final = np_inputs[..., 8]
        F0 = np_inputs[..., 9]
        mu_set = np_inputs[..., 10]
        t_final = np_inputs[..., 11]

        # Number of steps per batch element
        steps = np.maximum(1, np.ceil(t_final / step_size).astype(int))
        max_steps = int(steps.max())

        T_all = np.zeros((batch, max_steps), dtype=float)
        X_all = np.zeros((batch, max_steps), dtype=float)
        P_all = np.zeros((batch, max_steps), dtype=float)
        S_all = np.zeros((batch, max_steps), dtype=float)
        V_all = np.zeros((batch, max_steps), dtype=float)

        for i in range(batch):
            n_i = int(steps[i])
            tf_i = float(t_final[i])
            if n_i <= 1 or tf_i <= 0.0:
                T_all[i, :] = 0.0
                X_all[i, :] = X0[i]
                P_all[i, :] = P0[i]
                S_all[i, :] = S0[i]
                V_all[i, :] = V0[i]
                continue

            t_eval = np.linspace(0.0, tf_i, num=n_i)
            y0 = np.array([X0[i], P0[i], S0[i], V0[i]], dtype=float)

            # Pack the 8 control parameters for the ODE RHS
            control = (
                S_in[i],
                F0[i],
                mu_set[i],
                T_initial[i],
                T_final[i],
                pH_initial[i],
                pH_final[i],
                tf_i,
            )

            # Hard volume ceiling: stop integration when V reaches V_max
            def volume_limit_event(t, y):
                return (1.0 * self.V_max_liters) - y[
                    3
                ]  # to avoid numerical issues and mimic equipment constraints

            volume_limit_event.terminal = True
            volume_limit_event.direction = -1.0

            sol = solve_ivp(
                fun=lambda t, y: self._ode_rhs(t, y, control),
                t_span=(0.0, tf_i),
                y0=y0,
                t_eval=t_eval,
                method="RK45",
                dense_output=False,
                events=volume_limit_event,
            )

            actual_len = sol.y.shape[1]
            y_padded = (
                np.pad(sol.y, ((0, 0), (0, n_i - actual_len)), "edge")
                if actual_len < n_i
                else sol.y
            )

            T_all[i, :n_i] = t_eval
            X_all[i, :n_i] = y_padded[0, :]
            P_all[i, :n_i] = y_padded[1, :]
            S_all[i, :n_i] = y_padded[2, :]
            V_all[i, :n_i] = y_padded[3, :]

            if n_i < max_steps:
                T_all[i, n_i:] = t_eval[-1]
                X_all[i, n_i:] = y_padded[0, -1]
                P_all[i, n_i:] = y_padded[1, -1]
                S_all[i, n_i:] = y_padded[2, -1]
                V_all[i, n_i:] = y_padded[3, -1]

        return {
            "time_hours": torch.tensor(
                T_all[:, :max_steps], device=device, dtype=dtype
            ),
            "biomass_conc_gL": torch.tensor(
                X_all[:, :max_steps], device=device, dtype=dtype
            ).clamp_min(0.0),
            "product_conc_gL": torch.tensor(
                P_all[:, :max_steps], device=device, dtype=dtype
            ).clamp_min(0.0),
            "substrate_conc_gL": torch.tensor(
                S_all[:, :max_steps], device=device, dtype=dtype
            ).clamp_min(0.0),
            "volume_L": torch.tensor(
                V_all[:, :max_steps], device=device, dtype=dtype
            ).clamp(min=self.V_MIN_EPS, max=self.V_max_liters),
        }


class BioEthanolMultiStageODESimulator:
    """Simulates a 3-stage bioethanol seed train (Flask -> Seed -> Production).

    This class orchestrates three sequential simulations using the
    `SingleTankODESimulator` to model a full, interconnected process network.
    This is the main simulator class to be called by the BO problem definition.

    Input: 26-dimensional tensor
    Output: Dictionary containing the time-series for all 3 stages,
            plus the 11-dim simulation inputs used for each (for feature extraction).
    """

    def __init__(
        self, v_max_stages: Tuple[float, float, float] = (2.0, 500.0, 20000.0)
    ) -> None:
        # Use stage-specific tank simulators with realistic capacity limits per stage
        self.v_max_stages = v_max_stages
        self.tank_sim_P1 = SingleTankODESimulator(v_max=v_max_stages[0])
        self.tank_sim_P2 = SingleTankODESimulator(v_max=v_max_stages[1])
        self.tank_sim_P3 = SingleTankODESimulator(v_max=v_max_stages[2])
        self.V_MIN_EPS = 1e-9

    @torch.no_grad()
    def _prepare_stage1_inputs(self, P1_controls: torch.Tensor) -> torch.Tensor:
        """Convert 6 batch controls -> 12 single-tank sim inputs."""
        X0_1 = P1_controls[..., 0]
        S0_1 = P1_controls[..., 1]
        V0_1 = P1_controls[..., 2]
        T_1 = P1_controls[..., 3]
        pH_1 = P1_controls[..., 4]
        t_final_1 = P1_controls[..., 5]

        # Stage 1 starts with 0 product
        P0_1 = torch.zeros_like(X0_1)

        # Stage 1 is BATCH: S_in, F0, mu_set are 0
        S_in_1 = torch.zeros_like(X0_1)
        F0_1 = torch.zeros_like(X0_1)
        mu_set_1 = torch.zeros_like(X0_1)

        # Stage 1 is STATIC: T_initial = T_final
        T_initial_1 = T_1
        T_final_1 = T_1
        pH_initial_1 = pH_1
        pH_final_1 = pH_1

        # Standard 12-dim order
        return torch.stack(
            [
                X0_1,
                S0_1,
                V0_1,
                P0_1,
                S_in_1,
                T_initial_1,
                T_final_1,
                pH_initial_1,
                pH_final_1,
                F0_1,
                mu_set_1,
                t_final_1,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def _prepare_stage2_inputs(
        self, P2_controls: torch.Tensor, series_P1: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert 10 fed-batch controls + P1 output -> 12 sim inputs.

        This function performs the inoculum mixing logic, now including product.
        """
        # Unpack 10 controls for Stage 2
        V0_2 = P2_controls[..., 0]
        S0_2 = P2_controls[..., 1]
        S_in_2 = P2_controls[..., 2]
        F0_2 = P2_controls[..., 3]
        mu_set_2 = P2_controls[..., 4]
        T_initial_2 = P2_controls[..., 5]
        T_final_2 = P2_controls[..., 6]
        pH_initial_2 = P2_controls[..., 7]
        pH_final_2 = P2_controls[..., 8]
        t_final_2 = P2_controls[..., 9]

        # Get final (latent) state from Stage 1 simulation
        Xf_P1 = series_P1["biomass_conc_gL"][..., -1]
        Pf_P1 = series_P1["product_conc_gL"][..., -1]
        Sf_P1 = series_P1["substrate_conc_gL"][..., -1]
        Vf_P1 = series_P1["volume_L"][..., -1]

        # --- Inoculum Mixing Logic ---
        # This is the core of the process network connection.
        # The new initial state is the weighted average of the inoculum
        # (output of P1) and the new medium (inputs of P2).
        V_total_P2 = V0_2 + Vf_P1
        V_total_P2 = torch.clamp(V_total_P2, min=self.V_MIN_EPS)

        # X0_P2: Biomass comes only from the inoculum (P1)
        X0_P2 = (Xf_P1 * Vf_P1) / V_total_P2
        # S0_P2: Substrate comes from inoculum and new medium
        S0_P2 = (Sf_P1 * Vf_P1 + S0_2 * V0_2) / V_total_P2
        # P0_P2: Product comes only from the inoculum (P1)
        P0_P2 = (Pf_P1 * Vf_P1) / V_total_P2
        # V0_P2: Initial volume is the combined volume
        V0_P2 = V_total_P2

        # Standard 12-dim order
        return torch.stack(
            [
                X0_P2,
                S0_P2,
                V0_P2,
                P0_P2,
                S_in_2,
                T_initial_2,
                T_final_2,
                pH_initial_2,
                pH_final_2,
                F0_2,
                mu_set_2,
                t_final_2,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def _prepare_stage3_inputs(
        self, P3_controls: torch.Tensor, series_P2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert 10 fed-batch controls + P2 output -> 12 sim inputs.

        Includes inoculum fraction realism (0.5–30%) for Stage 2 → Stage 3 transfer.
        """
        # --- Unpack 10 fed-batch controls for Stage 3 ---
        V0_3 = P3_controls[..., 0]
        S0_3 = P3_controls[..., 1]
        S_in_3 = P3_controls[..., 2]
        F0_3 = P3_controls[..., 3]
        mu_set_3 = P3_controls[..., 4]
        T_initial_3 = P3_controls[..., 5]
        T_final_3 = P3_controls[..., 6]
        pH_initial_3 = P3_controls[..., 7]
        pH_final_3 = P3_controls[..., 8]
        t_final_3 = P3_controls[..., 9]

        # --- Final (latent) state from Stage 2 simulation ---
        Xf_P2 = series_P2["biomass_conc_gL"][..., -1]
        Pf_P2 = series_P2["product_conc_gL"][..., -1]
        Sf_P2 = series_P2["substrate_conc_gL"][..., -1]
        Vf_P2 = series_P2["volume_L"][..., -1]

        # --- Inoculum Mixing Logic (P2 → P3) ---
        # Compute inoculum fraction and clip to realistic limits (0.5–30%)
        inoc_frac = Vf_P2 / torch.clamp(V0_3 + Vf_P2, min=self.V_MIN_EPS)
        inoc_frac = torch.clamp(inoc_frac, 0.005, 0.3)  # 0.5%–30%

        # Adjust combined volume accordingly
        V_total_P3 = torch.clamp(Vf_P2 / inoc_frac, min=self.V_MIN_EPS)

        # Compute initial conditions after mixing inoculum with new medium
        X0_P3 = (Xf_P2 * Vf_P2) / V_total_P3
        S0_P3 = (Sf_P2 * Vf_P2 + S0_3 * (V_total_P3 - Vf_P2)) / V_total_P3
        P0_P3 = (Pf_P2 * Vf_P2) / V_total_P3
        V0_P3 = V_total_P3

        # Standard 12-dim order expected by SingleTankODESimulator
        return torch.stack(
            [
                X0_P3,
                S0_P3,
                V0_P3,
                P0_P3,
                S_in_3,
                T_initial_3,
                T_final_3,
                pH_initial_3,
                pH_final_3,
                F0_3,
                mu_set_3,
                t_final_3,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def simulate(
        self, inputs: torch.Tensor, step_size: float = 0.01
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run the full 3-stage seed train simulation.

        inputs: tensor of shape (batch, 26) containing the concatenated
                controls for P1 (6), P2 (10), and P3 (10).
        step_size: The time step for the uniform output grid.

        Returns:
            A nested dictionary containing both the simulation time-series
            and the 12-dim inputs used for each stage's simulation.
            This facilitates feature extraction without duplicating logic.
            {
                "series": {"P1": series_P1, "P2": series_P2, "P3": series_P3},
                "sim_inputs": {"P1": sim_inputs_P1, "P2": sim_inputs_P2, "P3": sim_inputs_P3}
            }

        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        # --- Split 26-dim Input Vector ---
        # P1: Shake Flask (Batch, 6 inputs)
        P1_controls = inputs[..., 0:6]
        # P2: Seed Fermenter (Fed-Batch, 10 inputs)
        P2_controls = inputs[..., 6:16]
        # P3: Production Fermenter (Fed-Batch, 10 inputs)
        P3_controls = inputs[..., 16:26]

        # --- Stage 1: Shake Flask ---
        sim_inputs_P1 = self._prepare_stage1_inputs(P1_controls)
        series_P1 = self.tank_sim_P1.simulate(sim_inputs_P1, step_size=step_size)

        # --- Stage 2: Seed Fermenter ---
        # The true, latent output of P1 (series_P1) is used as an
        # input to prepare the simulation for P2.
        sim_inputs_P2 = self._prepare_stage2_inputs(P2_controls, series_P1)
        series_P2 = self.tank_sim_P2.simulate(sim_inputs_P2, step_size=step_size)

        # --- Stage 3: Production Fermenter ---
        # The true, latent output of P2 (series_P2) is used as an
        # input to prepare the simulation for P3.
        sim_inputs_P3 = self._prepare_stage3_inputs(P3_controls, series_P2)
        series_P3 = self.tank_sim_P3.simulate(sim_inputs_P3, step_size=step_size)

        # Return all time-series and the inputs used to generate them
        return {
            "series": {"P1": series_P1, "P2": series_P2, "P3": series_P3},
            "sim_inputs": {
                "P1": sim_inputs_P1,
                "P2": sim_inputs_P2,
                "P3": sim_inputs_P3,
            },
        }
