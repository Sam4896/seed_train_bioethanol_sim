# Multi-Stage Bioethanol Seed Train Simulation

A high-fidelity, 26-dimensional simulator for a 3-stage bioethanol seed train process (Flask → Seed Fermenter → Production), designed as a benchmark for process-aware Bayesian Optimization.

## Overview

This simulator models a realistic industrial biotechnology workflow consisting of three sequential fermentation stages:

1. **Stage 1 (P1) - Shake Flask**: Batch fermentation in a small-scale flask
2. **Stage 2 (P2) - Seed Fermenter**: Fed-batch fermentation for biomass expansion
3. **Stage 3 (P3) - Production Fermenter**: Fed-batch fermentation for product production

The simulator uses citable kinetic models based on established literature for bioethanol fermentation, typically modeling *Saccharomyces cerevisiae* (yeast).

## Features

- **26-dimensional input space** with realistic bounds for each process stage
- **ODE-based simulation** using SciPy's `solve_ivp` with RK45 method
- **Dynamic controls** including linear temperature and pH ramps
- **Process network modeling** with latent state transfer between stages
- **DAG structure** compatible with POGPN (Partially Observable Gaussian Process Network)
- **Observation noise support** for realistic experimental conditions
- **Batched evaluation** for efficient optimization

## Installation

### Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):
   ```bash
   # Linux/Mac
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Windows (PowerShell)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

2. **Install dependencies and create virtual environment**:
   ```bash
   poetry install
   ```

3. **Activate the Poetry shell**:
   ```bash
   poetry shell
   ```

4. **Run your scripts**:
   ```bash
   poetry run python your_script.py
   # OR (if in poetry shell):
   python your_script.py
   ```

### Using pip/venv

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch botorch==0.14.0 "numpy>=1.23.2,<2" scipy matplotlib networkx
   ```

## Requirements

- Python >= 3.10, < 3.13
- torch
- botorch == 0.14.0
- numpy >= 1.23.2, < 2.0.0
- scipy
- matplotlib
- networkx

## Usage

### Basic Example

```python
import torch
from bioethanol_multi_stage import BioEthanolMultiStageSeedTrain

# Initialize the simulator
simulator = BioEthanolMultiStageSeedTrain(
    step_size=0.01,
    observation_noise_std=0.05  # Optional: add 5% observation noise
)

# Create a batch of input points (batch_size, 26)
# Each row is a 26-dimensional input vector
inputs = torch.rand(5, 26)  # 5 random samples

# Evaluate the simulator
outputs = simulator(inputs)

# Access outputs
print(f"P1 observations: {outputs['P1_obs'].shape}")      # (5, 2)
print(f"P2 observations: {outputs['P2_obs'].shape}")      # (5, 1)
print(f"P3 observations: {outputs['P3_obs'].shape}")      # (5, 1)
print(f"Objective (STY): {outputs['Objective'].shape}")   # (5, 1)
```

### With Proper Input Bounds

```python
import torch
from bioethanol_multi_stage import BioEthanolMultiStageSeedTrain

simulator = BioEthanolMultiStageSeedTrain()

# Get Sobol samples within bounds
n_samples = 10
samples = simulator.get_sobol_samples(n_samples)

# Evaluate
outputs = simulator(samples)

# Get objective values
objective_values = outputs['Objective']
print(f"Objective values: {objective_values.squeeze()}")
```

## Input Structure

The simulator accepts a 26-dimensional input vector organized as follows:

### Stage 1 (P1) - Shake Flask [indices 0-5]
- `0`: X0_1 - Initial biomass concentration (g/L) [0.5, 10.0]
- `1`: S0_1 - Initial substrate concentration (g/L) [10.0, 80.0]
- `2`: V0_1 - Initial volume (L) [0.5, 2.0]
- `3`: T_1 - Static temperature (K) [298.15, 313.15]
- `4`: pH_1 - Static pH [3.5, 5.5]
- `5`: t_final_1 - Duration (hr) [6.0, 48.0]

### Stage 2 (P2) - Seed Fermenter [indices 6-15]
- `6`: V0_2 - Initial medium volume (L) [50.0, 200.0]
- `7`: S0_2 - Initial medium substrate (g/L) [10.0, 80.0]
- `8`: S_in_2 - Feed substrate concentration (g/L) [100.0, 600.0]
- `9`: F0_2 - Initial feed rate (L/hr) [0.05, 5.0]
- `10`: mu_set_2 - Feed rate parameter (hr^-1) [0.01, 0.25]
- `11`: T_initial_2 - Start temperature (K) [298.15, 313.15]
- `12`: T_final_2 - End temperature (K) [298.15, 313.15]
- `13`: pH_initial_2 - Start pH [3.5, 5.5]
- `14`: pH_final_2 - End pH [3.5, 5.5]
- `15`: t_final_2 - Duration (hr) [12.0, 96.0]

### Stage 3 (P3) - Production Fermenter [indices 16-25]
- `16`: V0_3 - Initial medium volume (L) [500.0, 5000.0]
- `17`: S0_3 - Initial medium substrate (g/L) [10.0, 80.0]
- `18`: S_in_3 - Feed substrate concentration (g/L) [100.0, 600.0]
- `19`: F0_3 - Initial feed rate (L/hr) [0.5, 50.0]
- `20`: mu_set_3 - Feed rate parameter (hr^-1) [0.01, 0.25]
- `21`: T_initial_3 - Start temperature (K) [298.15, 313.15]
- `22`: T_final_3 - End temperature (K) [298.15, 313.15]
- `23`: pH_initial_3 - Start pH [3.5, 5.5]
- `24`: pH_final_3 - End pH [3.5, 5.5]
- `25`: t_final_3 - Duration (hr) [48.0, 168.0]

## Output Structure

The simulator returns a dictionary with the following keys:

- **`P1_obs`**: Final state observations from Stage 1 (batch, 2)
  - Xf_gL: Final biomass concentration (g/L)
  - Pf_gL: Final product concentration (g/L)

- **`P2_obs`**: Final state observations from Stage 2 (batch, 1)
  - Pf_gL: Final product concentration (g/L)

- **`P3_obs`**: Final state observations from Stage 3 (batch, 1)
  - Pf_gL: Final product concentration (g/L)

- **`Objective`**: Space-Time Yield (STY) from Stage 3 (batch, 1)
  - Product STY (g/L/hr), amortized over total process time

- **`inputs`**: Original input tensor (batch, 26)

- **`P1_controls`**, **`P2_controls`**, **`P3_controls`**: Control inputs for each stage

## Process DAG Structure

The simulator models a directed acyclic graph (DAG) representing the process network:

```
Stage 1 (Shake Flask)
  P1_controls (6) → P1_obs (Xf, Pf)
                          ↓
Stage 2 (Seed Fermenter)
  P2_controls (10) + P1_obs → P2_obs (Pf)
                                    ↓
Stage 3 (Production)
  P3_controls (10) + P2_obs → P3_obs (Pf)
         ↓                          ↓
         |------------------→Objective (STY)
```

Each stage's output (latent state) is used as inoculum for the next stage, with proper mixing calculations for volume, biomass, substrate, and product concentrations.

## Kinetic Models

The simulator uses citable kinetic models from established literature:

1. **Growth/Substrate**: Andrews/Haldane-type model for substrate inhibition
   - Reference: Andrews, J. F. (1968). *Biotechnology and Bioengineering*, 10(6)

2. **Product Formation**: Luedeking-Piret model (growth and non-growth associated)
   - Reference: Luedeking, R., & Piret, E. L. (1959). *J. Biochemical and Microbiological Technology and Engineering*, 1(4)

3. **Substrate Consumption**: Pirt model (growth, product, and maintenance)
   - Reference: Pirt, S. J. (1965). *Proceedings of the Royal Society B*, 163(991)

4. **Product Inhibition**: Ghose & Tyagi model for linear product inhibition
   - Reference: Ghose, T. K., & Tyagi, R. D. (1979). *Biotechnology and Bioengineering*, 21(8)

## File Structure

- `bioethanol_multi_stage.py`: Main simulator class (`BioEthanolMultiStageSeedTrain`)
- `bioethanol_multi_stage_ode.py`: ODE-based simulation engine with kinetic models
- `bioethanol_feature_extractor.py`: Feature extraction and objective computation
- `dag_experiment_base.py`: Base class for DAG-structured experiments

## Notes

- The simulator uses realistic reactor capacity limits: 2 L (Flask), 500 L (Seed), 20,000 L (Production)
- Volume limits are enforced during simulation (integration stops when V_max is reached)
- Observation noise can be added proportionally to signal magnitude
- The objective (STY) is computed from Stage 3 but penalizes inefficient seed stages through time amortization
- Inoculum fractions for Stage 2 → Stage 3 are constrained to realistic ranges (0.5-30%)

## Implementation Note

**Important Notice:** The kinetic models and theoretical foundations referenced in this simulation are based on established research literature (see [Kinetic Models](#kinetic-models) section). However, the implementation of the simulation code itself was developed with the assistance of Large Language Models (LLMs).

While the underlying scientific principles and mathematical models are drawn from peer-reviewed publications, users should be aware that:

- **Code Review Recommended**: The implementation should be thoroughly reviewed, tested, and validated before use in production or research applications.
- **No Warranty**: The code is provided "as is" without warranty of any kind. Users are responsible for verifying the correctness, accuracy, and suitability of the implementation for their specific use case.
- **Validation Required**: The simulation results should be validated against experimental data or established benchmarks where possible.
- **Continuous Improvement**: The implementation may contain errors or inefficiencies that require manual correction or optimization.

Users are encouraged to report any issues, bugs, or concerns they encounter when using this codebase.

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE.md](LICENSE.md) for the full license text.

