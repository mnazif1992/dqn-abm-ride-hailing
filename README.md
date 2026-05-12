# DQN+ABM Ride-Hailing Dispatch Optimization

> Doctoral thesis implementation: A closed-loop hybrid architecture combining
> **Deep Q-Network (DQN)** with **Agent-Based Modeling (ABM)** for real-time
> driver-passenger assignment in online taxi platforms, validated on
> empirical data from Tehran, Iran.

## Overview

This repository contains the full implementation of a hybrid reinforcement
learning framework for ride-hailing dispatch optimization. The system couples
a Mesa-based agent-based simulation of urban mobility with a deep
reinforcement learning agent that learns to assign drivers to passengers
under realistic stochastic conditions.

### Key Components

- **Agent-Based Model (ABM):** Calibrated simulation of passengers, drivers,
  and a central dispatcher using Mesa 2.x
- **Deep Q-Network (DQN):** PyTorch implementation of value-based RL with
  experience replay and target networks
- **Baselines:** Random, Greedy (nearest-driver), and Hungarian assignment
  for comparative evaluation
- **Empirical Calibration:** Distributions derived from 104,770 real ride
  records (Tehran, Farvardin 1403)

## Project Structure

\`\`\`
dqn-abm-ride-hailing/
├── data/ # Raw, processed, and calibration data (gitignored)
├── src/
│ ├── abm/ # Mesa agents and model
│ ├── dqn/ # Neural network, replay buffer, training loop
│ ├── baselines/ # Comparison algorithms
│ └── utils/ # Helper functions
├── experiments/ # Configs, logs, and results
├── notebooks/ # Jupyter notebooks for EDA and analysis
├── figures/ # Generated plots
├── tests/ # Unit tests
└── docs/ # Additional documentation
\`\`\`

## Installation

\`\`\`bash

# Clone the repository

git clone https://github.com/USERNAME/dqn-abm-ride-hailing.git
cd dqn-abm-ride-hailing

# Create and activate virtual environment (Python 3.12 recommended)

python3 -m venv venv
source venv/bin/activate # On Windows: venv\\Scripts\\activate

# Install dependencies

pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## Data Availability

Raw ride data is **not** included in this repository due to:

- Confidentiality agreements with the data provider
- Ethical considerations regarding user privacy

Researchers interested in reproducing the results may:

- Request anonymized data through institutional channels
- Use synthetic data generators in \`src/utils/data_simulation.py\` (TBD)
- Adapt the framework to their own datasets

## Citation

If you use this code in your research, please cite:

\`\`\`bibtex
@phdthesis{nazif2026dqnabm,
author = {Nazif, Mohammadreza},
title = {A Closed-Loop DQN-ABM Framework for Ride-Hailing
Dispatch Optimization},
school = {[Tehran University]},
year = {2026},
type = {PhD Dissertation}
}
\`\`\`

## Reproducibility

See [\`docs/REPRODUCIBILITY.md\`](docs/REPRODUCIBILITY.md) for full
instructions on reproducing experiments (added in later phases).

## Author

**Mohammadreza Nazif**
PhD Candidate, [Department of Industrial Management], [Tehran University]
📧 nazif.mohammadreza@gmail.com

## License

This project is licensed under the MIT License — see the
[LICENSE](LICENSE) file for details.
