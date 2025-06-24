<h1 align="center">Multi-Agent Reinforcement Learning: Resource Competition and Cooperation</h1>
<h2 align="center">Reinforcement Learning - Final Project</h2>
<p align="center">
  <b>Anastasia Chernavskaia, Moritz Peist, Nicolas Rauth</b><br>
  Barcelona School of Economics · 2025<br>
  DSDM Program, Barcelona School of Economics
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg">
</p>

![BSE Logo](/latex/imgs/BSE%20Barcelona%20Graduate%20School%20of%20Economics.svg)
---

## Project Overview

This project investigates **Multi-Agent Reinforcement Learning (MARL)** with a focus on resource competition and cooperation, developed as part of the BSE Reinforcement Learning course. We combine a comprehensive literature review of competitive multi-agent systems with practical implementation of PPO in cooperative environments.

**Project Type**: Literature Review + Code Replication  
**Focus Area**: Multi-Agent RL, Resource Competition, Cooperation  
**Key Reference**: [OpenAI's "Learning to cooperate, compete, and communicate"](https://openai.com/index/learning-to-cooperate-compete-and-communicate/)

![Training dashboard](/training_dashboard_with_coord_ma.gif)
*Note*: Multi-agent coordination dashboard. Clockwise from top-left: showing reward progression, agent behaviors, value loss, and coordination score during PPO training.

## Structure

```
├── latex/                      # Literature review document
│   ├── main.tex               # Main LaTeX document
│   ├── chapters/content.tex   # Literature review content
│   ├── appendix.tex          # Implementation appendix
│   └── references.bib        # Bibliography
├── marl_library/              # Custom MARL utilities
│   ├── __init__.py           # Library initialization
│   └── visualization.py      # Training visualization tools
├── project.toml              # Project dependencies
└── README.md                 # This file
```

## Research Focus

### Literature Review Topics

- **Multi-Agent Policy Gradient Methods** (MADDPG, COMA)
- **Value Function Factorization** (QMIX, QTRAN, QPLEX)
- **Cooperative vs Competitive Learning**
- **Resource Allocation in Multi-Agent Systems**
- **Communication and Coordination Mechanisms**

### Implementation

- **Environment**: Simple Spread (Multi-Agent Particle Environment)
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Paradigm**: Centralized Training, Decentralized Execution
- **Challenge**: Coordinate 3 agents to cover landmarks without collision

## Key Insights

**From Literature Review:**

- Multi-agent environments create natural curricula where difficulty scales with competitor skill
- No stable equilibrium exists - continuous pressure for improvement
- Centralized critics enable stable learning in non-stationary environments

**From Implementation:**

- PPO demonstrates surprising effectiveness in cooperative multi-agent settings
- Resource competition emerges even in cooperative tasks (spatial positioning)
- Coordination strategies develop through individual learning processes

## Installation

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip

### Setup

```bash
# Clone repository
git clone <repository-url>
cd rl-marl-project

# Install dependencies with UV
uv sync

# Or with pip, but uv is strongly preferred
pip install -e .
```

### Key Dependencies

- `torch>=2.3.0` - Deep learning framework
- `pettingzoo[mpe]>=1.25.0` - Multi-agent environments
- `stable-baselines3>=2.6.0` - RL algorithms
- `matplotlib` - Visualization
- `wandb>=0.20.1` - Experiment tracking

## Usage

### Running Experiments

```python
from marl_library.visualization import create_training_history_gif

# Create training visualization
create_training_history_gif()
```

## Results

### Training Metrics

The Simple Spread environment demonstrates key MARL challenges:

- **Episode Rewards**: Convergence from -50 to +20 over 30k timesteps
- **Coordination**: Emergent strategies for landmark coverage
- **Resource Competition**: Implicit spatial resource allocation

### Literature Findings

- **MADDPG** enables stable learning through centralized critics
- **Value factorization** methods (QMIX) handle credit assignment
- **PPO** surprisingly effective in cooperative multi-agent settings

## References

**Primary Reference:**
> Lowe, R., Mordatch, I., Abbeel, P., Wu, Y., Tamar, A., & Harb, J. (2017). Learning to cooperate, compete, and communicate. *OpenAI Blog*.

**Key Papers:**

- Foerster et al. (2018). Counterfactual Multi-Agent Policy Gradients (COMA)
- Rashid et al. (2020). Monotonic Value Function Factorisation (QMIX)
- Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
