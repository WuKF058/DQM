# Privacy-Preserving Quantum Representation for Distributed Multi-Agent Reinforcement Learning in Coordinated Dispatch of Distribution Networks and Microgrids

[//]: # ([![Paper]&#40;https://img.shields.io/badge/Paper-IEEE--TSG-blue&#41;]&#40;#&#41; )
[![Python 3.10.8](https://img.shields.io/badge/python-3.10.8-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/WuKF058/DQM)

Official implementation of the paper **"Privacy-Preserving Quantum Representation for Distributed Multi-Agent Reinforcement Learning in Coordinated Dispatch of Distribution Networks and Microgrids"**.
---

## 📌 Problem Overview & Motivation

Coordinated economic dispatch between active distribution grids (DGs) and multiple microgrids (MGs) is essential for modern smart grid operation. However, deploying classical and quantum multi-agent reinforcement learning (MARL) in real-world energy networks introduces three coupled technical bottlenecks:

1. **High-Dimensional Operational States**: The high density of distributed energy resources (DERs), flexible loads, and battery energy storage systems creates exponentially expanding Hilbert and state spaces, making scalable state representation challenging for classical networks.
2. **Dual-Privacy Constraints**: Physical entities (DGs and MGs) belong to distinct operational domains and strictly forbid sharing raw operational data. Furthermore, standard Federated Learning (FL) requires sharing full model weights, exposing proprietary local network architectures.
3. **Quantum Optimization Bottlenecks**: Standard Variational Quantum Circuits (VQCs) suffer from severe **barren plateaus** (vanishing gradients) and prohibitive simulation/computational overhead during iterative updates in MARL.

To solve these challenges, this paper presents **DQM (Decoupled Quantum Representation Mechanism)** seamlessly integrated with a **Dual-Privacy Split MARL (SL)** architecture.

---

## 🔥 Key Contributions

- **Split MARL Architecture (SL)**:
  - Formulated under the Centralized Training with Decentralized Execution (CTDE) paradigm.
  - Vertically splits local Critic networks into local feature extractors and a central server aggregator.
  - Transmits perturbed intermediate representations (*smashed data*) to prevent raw operational data leakage and conceal heterogeneous local model architectures.
- **Decoupled Quantum Representation Mechanism (DQM)**:
  - Decouples quantum state representation from variational quantum circuit optimization.
  - Utilizes a parameter-free Quantum Kernel (`ZZFeatureMap`) to capture non-linear spatiotemporal dependencies in Hilbert space.
  - Introduces a Graph Laplacian Regularizer into the Critic loss, smoothing value function estimates without updating quantum parameters.
  - Reduces average training time by **78%** compared to VQC architectures while completely avoiding barren plateaus.
- **Nash Bargaining Team Reward**:
  - Incorporates a Nash-inspired smooth bargaining team reward function to guide agents toward Pareto-efficient coordination and prevent monopolistic pricing strategies.
- **Superconducting Quantum Hardware Deployment**:
  - Verified the quantum kernel evaluation subroutine on the **"Origin Wukong" 180-qubit superconducting quantum processor** under realistic NISQ conditions.

---

## 🛠️ Framework Architecture

```text
=========================================================================================
                           SPLIT MULTI-AGENT RL ARCHITECTURE
=========================================================================================

  [Client 1 (MG 1)] ──────► Local Hidden Layer ──► Smashed Data H1 ──┐
  [Client 2 (MG 2)] ──────► Local Hidden Layer ──► Smashed Data H2 ──┼──► [Server Aggregator Critic]
         ...                                                         │    ├── Aggregation Operator
  [Client N (DG)]   ──────► Local Hidden Layer ──► Smashed Data HN ──┘    ├── DQM Quantum Kernel Regularizer
                                                                          └── Value Output V(s)
                                                                                     │
  [Decentralized Classical Actors] ◄──────────── Estimated Advantage A(s,a) ─────────┘
=========================================================================================
```

---

## 📁 Repository Structure

```text
DQM/
├── README.md                     # Project documentation
├── environment.yml                # Conda environment configuration file
├── requirements.txt               # Pip dependency specifications
├── data/                          # IEEE 33-bus topology, load, PV, and grid pricing datasets
├── configs/                       # Configuration files for grid environments & training hyperparameters
├── src/                           # Core source code
│   ├── envs/                      # Power grid physical environments & AC power flow solvers
│   └── algorithms/                # Split-PPO, Split-A2C, and Split-SAC algorithms
├── baselines/                     # Baseline implementations (Classical MLP and VQC-based MARL)
├── experiments/                   # Experiment execution scripts & real QPU hardware evaluation
├── scripts/                       # Reproduction shell scripts
│   ├── reproduce_main_results.sh  # Script to reproduce main benchmark comparison results
│   ├── reproduce_ablation.sh      # Script to run ablation studies
│   └── reproduce_sensitivity.sh   # Script to evaluate parameter sensitivity
├── results/                       # Saved training logs, evaluation metrics, and figures
└── LICENSE                        # License agreement
```


## Contact
For technical support or data access requests, you can send emails:
- **Kaifeng Wu**  
  📧 wukaifeng@henu.edu.cn 
  🏛 Henan University
- **Hengji Li**  
  📧 lihengji@henu.edu.cn
  🏛 Henan University
