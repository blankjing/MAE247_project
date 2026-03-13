# MAE247_project
**Simulation code for MAE247 project**

## Overview
This repository contains the simulation code and supplementary materials for the **MAE247** course project. The project focuses on developing and verifying a simplified simulation model inspired by the paper:

> *"Resiliency Through Collaboration in Heterogeneous Multi-Robot Systems"*

The primary goal of this project is to demonstrate how heterogeneous multi-robot systems can overcome individual environmental constraints (e.g., crossing a river) through pairwise capability sharing and collaboration. 

## Simulation Scenarios
The codebase includes both baseline scenarios and relatively complex multi-agent environments. To reflect the biological concepts introduced in the paper, the simulations specifically model the following interaction paradigms:

1. **Mutualism:** Two robots collaborate, and both benefit from the shared capabilities to reach their respective goals.
2. **Commensalism:** A highly capable robot (e.g., an amphibious "Big Turtle") assists another robot (e.g., a "Rabbit") without gaining or losing anything itself, as it is naturally immune to the environmental constraint.
3. **Mixed Collaboration:** A complex, dynamic multi-agent scenario that simultaneously incorporates both mutualism pairs and commensalism dispatching.

## File Structure
* `one_rabbit_one_turtle_simple_sim.py`: A brief and straightforward simulation demonstrating the core concept of pairwise capability sharing.
* `one_rabbit_one_turtle_complex_sim.py`: A relatively more complex 1-on-1 scenario (Mutualism) with dynamic state machines and stricter Control Barrier Function (CBF) constraints.
* `multi_robots_complex_sim.py`: The ultimate mixed scenario featuring 6 agents (multiple rabbits and turtles), demonstrating a dynamic dispatcher for commensalism alongside a mutualism pair.

## Dependencies
To run the simulations locally, please ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib casadi
