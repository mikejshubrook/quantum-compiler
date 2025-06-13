# Quantum Compiler

This project implements a quantum compilation algorithm that decomposes arbitrary two-qubit gates into the native gate sets of different quantum computing architectures.

# Project Overview
The current implementation is tailored for a trapped-ion quantum computer, using the Molmer–Sørensen (MS) gate as the two-qubit entangling gate and an XYX Euler basis for single-qubit rotations. If you're using a different architecture (e.g. superconducting qubits), you’ll need to adjust both the gate set and the decomposition strategy accordingly.

# Demo & Theory Explanation
For a walkthrough of the underlying theory and a demo of an older version of this project, check out this video on my YouTube channel: https://www.youtube.com/watch?v=NSC6NrH7Y-g

# Architecture-Specific Notes
The file trapped_ion_noise.py includes a noise model specifically for trapped-ion systems.
For other architectures, we recommend creating separate noise model files (e.g. superconducting_noise.py) and updating the compiler accordingly.

# Development Notes
This is an ongoing project, and parts of the codebase may include:
- TODO comments for future improvements
- Ideas and notes for extended features
- Work-in-progress features



