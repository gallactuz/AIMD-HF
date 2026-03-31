AIMD-HF
Educational Ab Initio Molecular Dynamics with Restricted Hartree–Fock / STO-3G

Anderson Aparecido do Espirito Santo¹
Marcos Henrique de Paula Dias da Silva²

¹ Independent Researcher, Araraquara – SP, Brazil
² Instituto Federal do Rio de Janeiro (IFRJ), Brazil

Abstract

This repository contains a fully self-contained, educational implementation of real-time Ab Initio Molecular Dynamics (AIMD) using the Restricted Hartree–Fock (RHF) method with the minimal STO-3G basis set.

The code strongly prioritizes conceptual clarity and pedagogical value over computational performance. Every fundamental step of quantum chemistry and molecular simulation is implemented explicitly and clearly commented in a single C++ file.

Introduction

AIMD-HF is an interactive educational simulator that couples a full quantum-mechanical electronic structure calculation (Restricted Hartree–Fock) with classical nuclear motion in real time.

Unlike conventional molecular dynamics packages that rely on pre-fitted force fields, this program computes the energy and forces on-the-fly by solving the electronic Schrödinger equation approximately via the Hartree–Fock method at every molecular dynamics step.

This allows users to directly visualize the intimate relationship between electronic structure and atomic dynamics.

Key Features
Complete Restricted Hartree–Fock (RHF) implementation with STO-3G basis set
Explicit evaluation of all one- and two-electron integrals:
Overlap
Kinetic energy
Nuclear attraction
Electron repulsion
Self-Consistent Field (SCF) procedure with adaptive damping and Jacobi diagonalization
Nuclear forces computed via central finite differences
Velocity Verlet integrator (BAOAB splitting)
Langevin thermostat (friction + stochastic noise)
Berendsen barostat for pressure control
Reflective (elastic) boundary conditions in a cubic simulation box
Real-time 3D visualization using OpenGL and GLUT
Dynamic chemical bond visualization (yellow cylinders)
Real-time rolling graph of Hartree–Fock energy
Heads-Up Display (HUD) with physical observables
Interactive Educational Panels (v1.1)
[O] Molecular Orbital Diagram
Energies, occupancy, spin arrows, HOMO/LUMO
[M] Mulliken Population Analysis
Gross population and atomic charges
[E] Energy Decomposition
T + V_ne + V_ee + V_nn with visual bars
[F] Force Visualization
Real-time quantum force vectors (red 3D arrows)
Architecture
Multi-threaded design (quantum calculations run in background thread)
Mutex-protected render buffers for smooth visualization
Compilation (Linux)

g++ aimd_hf.cpp -o aimd_hf -lGL -lGLU -lglut -lm -lpthread -O2

Execution

./aimd_hf [N]

N = number of hydrogen atoms
Must be even (closed-shell RHF)
Default: 2
Keyboard Controls
Key	Action
W / S	Zoom in / out
A / D	Move left / right
Z / X	Move up / down
/ - | Increase / decrease temperature (±50 K)
O | Toggle Molecular Orbital Diagram
M | Toggle Mulliken Analysis
E | Toggle Energy Decomposition
F | Toggle Force Arrows
ESC | Exit
Educational Value

This project was designed as a teaching tool for quantum chemistry and molecular simulation. Users can explore:

SCF convergence during real-time MD
Coupling between electronic structure and nuclear motion
Temperature effects on geometry and electronic properties
Mulliken charge fluctuations
HOMO–LUMO gap evolution
Real quantum forces (visualized in real time)
Energy decomposition contributions:
Kinetic
Nuclear attraction
Electron-electron repulsion
Nuclear repulsion
Effects of numerical approximations in quantum chemistry
Design Philosophy
Clarity over performance
Explicit implementation of all core algorithms
Clear comments for every major step:
Integrals
SCF
Forces
MD
Minimal dependencies (only OpenGL/GLUT)
Strong pedagogical focus
Ideal for understanding what happens "under the hood"
Limitations
O(N^4) scaling due to explicit computation of two-electron integrals
Numerical force calculation (6N + 1 SCF per step)
Practical limit: ~12 hydrogen atoms
STO-3G is a minimal basis set (qualitative results only)
Not suitable for production simulations
License

This project is released under the MIT License.

You are free to use, modify, and distribute this code (including commercially), provided that the original copyright notice is included.

Authors

Anderson Aparecido do Espirito Santo
Independent Researcher, Araraquara – SP, Brazil

Marcos Henrique de Paula Dias da Silva
Instituto Federal do Rio de Janeiro (IFRJ), Brazil
