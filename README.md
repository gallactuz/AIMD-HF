AIMD-HF: A Minimal, Transparent, and Interactive Educational Implementation of Hartree–Fock Ab Initio Molecular Dynamics

Authors:
Anderson Aparecido do Espirito Santo
Marcos Henrique de Paula Dias da Silva (IFRJ)

Abstract

This repository contains a fully self-contained, educational implementation of real-time Ab Initio Molecular Dynamics (AIMD) using the Restricted Hartree–Fock (RHF) method with the minimal STO-3G basis set.

The simulator supports multiple atomic systems, including hydrogen clusters (Hₙ), helium atoms (Heₙ), and protonated helium hydride systems (HeH⁺), enabling the exploration of both neutral and charged quantum systems.

The code strongly prioritizes conceptual clarity and pedagogical value over computational performance. Every fundamental step of quantum chemistry and molecular simulation is implemented explicitly and clearly commented in a single C++ file.

Introduction

AIMD-HF is an interactive educational simulator that couples a full quantum-mechanical electronic structure calculation (Restricted Hartree–Fock) with classical nuclear motion in real time.

Unlike conventional molecular dynamics packages that rely on pre-fitted force fields, this program computes the energy and forces on-the-fly by solving the electronic Schrödinger equation approximately via the Hartree–Fock method at every molecular dynamics step.

This allows users to directly visualize the relationship between electronic structure and atomic dynamics.

The inclusion of helium and HeH⁺ systems enables exploration of weak interactions, charge asymmetry, and thermally driven dissociation processes.

Key Features
Multi-system support:
Hydrogen clusters (Hₙ)
Helium atoms (Heₙ)
Protonated helium hydride systems (HeH⁺)
Complete Restricted Hartree–Fock (RHF) implementation with STO-3G basis set
Explicit evaluation of all one- and two-electron integrals:
Overlap (S)
Kinetic energy (T)
Nuclear attraction (Vₙₑ)
Electron repulsion integrals (ERI)
Self-Consistent Field (SCF) procedure with adaptive damping and Jacobi diagonalization
Nuclear forces computed via central finite differences
Velocity Verlet integrator
Langevin thermostat (friction + stochastic noise)
Berendsen barostat for pressure control
Reflective (elastic) boundary conditions in a cubic simulation box
Real-time 3D visualization using OpenGL and GLUT
Dynamic chemical bond visualization (yellow cylinders)
Real-time rolling graph of Hartree–Fock energy
Heads-Up Display (HUD) with physical observables
Interactive Educational Panels (no extra SCF cost):
[O] Molecular Orbitals (HOMO/LUMO)
[M] Mulliken Population Analysis
[E] Energy Decomposition (T + Vₙₑ + Vₑₑ + Vₙₙ)
[F] Quantum force vectors
Multi-threaded design (quantum calculations in background thread)
Mutex-protected rendering buffers
Installation and Compilation
Linux

Debian/Ubuntu:

sudo apt update
sudo apt install build-essential freeglut3-dev libglu1-mesa-dev

Fedora:

sudo dnf install gcc-c++ freeglut-devel mesa-libGLU-devel

Compile:

g++ aimd_hf.cpp -o aimd_hf -lGL -lGLU -lglut -lm -lpthread -O2

Run:

./aimd_hf
Windows (MSYS2 recommended)

Step 1 – Install MSYS2
https://www.msys2.org

Step 2 – Install dependencies:

pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-freeglut

Step 3 – Compile:

g++ aimd_hf.cpp -o aimd_hf.exe -lfreeglut -lopengl32 -lglu32 -O2

Step 4 – Run:

./aimd_hf.exe
Execution Parameters
./aimd_hf h
./aimd_hf h N
./aimd_hf he
./aimd_hf he N
./aimd_hf heh+
./aimd_hf heh+ N

Notes:

N = number of atoms or pairs
Default system is H₂
For hydrogen systems (Hₙ), N must be even (closed-shell RHF)
Charged systems (HeH⁺) are supported
Electron count is automatically adjusted
Keyboard Controls
Key	Action
W / S	Zoom
A / D	Move left/right
Z / X	Move up/down
+ / -	Temperature
O	Orbitals
M	Mulliken
E	Energy decomposition
F	Forces
ESC	Exit
Educational Value

This project was designed as a teaching tool for quantum chemistry and molecular simulation. Users can explore:

SCF convergence during real-time dynamics
Coupling between electronic structure and nuclear motion
Temperature effects on molecular behavior
Charge redistribution (Mulliken populations)
HOMO–LUMO evolution
Energy component contributions
Quantum mechanical forces
Supported Systems and Physics
Hₙ: covalent bonding and many-body effects
Heₙ: weak interactions (no true bonding in RHF/STO-3G)
HeH⁺: ionic bonding and charge asymmetry
Design Philosophy
Clarity over performance
Explicit implementation of all algorithms
Minimal dependencies
Strong pedagogical focus
Limitations
O(N⁴) scaling
Numerical forces (finite differences)
Limited to small systems (~12 atoms)
Minimal basis (STO-3G)
Only H and He supported
Supporting Information

Source code:
https://github.com/gallactuz/AIMD-HF
