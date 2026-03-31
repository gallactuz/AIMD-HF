
// ============================================================
// Ab initio Molecular Dynamics (AIMD)
// Method: Hartree-Fock (RHF) / STO-3G basis set
// Real-time visualization with OpenGL
//
// Author  : Anderson Aparecido do Espirito Santo
// Version : 1.1 — Educational Panels
// Date    : 2026
//
// Description:
//   Ab initio molecular dynamics for hydrogen systems.
//   Full SCF Hartree-Fock for electronic energy and forces.
//   Educational panels extracted at zero extra SCF cost:
//     O → Molecular orbital diagram (energies + occupancy)
//     M → Mulliken population analysis (gross pop + net charge)
//     E → Energy decomposition (T + V_ne + V_ee + V_nn)
//     F → Red force arrows on each atom (3D)
//
// Compile:
//   Linux : g++ aimd_hf.cpp -o aimd_hf -lGL -lGLU -lglut -lm -lpthread
//
// Controls:
//   W/S : zoom in/out       A/D : camera left/right
//   Z/X : camera up/down    +/- : temperature ±50 K
//   O   : orbital diagram   M   : Mulliken populations
//   E   : energy decomp     F   : force arrows
//   ESC : exit
// ============================================================

#include <GL/glut.h>   // OpenGL Utility Toolkit: window, rendering loop, input events
#include <pthread.h>   // POSIX threads: physics runs in a background thread
#include <math.h>      // Math: sqrt, cos, sin, exp, log, erf, atan2, pow, fabs
#include <stdlib.h>    // Standard library: rand, srand, exit, atoi, malloc
#include <stdio.h>     // Standard I/O: printf, fprintf, sprintf
#include <string.h>    // Memory/string: memset, memcpy
#include <time.h>      // Time: time(NULL) seeds the random number generator

// ====================== SIMULATION PARAMETERS ======================
int NUM_ATOMS = 2;              // Number of hydrogen atoms; overridable from command line
#define dt            2.0f      // MD time step in atomic units (1 a.u. ≈ 24.19 attoseconds)
#define BOND_CUTOFF   1.6f      // Maximum interatomic distance (bohr) to draw a bond cylinder

#define MAX_ATOM      12                    // Maximum atoms the arrays can hold
#define NP            3                     // Primitives per contracted STO-3G basis function
#define MAX_BASIS     (MAX_ATOM * NP)       // Total primitive basis size: one contracted fn per atom
#define HISTORY_LEN   300                   // Length of the rolling HF energy history ring buffer

// ====================== CAMERA GLOBALS ======================
GLdouble obs_x = 0, obs_y = 0, obs_z = 7.0; // Camera (eye) position in 3D world coordinates (bohr)
static int   slices        = 20;             // Sphere tessellation: longitude subdivisions
static int   stacks        = 20;             // Sphere tessellation: latitude subdivisions
static float atom_radius   = 0.28f;          // Visual radius of each atom sphere (OpenGL units)
static float proton_mass   = 1836.0f;        // Proton mass in atomic units (electron mass = 1)
static float box_size      = 5.0f;           // Cubic simulation box side length (bohr)
static float T_target      = 300.0f;         // Target temperature for the Langevin thermostat (K)
static float target_pressure = 0.001f;       // Target pressure for the Berendsen barostat (a.u.)

// ====================== ATOM STRUCTURE ======================
struct Atom {
    float x,  y,  z;               // Current Cartesian position (bohr)
    float vx_old, vy_old, vz_old;  // Half-step velocity used by Velocity Verlet (bohr/a.u.)
    float vx,  vy,  vz;            // Full-step velocity (kept for reference; integration uses vx_old)
    float ax,  ay,  az;            // Acceleration = Force/mass (bohr/a.u.²), updated each MD step
    float mass, radius;            // Atomic mass (a.u.) and visual/collision radius
};

// ====================== PHYSICS-SIDE BUFFERS ======================
static struct Atom atoms[MAX_ATOM];      // Main atom array written by the physics thread
static double E_hf_cached = 0.0;        // Last converged SCF total energy (Ha), cached for logging
static float  T_inst      = 0.0f;       // Instantaneous kinetic temperature (K) from v²
static float  P_inst      = 0.0f;       // Instantaneous virial pressure (a.u.)
static float  min_bond    = 0.0f;       // Shortest interatomic distance found this step (bohr)
static double energy_history[HISTORY_LEN]; // Ring buffer of HF energies for the on-screen graph
static int    history_count = 0;        // Total entries ever written to the ring buffer
static int    md_step       = 0;        // MD step counter, incremented once per stepMD() call

// ====================== EDUCATIONAL DATA — PHYSICS SIDE ======================
// All quantities below are extracted from the already-converged SCF; zero extra cost.
static double g_orbital_eps[MAX_BASIS];           // Molecular orbital energies, sorted ascending (Ha)
static double g_mo_coeff[MAX_BASIS][MAX_BASIS];   // MO coefficient matrix C[AO index][MO index]
static double g_mulliken_q[MAX_ATOM];             // Mulliken net charge per atom: q = Z - gross_pop
static double g_E_kin = 0.0;   // Electronic kinetic energy: Tr(P * T_kin)      (Ha)
static double g_E_vne = 0.0;   // Nuclear attraction energy: Tr(P * V_ne)        (Ha)
static double g_E_vee = 0.0;   // Electron-electron repulsion: 1/2 Tr(P * G)     (Ha)
static double g_E_nuc = 0.0;   // Nuclear-nuclear repulsion (purely geometric)    (Ha)
static int    g_nb  = 2;       // Number of contracted basis functions = NUM_ATOMS
static int    g_occ = 1;       // Number of occupied MOs = NUM_ATOMS / 2

// ====================== RENDER-SIDE BUFFERS ======================
// These are thread-safe copies protected by render_mutex, read by the OpenGL thread.
static struct Atom atoms_r[MAX_ATOM];    // Snapshot of atom positions/velocities for rendering
static double E_r     = 0.0;            // HF energy copy displayed in the HUD
static float  T_r     = 0.0f;           // Instantaneous temperature copy for HUD
static float  P_r     = 0.0f;           // Instantaneous pressure copy for HUD
static float  bond_r  = 0.0f;           // Minimum bond distance copy for HUD
static int    step_r  = 0;              // MD step copy for HUD
static double ehist_r[HISTORY_LEN];     // Energy history copy for graph rendering
static int    hcount_r = 0;             // Energy history count copy for graph
static int    computing = 0;            // 1 while SCF is running; 0 when render buffer is fresh

// ====================== EDUCATIONAL DATA — RENDER SIDE ======================
static double orbital_eps_r[MAX_BASIS];           // Orbital energies copy for panel O
static double mo_coeff_r[MAX_BASIS][MAX_BASIS];   // MO coefficients copy for panel O
static double mulliken_q_r[MAX_ATOM];             // Mulliken charges copy for panel M
static double E_kin_r = 0.0, E_vne_r = 0.0;      // Energy component copies for panel E
static double E_vee_r = 0.0, E_nuc_r = 0.0;      // Energy component copies for panel E
static int    nb_r = 2, occ_r = 1;               // Basis size and occupancy copies for panel O

// ====================== PANEL TOGGLE FLAGS ======================
static int show_orbital    = 0;  // 1 → display orbital diagram panel (key O)
static int show_mulliken   = 0;  // 1 → display Mulliken population panel (key M)
static int show_energy_dec = 0;  // 1 → display energy decomposition panel (key E)
static int show_forces     = 0;  // 1 → draw red force arrows in 3D view (key F)

// Mutex protecting all render-side buffers from simultaneous read/write
static pthread_mutex_t render_mutex = PTHREAD_MUTEX_INITIALIZER;

// ====================== BOX-MULLER NORMAL RANDOM GENERATOR ======================
double gauss_rand() {                              // Returns a sample from N(0,1) via Box-Muller transform
    static int   has_spare = 0;                    // 1 if a spare value was saved from the previous call
    static double spare;                           // Spare normal variate stored from last Box-Muller pair
    if (has_spare) { has_spare = 0; return spare; } // Return cached spare and clear the flag
    has_spare = 1;                                 // Mark that we will produce a spare this call
    double u, v, s;                                // Two uniform samples and their squared norm
    do {
        u = (rand() / (double)RAND_MAX) * 2.0 - 1.0; // Uniform sample in [-1, 1]
        v = (rand() / (double)RAND_MAX) * 2.0 - 1.0; // Uniform sample in [-1, 1]
        s = u * u + v * v;                         // Sum of squares; must lie strictly inside unit disk
    } while (s >= 1.0 || s == 0.0);               // Reject outside disk or origin (prevents log(0))
    s = sqrt(-2.0 * log(s) / s);                  // Box-Muller scale factor: sqrt(-2 ln s / s)
    spare = v * s;                                 // Save second standard normal for next call
    return u * s;                                  // Return first standard normal
}

// ====================== HF INTEGRAL HELPERS ======================
double F0(double t) {                              // Boys function F0(t) = (1/2)sqrt(pi/t) erf(sqrt(t))
    if (t < 1e-12) return 1.0;                    // Limiting value F0(0) = 1 (avoids 0/0 at coincident centers)
    return 0.5 * sqrt(M_PI / t) * erf(sqrt(t));  // Closed-form Boys F0 for s-type integrals
}

double norm_gauss(double alpha) {                  // Normalization constant for a 1s Gaussian: (2α/π)^(3/4)
    return pow(2.0 * alpha / M_PI, 0.75);         // Ensures <φ|φ> = 1 for a single primitive Gaussian
}

double dist2(const double A[3], const double B[3]) { // Squared Euclidean distance |A-B|² in 3D (no sqrt)
    double dx = A[0]-B[0], dy = A[1]-B[1], dz = A[2]-B[2]; // Component differences
    return dx*dx + dy*dy + dz*dz;                 // Sum of squares: cheaper than computing |A-B|
}

float atomDistance(struct Atom a, struct Atom b) { // Euclidean distance between two atoms (float precision)
    return sqrtf(                                  // Square root of sum of squared coordinate differences
        (a.x-b.x)*(a.x-b.x) +                    // x-component squared
        (a.y-b.y)*(a.y-b.y) +                    // y-component squared
        (a.z-b.z)*(a.z-b.z)                       // z-component squared
    );
}

// ====================== JACOBI EIGENVALUE DIAGONALIZATION ======================
void jacobi(int n,                                 // Matrix dimension (number of basis functions)
            double A[MAX_BASIS][MAX_BASIS],         // Input: real symmetric matrix; overwritten during algorithm
            double V[MAX_BASIS][MAX_BASIS],         // Output: eigenvector matrix (columns are eigenvectors)
            double eig[MAX_BASIS]) {                // Output: eigenvalue array (diagonal of diagonalized A)
    memset(V, 0, sizeof(double)*MAX_BASIS*MAX_BASIS); // Initialize V to zero before setting identity
    for (int i = 0; i < n; i++) V[i][i] = 1.0;   // V starts as identity matrix I (eigenvectors = standard basis)
    for (int iter = 0; iter < 200; iter++) {       // Up to 200 Jacobi rotation sweeps through the matrix
        int p = 0, q = 1;                          // Indices of the largest off-diagonal element (start at [0][1])
        double maxval = fabs(A[p][q]);             // Track the maximum absolute off-diagonal value found so far
        for (int i = 0; i < n; i++)               // Search all rows
            for (int j = i+1; j < n; j++)         // Only upper triangle (symmetric matrix)
                if (fabs(A[i][j]) > maxval) {      // If this element is larger than current max
                    maxval = fabs(A[i][j]);         // Update maximum
                    p = i; q = j;                  // Record location of new maximum
                }
        if (maxval < 1e-10) break;                 // Convergence: all off-diagonal elements are negligible
        double theta = 0.5 * atan2(2.0*A[p][q], A[q][q]-A[p][p]); // Rotation angle to zero out A[p][q]
        double c = cos(theta), s = sin(theta);     // Cosine and sine of the Jacobi rotation angle
        double App = c*c*A[p][p] - 2*s*c*A[p][q] + s*s*A[q][q]; // Updated diagonal element A[p][p]
        double Aqq = s*s*A[p][p] + 2*s*c*A[p][q] + c*c*A[q][q]; // Updated diagonal element A[q][q]
        A[p][p] = App; A[q][q] = Aqq; A[p][q] = A[q][p] = 0.0;  // Apply rotation: zero out [p][q] pivot
        for (int k = 0; k < n; k++) {             // Update remaining off-diagonal elements affected by rotation
            if (k == p || k == q) continue;        // Skip pivot rows (already updated above)
            double Akp = c*A[k][p] - s*A[k][q];  // Rotated element in row k, column p
            double Akq = s*A[k][p] + c*A[k][q];  // Rotated element in row k, column q
            A[k][p] = A[p][k] = Akp;             // Apply symmetrically to both sides
            A[k][q] = A[q][k] = Akq;             // Apply symmetrically to both sides
        }
        for (int k = 0; k < n; k++) {             // Accumulate Givens rotation into eigenvector matrix V
            double vkp = c*V[k][p] - s*V[k][q];  // Rotate column p of V
            double vkq = s*V[k][p] + c*V[k][q];  // Rotate column q of V
            V[k][p] = vkp; V[k][q] = vkq;        // Store updated eigenvector components
        }
    }
    for (int i = 0; i < n; i++) eig[i] = A[i][i]; // Extract final eigenvalues from the now-diagonal A
}

// ====================== HARTREE-FOCK SCF ENERGY ======================
// Computes the RHF total energy for NB hydrogen atoms at positions R_atom_in.
// If do_analysis == 1, also extracts orbital energies, Mulliken charges, and
// energy components — all from data already computed during SCF, at zero extra cost.
double compute_hf_energy(const double R_atom_in[MAX_ATOM][3], // Nuclear positions (bohr)
                         int NB,                              // Number of atoms (= number of basis functions)
                         int do_analysis) {                   // 1 = extract educational data; 0 = skip
    double R_atom[MAX_ATOM][3];                               // Local working copy of nuclear positions
    for (int i = 0; i < NB; i++)                              // Loop over atoms
        for (int k = 0; k < 3; k++)                           // Loop over x, y, z
            R_atom[i][k] = R_atom_in[i][k];                  // Copy positions into local array

    // STO-3G contraction coefficients and exponents for the H 1s orbital
    double d_coef[NP] = {0.15432897, 0.53532814, 0.44463454}; // STO-3G contraction coefficients
    double a_exp[NP]  = {3.42525091, 0.62391373, 0.16885540}; // STO-3G Gaussian exponents (bohr⁻²)

    double a_prim[MAX_BASIS];       // Exponent of each primitive Gaussian (indexed by atom*NP + primitive)
    double R_prim[MAX_BASIS][3];    // Center of each primitive Gaussian (placed on its parent nucleus)
    for (int A = 0; A < NB; A++)   // Loop over atoms
        for (int p = 0; p < NP; p++) { // Loop over primitives for atom A
            int idx = A*NP + p;         // Global primitive index
            a_prim[idx] = a_exp[p];     // Assign exponent from STO-3G table
            for (int k = 0; k < 3; k++) R_prim[idx][k] = R_atom[A][k]; // Center on nucleus A
        }

    // ---- One-electron integral matrices ----
    double S[MAX_BASIS][MAX_BASIS]     = {{0}};  // Overlap matrix S[μ][ν] = <φ_μ|φ_ν>
    double T_kin[MAX_BASIS][MAX_BASIS] = {{0}};  // Kinetic energy matrix T[μ][ν] = <φ_μ|-½∇²|φ_ν>
    double Vne[MAX_BASIS][MAX_BASIS]   = {{0}};  // Nuclear attraction matrix V[μ][ν] = <φ_μ|Σ_A -Z_A/r_A|φ_ν>
    double Hcore[MAX_BASIS][MAX_BASIS];          // Core Hamiltonian: H = T + V_ne (one-electron part)

    for (int i = 0; i < NB; i++)               // Row index over contracted basis functions
        for (int j = i; j < NB; j++) {         // Column index (upper triangle; exploits Hermitian symmetry)
            for (int p = 0; p < NP; p++)        // Primitive index for function i
                for (int q = 0; q < NP; q++) {  // Primitive index for function j
                    int ip = i*NP+p, jq = j*NP+q;           // Global primitive indices
                    double al1 = a_prim[ip], al2 = a_prim[jq]; // Exponents of the two primitives
                    double N1 = norm_gauss(al1), N2 = norm_gauss(al2); // Normalization constants
                    double gam   = al1 + al2;               // γ = α₁ + α₂ (sum of exponents)
                    double ratio = al1 * al2 / gam;         // α₁α₂/γ (product-over-sum ratio)
                    double Rab2  = dist2(R_prim[ip], R_prim[jq]); // |R_i - R_j|² (primitive centers)
                    double Rp[3];                            // Gaussian product center P = (α₁R₁+α₂R₂)/γ
                    for (int k = 0; k < 3; k++)              // Compute each component of product center
                        Rp[k] = (al1*R_prim[ip][k] + al2*R_prim[jq][k]) / gam;
                    double Sij = pow(M_PI/gam, 1.5) * exp(-ratio*Rab2); // Overlap integral of two s-Gaussians
                    double Tij = ratio * (3.0 - 2.0*ratio*Rab2) * Sij;  // Kinetic energy integral
                    double Vij = 0.0;                        // Accumulator for nuclear attraction from all nuclei
                    for (int A2 = 0; A2 < NB; A2++) {        // Sum over all nuclei (Z=1 for hydrogen)
                        double rPA2 = dist2(Rp, R_atom[A2]); // |P - R_A|² distance from product center to nucleus
                        Vij -= (2.0*M_PI/gam) * exp(-ratio*Rab2) * F0(gam*rPA2); // V integral via Boys F0
                    }
                    double cc = d_coef[p] * d_coef[q] * N1 * N2; // Combined contraction weight
                    S[i][j]     += cc * Sij;   // Accumulate contracted overlap
                    T_kin[i][j] += cc * Tij;   // Accumulate contracted kinetic energy
                    Vne[i][j]   += cc * Vij;   // Accumulate contracted nuclear attraction
                }
            S[j][i]     = S[i][j];     // Exploit Hermitian symmetry to fill lower triangle
            T_kin[j][i] = T_kin[i][j]; // Exploit Hermitian symmetry
            Vne[j][i]   = Vne[i][j];   // Exploit Hermitian symmetry
        }

    for (int i = 0; i < NB; i++)    // Build core Hamiltonian row by row
        for (int j = 0; j < NB; j++) // Column by column
            Hcore[i][j] = T_kin[i][j] + Vne[i][j]; // H_core = T_kin + V_ne (no e-e repulsion yet)

    // ---- Nuclear repulsion energy (purely geometric) ----
    double E_nuc = 0.0;                          // Initialize nuclear repulsion accumulator
    for (int A = 0; A < NB; A++)                 // Outer loop over atom pairs (A < B)
        for (int B = A+1; B < NB; B++)           // Avoid double counting
            E_nuc += 1.0 / sqrt(dist2(R_atom[A], R_atom[B])); // E_nuc += Z_A Z_B / R_AB (Z=1 for H)

    // ---- Two-electron repulsion integrals (ERI): (ij|kl) ----
    static double eri[MAX_BASIS][MAX_BASIS][MAX_BASIS][MAX_BASIS]; // 4-index ERI tensor: (ij|kl)
    memset(eri, 0, sizeof(eri));                 // Zero all ERI values before computing
    for (int i = 0; i < NB; i++) for (int j = 0; j < NB; j++)  // Loop over all four basis indices
    for (int k = 0; k < NB; k++) for (int l = 0; l < NB; l++) {
        double val = 0.0;                        // Contracted ERI accumulator for index set (i,j,k,l)
        for (int p = 0; p < NP; p++) for (int q = 0; q < NP; q++) // Primitives for functions i and j
        for (int r = 0; r < NP; r++) for (int s = 0; s < NP; s++) { // Primitives for functions k and l
            int ip=i*NP+p, jq=j*NP+q, kr=k*NP+r, ls=l*NP+s; // Global primitive indices
            double a1=a_prim[ip], a2=a_prim[jq]; // Exponents for electron-1 pair
            double a3=a_prim[kr], a4=a_prim[ls]; // Exponents for electron-2 pair
            double N1=norm_gauss(a1), N2=norm_gauss(a2); // Normalizations for electron-1 primitives
            double N3=norm_gauss(a3), N4=norm_gauss(a4); // Normalizations for electron-2 primitives
            double g1=a1+a3, g2=a2+a4, g=g1+g2;  // γ₁, γ₂, and combined exponent γ=γ₁+γ₂
            double Rp2[3], Rq2[3];                // Product centers for electron-1 (Rp2) and electron-2 (Rq2)
            for (int dd = 0; dd < 3; dd++) {      // Compute product centers component by component
                Rp2[dd] = (a1*R_prim[ip][dd] + a3*R_prim[kr][dd]) / g1; // Electron-1 product center
                Rq2[dd] = (a2*R_prim[jq][dd] + a4*R_prim[ls][dd]) / g2; // Electron-2 product center
            }
            double rab2 = dist2(R_prim[ip], R_prim[kr]); // |R_i - R_k|² for electron-1 Gaussian pair
            double rcd2 = dist2(R_prim[jq], R_prim[ls]); // |R_j - R_l|² for electron-2 Gaussian pair
            double rpq2 = dist2(Rp2, Rq2);               // |Rp - Rq|² distance between product centers
            double pref  = 2.0*pow(M_PI, 2.5) / (g1*g2*sqrt(g)); // ERI prefactor: 2π^(5/2)/(γ₁γ₂√γ)
            double boys_arg = (g1*g2/g) * rpq2;           // Boys function argument: (γ₁γ₂/γ)|Rp-Rq|²
            val += d_coef[p]*d_coef[q]*d_coef[r]*d_coef[s] * N1*N2*N3*N4 * pref  // Weight × norms × prefactor
                 * exp(-a1*a3/g1*rab2 - a2*a4/g2*rcd2)    // Gaussian decay for both electron pairs
                 * F0(boys_arg);                           // Boys function for Coulomb integral
        }
        eri[i][j][k][l] = val; // Store contracted ERI for index combination (i,j,k,l)
    }

    // ====================== LÖWDIN ORTHOGONALIZATION: build X = S^{-1/2} ======================
    double Scopy[MAX_BASIS][MAX_BASIS]; // Working copy of S (Jacobi overwrites its input)
    double U[MAX_BASIS][MAX_BASIS];     // Eigenvectors of S: columns of U
    double eigS[MAX_BASIS];            // Eigenvalues of S: s_k
    memcpy(Scopy, S, sizeof(S));       // Copy overlap matrix before diagonalization
    jacobi(NB, Scopy, U, eigS);        // Diagonalize S = U diag(eigS) U^T

    double X[MAX_BASIS][MAX_BASIS] = {{0}}; // Löwdin orthogonalizer X = S^{-1/2} = U diag(1/√s_k) U^T
    for (int i = 0; i < NB; i++)            // Row index
        for (int j = 0; j < NB; j++)        // Column index
            for (int k = 0; k < NB; k++)    // Sum over eigenvectors
                if (eigS[k] > 1e-10)         // Skip near-zero eigenvalues to avoid numerical divergence
                    X[i][j] += U[i][k] * (1.0/sqrt(eigS[k])) * U[j][k]; // X[i][j] = Σ_k U[i][k]/√s_k U[j][k]

    // ====================== SCF ITERATIVE PROCEDURE ======================
    double P[MAX_BASIS][MAX_BASIS] = {{0}}; // Density matrix P[μ][ν]; initialized to zero for start
    double E_old  = 0.0;   // Total energy from previous SCF iteration (convergence check)
    double damp   = 0.6;   // Density matrix mixing parameter: large initially for stability
    int    occ    = NB/2;  // Number of occupied MOs: each H has 1 electron, 2 per orbital (RHF)
    int    max_iter = 120; // Maximum SCF iterations before declaring non-convergence
    int    iter;           // Current SCF iteration index
    double Etot   = 0.0;  // Total HF energy (electronic + nuclear), updated each SCF iteration

    // Storage for the last computed (converged or best) SCF quantities — used for analysis
    double C_final[MAX_BASIS][MAX_BASIS]  = {{0}}; // Final MO coefficient matrix C = X × C' (AO basis)
    double eps_final[MAX_BASIS]           = {0};   // Final sorted orbital energies (Ha)
    double P_final[MAX_BASIS][MAX_BASIS]  = {{0}}; // Final converged density matrix
    double G_final[MAX_BASIS][MAX_BASIS]  = {{0}}; // Final two-electron Fock contribution G

    for (iter = 0; iter < max_iter; iter++) { // Main SCF loop: iterate to self-consistency

        // ---- Build two-electron Fock contribution G ----
        double G[MAX_BASIS][MAX_BASIS] = {{0}}; // G[i][j] = Σ_pq P[p][q][(ij|qp) - 1/2(ip|qj)]
        for (int i = 0; i < NB; i++)            // Fock row index
            for (int j = 0; j < NB; j++)        // Fock column index
                for (int p2 = 0; p2 < NB; p2++) // Density matrix row index
                    for (int q = 0; q < NB; q++) // Density matrix column index
                        G[i][j] += P[p2][q] * (eri[i][q][j][p2] - 0.5*eri[i][q][p2][j]); // Coulomb - 1/2 Exchange

        // ---- Build Fock matrix F = H_core + G ----
        double F_mat[MAX_BASIS][MAX_BASIS];      // Fock matrix in AO basis
        for (int i = 0; i < NB; i++)            // Row
            for (int j = 0; j < NB; j++)        // Column
                F_mat[i][j] = Hcore[i][j] + G[i][j]; // F = H_core + G (mean-field electron repulsion added)

        // ---- Transform Fock matrix to orthogonal basis: F' = X^T F X ----
        double Fp[MAX_BASIS][MAX_BASIS] = {{0}}; // Transformed Fock matrix F' in Löwdin basis
        for (int i = 0; i < NB; i++)             // Row of F'
            for (int j = 0; j < NB; j++)         // Column of F'
                for (int k = 0; k < NB; k++)     // Inner sum index k
                    for (int l = 0; l < NB; l++) // Inner sum index l
                        Fp[i][j] += X[k][i] * F_mat[k][l] * X[l][j]; // F'[i][j] = Σ_{kl} X[k][i] F[k][l] X[l][j]

        // ---- Diagonalize F' to get orbital energies eps and coefficients C' ----
        double Acopy[MAX_BASIS][MAX_BASIS]; // Working copy of F' for Jacobi (overwrites input)
        double Cp[MAX_BASIS][MAX_BASIS];    // Eigenvectors of F' in orthogonal basis
        double eps[MAX_BASIS];             // Orbital energies (eigenvalues of F')
        memcpy(Acopy, Fp, sizeof(Fp));     // Copy F' before diagonalization
        jacobi(NB, Acopy, Cp, eps);        // Diagonalize: F' = Cp diag(eps) Cp^T

        // ---- Sort orbital energies and eigenvectors in ascending order ----
        for (int i = 0; i < NB-1; i++)              // Outer bubble sort pass
            for (int jj = i+1; jj < NB; jj++)       // Inner comparison
                if (eps[jj] < eps[i]) {              // If orbital jj has lower energy, swap
                    double tmp = eps[i]; eps[i] = eps[jj]; eps[jj] = tmp; // Swap eigenvalues
                    for (int k = 0; k < NB; k++) {   // Swap corresponding eigenvector columns
                        tmp = Cp[k][i]; Cp[k][i] = Cp[k][jj]; Cp[k][jj] = tmp;
                    }
                }

        // ---- Back-transform MO coefficients to AO basis: C = X × C' ----
        double C[MAX_BASIS][MAX_BASIS] = {{0}}; // MO coefficients in original AO basis
        for (int i = 0; i < NB; i++)            // AO index μ
            for (int j = 0; j < NB; j++)        // MO index a
                for (int k = 0; k < NB; k++)    // Orthogonal basis index k
                    C[i][j] += X[i][k] * Cp[k][j]; // C[μ][a] = Σ_k X[μ][k] C'[k][a]

        // ---- Build new density matrix from occupied MOs ----
        double Pnew[MAX_BASIS][MAX_BASIS] = {{0}}; // New density matrix P[μ][ν] = 2 Σ_occ C[μ][a] C[ν][a]
        for (int i = 0; i < NB; i++)               // AO index μ
            for (int j = 0; j < NB; j++)           // AO index ν
                for (int k = 0; k < occ; k++)      // Sum only over occupied orbitals
                    Pnew[i][j] += 2.0 * C[i][k] * C[j][k]; // Factor 2 accounts for spin degeneracy (α+β)

        // ---- Compute electronic energy: E_elec = 1/2 Tr[P(H_core + F)] ----
        double Eelec = 0.0;                          // Electronic energy accumulator
        for (int i = 0; i < NB; i++)                 // AO row index
            for (int j = 0; j < NB; j++)             // AO column index
                Eelec += 0.5 * Pnew[i][j] * (Hcore[i][j] + F_mat[i][j]); // Weighted trace formula

        Etot = Eelec + E_nuc;                        // Total HF energy = electronic + nuclear repulsion

        // ---- Check convergence ----
        double deltaE = fabs(Etot - E_old);          // Absolute energy change this iteration
        double deltaP = 0.0;                         // Density matrix change (element-wise L1 norm)
        for (int i = 0; i < NB; i++)                 // Row
            for (int j = 0; j < NB; j++)             // Column
                deltaP += fabs(Pnew[i][j] - P[i][j]); // Accumulate |P_new - P_old| element by element

        if (deltaE < 1e-8 && deltaP < 1e-6) {       // Both energy and density must be stable to converge
            memcpy(C_final,   C,    sizeof(C));      // Save final MO coefficients
            memcpy(eps_final, eps,  sizeof(eps));    // Save final orbital energies
            memcpy(P_final,   Pnew, sizeof(Pnew));  // Save final density matrix
            memcpy(G_final,   G,    sizeof(G));      // Save final two-electron Fock contribution
            break;                                   // Exit SCF loop: converged
        }

        // ---- Adaptive damping: reduce mixing fraction as SCF progresses ----
        if (iter > 8)  damp = 0.35;  // Moderate damping after 8 iterations
        if (iter > 20) damp = 0.20;  // Light damping after 20 iterations (allow faster convergence)

        // ---- Damped density update: P ← (1-α)P_old + α P_new ----
        for (int i = 0; i < NB; i++)                 // Row
            for (int j = 0; j < NB; j++)             // Column
                P[i][j] = (1.0-damp)*P[i][j] + damp*Pnew[i][j]; // Mix old and new density matrices

        E_old = Etot; // Store current total energy for convergence check next iteration

        // Always keep the latest quantities in case SCF does not fully converge
        memcpy(C_final,   C,    sizeof(C));   // Update best MO coefficients
        memcpy(eps_final, eps,  sizeof(eps)); // Update best orbital energies
        memcpy(P_final,   Pnew, sizeof(Pnew)); // Update best density matrix
        memcpy(G_final,   G,    sizeof(G));   // Update best two-electron term
    }

    // ---- Print convergence report once per MD step ----
    static int last_step_reported = -1;              // Tracks last MD step printed; prevents duplicate output
    if (md_step != last_step_reported) {             // Only print once per MD step
        if (iter >= max_iter-1)                      // Loop ran to max_iter without breaking
            printf("Warning: SCF did not converge after %d iterations\n", max_iter);
        else if (iter > 30)                          // Converged, but required many iterations
            printf("SCF converged in %d iterations\n", iter+1);
        last_step_reported = md_step;                // Update last-reported step
    }

    // ====================== EDUCATIONAL ANALYSIS ======================
    // All quantities below are derived from SCF data already in memory.
    // This section adds ZERO extra SCF evaluations (no new integrals, no new diagonalizations).
    if (do_analysis) {

        // 1. Orbital energies and MO coefficients
        g_nb  = NB;   // Store number of basis functions for rendering
        g_occ = occ;  // Store number of occupied MOs for rendering
        for (int i = 0; i < NB; i++) {                        // Loop over MOs
            g_orbital_eps[i] = eps_final[i];                  // Copy orbital energy ε_i (Ha)
            for (int j = 0; j < NB; j++)                      // Loop over AO indices
                g_mo_coeff[i][j] = C_final[i][j];             // Copy MO coefficient C[AO i][MO j]
        }

        // 2. Mulliken gross population analysis
        //    Gross population of atom A: n_A = Σ_{μ∈A} Σ_ν P[μ][ν] S[ν][μ] = (P·S)[A][A]
        //    For H: Z_A = 1 nuclear charge, so net charge q_A = 1 - n_A
        for (int A = 0; A < NB; A++) {         // Loop over atoms (one contracted AO per atom in STO-3G)
            double gross = 0.0;                // Gross electron population accumulator for atom A
            for (int nu = 0; nu < NB; nu++)    // Sum over all basis functions ν
                gross += P_final[A][nu] * S[nu][A]; // Contribution to gross population: P[A][ν] S[ν][A]
            g_mulliken_q[A] = 1.0 - gross;    // Net charge = nuclear charge Z=1 minus gross population
        }

        // 3. Energy decomposition from converged P, T_kin, Vne, G
        //    E_kin = Tr(P · T_kin)          → electronic kinetic energy
        //    E_vne = Tr(P · V_ne)           → electron–nuclear attraction
        //    E_vee = 1/2 Tr(P · G)          → electron–electron repulsion (mean-field)
        //    E_nuc already computed above    → nuclear–nuclear repulsion (geometric only)
        double Ekin_comp = 0.0, Evne_comp = 0.0, Evee_comp = 0.0; // Component accumulators
        for (int i = 0; i < NB; i++)          // Row index
            for (int j = 0; j < NB; j++) {   // Column index
                Ekin_comp += P_final[i][j] * T_kin[i][j];         // Kinetic: Tr(P·T)
                Evne_comp += P_final[i][j] * Vne[i][j];           // Nuclear attraction: Tr(P·V_ne)
                Evee_comp += 0.5 * P_final[i][j] * G_final[i][j]; // Electron repulsion: 1/2 Tr(P·G)
            }
        g_E_kin = Ekin_comp;  // Store kinetic energy component
        g_E_vne = Evne_comp;  // Store nuclear attraction component
        g_E_vee = Evee_comp;  // Store electron repulsion component
        g_E_nuc = E_nuc;      // Store nuclear repulsion (already computed geometrically)
    }

    return Etot; // Return converged (or best) total HF energy in Hartree atomic units
}

// ====================== HF FORCES VIA FINITE DIFFERENCE ======================
// Computes nuclear accelerations as a = F/m where F = -dE/dR (numerical gradient).
// Central difference: F_α ≈ -[E(R+δ) - E(R-δ)] / (2δ) for each coordinate α.
// Only the energy at the central geometry (no displacement) triggers educational analysis.
void compute_hf_accelerations(struct Atom *atoms_in, // Atom array to update (ax,ay,az modified in place)
                               int NB,               // Number of atoms
                               double *E_out) {      // Output: HF energy at current geometry (Ha)
    double R[MAX_ATOM][3];                           // Local copy of nuclear coordinates for displacement
    for (int i = 0; i < NB; i++) {                  // Copy atom positions into the R array
        R[i][0] = atoms_in[i].x;                    // x coordinate
        R[i][1] = atoms_in[i].y;                    // y coordinate
        R[i][2] = atoms_in[i].z;                    // z coordinate
    }

    const double delta = 0.001;                      // Finite-difference displacement: 0.001 bohr
    *E_out = compute_hf_energy(R, NB, 1);           // Energy at central geometry + extract analysis data

    for (int i = 0; i < NB; i++) {                  // Loop over each atom to compute forces
        double Rs[3] = {R[i][0], R[i][1], R[i][2]}; // Save original position before displacement
        double Ep, Em;                               // E(R+δ) and E(R-δ) for central difference

        // x-component of force
        R[i][0] = Rs[0]+delta; Ep = compute_hf_energy(R, NB, 0); // E at +δx (no analysis, save cost)
        R[i][0] = Rs[0]-delta; Em = compute_hf_energy(R, NB, 0); // E at -δx
        R[i][0] = Rs[0];                                          // Restore x coordinate
        atoms_in[i].ax = -(Ep-Em)/(2.0*delta) / atoms_in[i].mass; // a_x = -dE/dx / m

        // y-component of force
        R[i][1] = Rs[1]+delta; Ep = compute_hf_energy(R, NB, 0); // E at +δy
        R[i][1] = Rs[1]-delta; Em = compute_hf_energy(R, NB, 0); // E at -δy
        R[i][1] = Rs[1];                                          // Restore y coordinate
        atoms_in[i].ay = -(Ep-Em)/(2.0*delta) / atoms_in[i].mass; // a_y = -dE/dy / m

        // z-component of force
        R[i][2] = Rs[2]+delta; Ep = compute_hf_energy(R, NB, 0); // E at +δz
        R[i][2] = Rs[2]-delta; Em = compute_hf_energy(R, NB, 0); // E at -δz
        R[i][2] = Rs[2];                                          // Restore z coordinate
        atoms_in[i].az = -(Ep-Em)/(2.0*delta) / atoms_in[i].mass; // a_z = -dE/dz / m
    }
}

// ====================== REFLECTIVE BOUNDARY CONDITIONS ======================
// Applies elastic reflection when an atom crosses a wall of the cubic box.
// Repositions the atom just inside the wall and reverses the normal velocity component.
void applyBoundaryConditions(struct Atom *e) {   // Pointer to the atom to confine
    float h = box_size / 2.0f;                  // Half-box: valid range is [-h, +h] in each dimension
    if (e->x + e->radius >  h) { e->x =  h - e->radius; e->vx_old = -e->vx_old; } // +x wall reflection
    if (e->x - e->radius < -h) { e->x = -h + e->radius; e->vx_old = -e->vx_old; } // -x wall reflection
    if (e->y + e->radius >  h) { e->y =  h - e->radius; e->vy_old = -e->vy_old; } // +y wall reflection
    if (e->y - e->radius < -h) { e->y = -h + e->radius; e->vy_old = -e->vy_old; } // -y wall reflection
    if (e->z + e->radius >  h) { e->z =  h - e->radius; e->vz_old = -e->vz_old; } // +z wall reflection
    if (e->z - e->radius < -h) { e->z = -h + e->radius; e->vz_old = -e->vz_old; } // -z wall reflection
}

// ====================== LANGEVIN THERMOSTAT ======================
// Rescales velocities toward the target temperature using a Langevin scheme:
//   v ← c1·v + c2·ξ  where c1 = e^{-γΔt}, c2 = sqrt((1-c1²)k_B T / m), ξ ~ N(0,1)
// Returns the instantaneous temperature BEFORE the thermostat is applied.
float controlTemperature(float T_desired) {       // Target temperature in Kelvin
    const float kB = 3.1668114e-6f;              // Boltzmann constant in atomic units (Ha/K)
    float K = 0.0f;                              // Total kinetic energy accumulator
    for (int i = 0; i < NUM_ATOMS; i++) {        // Loop over all atoms
        struct Atom *e = &atoms[i];              // Pointer to current atom
        K += 0.5f * e->mass * (                 // K += 1/2 m v²
            e->vx_old*e->vx_old +               // v_x² contribution
            e->vy_old*e->vy_old +               // v_y² contribution
            e->vz_old*e->vz_old);               // v_z² contribution
    }
    float T_inst_local = (2.0f*K) / (3.0f*NUM_ATOMS*kB); // T = 2K/(3Nk_B) from equipartition theorem
    float c1  = expf(-0.1f * dt);                         // Friction decay factor: c1 = e^{-γΔt}, γ=0.1
    float c2b = sqrtf((1.0f - c1*c1) * kB * T_desired);  // Noise amplitude prefactor (before 1/√m)
    for (int i = 0; i < NUM_ATOMS; i++) {                  // Apply Langevin stochastic update to each atom
        struct Atom *e = &atoms[i];                        // Pointer to current atom
        float c2 = c2b / sqrtf(e->mass);                  // Per-atom noise amplitude: c2 / sqrt(m)
        e->vx_old = c1*e->vx_old + c2*(float)gauss_rand(); // v_x ← c1 v_x + c2 ξ_x
        e->vy_old = c1*e->vy_old + c2*(float)gauss_rand(); // v_y ← c1 v_y + c2 ξ_y
        e->vz_old = c1*e->vz_old + c2*(float)gauss_rand(); // v_z ← c1 v_z + c2 ξ_z
    }
    return T_inst_local; // Return instantaneous temperature before thermostat (for HUD display)
}

// ====================== VIRIAL PRESSURE ESTIMATOR ======================
// Computes instantaneous pressure via the virial theorem:
//   P = (2K + Σ r·F) / (3V)  where F = m·a from HF forces
float computePressure() {
    float K      = 0.0f; // Total kinetic energy accumulator
    float virial = 0.0f; // Virial Σ r·F = Σ r·(m·a) accumulator
    for (int i = 0; i < NUM_ATOMS; i++) {           // Loop over all atoms
        struct Atom *e = &atoms[i];                 // Pointer to current atom
        K += 0.5f * e->mass * (                    // Kinetic energy: 1/2 m v²
            e->vx_old*e->vx_old +                  // v_x² term
            e->vy_old*e->vy_old +                  // v_y² term
            e->vz_old*e->vz_old);                  // v_z² term
        virial += e->x * e->mass * e->ax           // Virial x-component: x · (m · a_x)
               +  e->y * e->mass * e->ay           // Virial y-component: y · (m · a_y)
               +  e->z * e->mass * e->az;          // Virial z-component: z · (m · a_z)
    }
    float V = box_size * box_size * box_size;      // Simulation box volume: V = L³ (bohr³)
    return (2.0f*K + virial) / (3.0f*V);           // Virial pressure: P = (2K + W) / (3V)
}

// ====================== BERENDSEN BAROSTAT ======================
// Rescales the box and all atom positions to drive pressure toward the target.
// Scale factor: λ = 1 - κ(Δt/τ)(P_target - P_current), applied to L and all r_i.
void rescaleBox(float P_current) {                 // Current instantaneous pressure (a.u.)
    float scale = 1.0f - 0.0005f*(dt/5.0f)*(target_pressure - P_current); // Barostat scale factor λ
    box_size *= scale;                             // Scale box side length: L ← λ L
    for (int i = 0; i < NUM_ATOMS; i++) {          // Scale all atom positions proportionally
        atoms[i].x *= scale;                       // x ← λ x
        atoms[i].y *= scale;                       // y ← λ y
        atoms[i].z *= scale;                       // z ← λ z
    }
}

// ====================== MD INTEGRATION STEP (Velocity Verlet + Langevin) ======================
// One full BAOAB step:
//   1. First half-velocity kick:   v += 1/2 a Δt
//   2. Langevin thermostat:        v ← c1 v + c2 ξ  (O step)
//   3. Position update:            r += v Δt
//   4. Compute new HF forces:      a = F/m from SCF
//   5. Second half-velocity kick:  v += 1/2 a_new Δt
//   6. Pressure + barostat
void stepMD() {
    // --- Step 1: First half-velocity update (B step) ---
    for (int i = 0; i < NUM_ATOMS; i++) {          // Loop over all atoms
        atoms[i].vx_old += 0.5f * atoms[i].ax * dt; // v_x += 1/2 a_x Δt (first Velocity Verlet kick)
        atoms[i].vy_old += 0.5f * atoms[i].ay * dt; // v_y += 1/2 a_y Δt
        atoms[i].vz_old += 0.5f * atoms[i].az * dt; // v_z += 1/2 a_z Δt
    }

    // --- Step 2: Langevin thermostat (O step) ---
    T_inst = controlTemperature(T_target); // Apply stochastic friction + noise; returns T before rescaling

    // --- Step 3: Position update (A step) ---
    for (int i = 0; i < NUM_ATOMS; i++) {             // Loop over all atoms
        atoms[i].x += atoms[i].vx_old * dt;           // x += v_x Δt (advance position)
        atoms[i].y += atoms[i].vy_old * dt;           // y += v_y Δt
        atoms[i].z += atoms[i].vz_old * dt;           // z += v_z Δt
        applyBoundaryConditions(&atoms[i]);            // Reflect off box walls if needed
    }

    // --- Step 4: Compute new HF forces (the expensive step: 6N+1 SCF evaluations) ---
    compute_hf_accelerations(atoms, NUM_ATOMS, &E_hf_cached); // Updates ax,ay,az; caches energy; extracts analysis

    // --- Step 5: Second half-velocity update (B step) ---
    for (int i = 0; i < NUM_ATOMS; i++) {              // Loop over all atoms
        atoms[i].vx_old += 0.5f * atoms[i].ax * dt;   // v_x += 1/2 a_x_new Δt (second Velocity Verlet kick)
        atoms[i].vy_old += 0.5f * atoms[i].ay * dt;   // v_y += 1/2 a_y_new Δt
        atoms[i].vz_old += 0.5f * atoms[i].az * dt;   // v_z += 1/2 a_z_new Δt
    }

    // --- Step 6: Pressure measurement and barostat ---
    P_inst = computePressure();  // Compute instantaneous virial pressure from current state
    rescaleBox(P_inst);          // Apply Berendsen barostat: scale box and positions if needed

    // --- Update minimum bond distance ---
    min_bond = 1e9f;                                   // Initialize minimum to a very large sentinel
    for (int i = 0; i < NUM_ATOMS; i++)                // Loop over all atom pairs (outer)
        for (int j = i+1; j < NUM_ATOMS; j++) {        // Only unique pairs i < j
            float d = atomDistance(atoms[i], atoms[j]); // Distance between atoms i and j
            if (d < min_bond) min_bond = d;            // Update minimum if this pair is closer
        }

    // --- Update ring buffer and counters ---
    energy_history[history_count % HISTORY_LEN] = E_hf_cached; // Store current energy in ring buffer
    history_count++;  // Increment total entries (wraps logically via modulo on read)
    md_step++;        // Increment MD step counter

    // --- Console output for this step ---
    printf("\n=== STEP %d (t=%.4f fs) ===\n", md_step, md_step*dt*0.02418884f); // Step and physical time
    printf("  E(HF)    : %+.8f Ha  (%+.4f eV)\n", E_hf_cached, E_hf_cached*27.2114); // Total HF energy
    printf("  E_kin    : %+.6f Ha   E_Vne : %+.6f Ha\n", g_E_kin, g_E_vne); // Kinetic + nuclear attraction
    printf("  E_Vee    : %+.6f Ha   E_Vnn : %+.6f Ha\n", g_E_vee, g_E_nuc); // e-e repulsion + nuc repulsion
    printf("  T_inst   : %.2f K  (target: %.0f K)\n", T_inst, T_target);     // Temperature status
    printf("  Pressure : %+.6f a.u.\n", P_inst);                             // Virial pressure
    printf("  d_min    : %.4f bohr = %.4f Ang\n", min_bond, min_bond*0.529177f); // Min interatomic distance
    for (int i = 0; i < NUM_ATOMS; i++)                                       // Print Mulliken charge per atom
        printf("  Mulliken q[%d] : %+.4f e\n", i, g_mulliken_q[i]);
}

// ====================== PHYSICS BACKGROUND THREAD ======================
// Runs stepMD() in an infinite loop. After each MD step, acquires the mutex
// and copies all physics state into the render buffers so the OpenGL thread
// can read consistent data without blocking on the expensive SCF calculation.
void* physics_loop(void* arg) {
    (void)arg;                                      // Suppress unused-parameter warning
    while (1) {                                     // Infinite loop: physics runs continuously
        pthread_mutex_lock(&render_mutex);           // Acquire lock to update the 'computing' flag
        computing = 1;                              // Signal to HUD: SCF is running
        pthread_mutex_unlock(&render_mutex);         // Release lock immediately (minimal hold time)

        stepMD();                                   // Execute one full MD step (potentially very slow)

        pthread_mutex_lock(&render_mutex);           // Acquire lock to write all render-side buffers
        memcpy(atoms_r, atoms, sizeof(atoms));       // Copy atom positions/velocities/forces to render buffer
        E_r     = E_hf_cached;                     // Copy HF energy for HUD display
        T_r     = T_inst;                          // Copy instantaneous temperature for HUD
        P_r     = P_inst;                          // Copy instantaneous pressure for HUD
        bond_r  = min_bond;                        // Copy minimum bond distance for HUD
        step_r  = md_step;                         // Copy MD step counter for HUD
        memcpy(ehist_r, energy_history, sizeof(energy_history)); // Copy energy history for graph
        hcount_r = history_count;                  // Copy history count so graph knows ring buffer state
        // Copy educational data — panel O
        memcpy(orbital_eps_r, g_orbital_eps, sizeof(g_orbital_eps)); // Orbital energies
        memcpy(mo_coeff_r,    g_mo_coeff,    sizeof(g_mo_coeff));    // MO coefficients
        // Copy educational data — panel M
        memcpy(mulliken_q_r, g_mulliken_q, sizeof(g_mulliken_q)); // Mulliken charges
        // Copy educational data — panel E
        E_kin_r = g_E_kin;  // Electronic kinetic energy
        E_vne_r = g_E_vne;  // Nuclear attraction energy
        E_vee_r = g_E_vee;  // Electron-electron repulsion energy
        E_nuc_r = g_E_nuc;  // Nuclear-nuclear repulsion energy
        // Copy basis metadata
        nb_r  = g_nb;   // Number of basis functions
        occ_r = g_occ;  // Number of occupied MOs
        computing = 0;  // Signal to HUD: render buffer is fresh and consistent
        pthread_mutex_unlock(&render_mutex);         // Release lock so OpenGL thread can safely read
    }
    return NULL; // Unreachable; required for void* function signature
}

// ====================== 2D TEXT RENDERING HELPER ======================
void drawText(float x, float y, const char *str, void *font) { // Renders ASCII string at 2D ortho position
    glRasterPos2f(x, y);                            // Set raster (pen) position in 2D orthographic space
    for (int i = 0; str[i]; i++)                    // Loop over each character of the string
        glutBitmapCharacter(font, str[i]);          // Render character using GLUT bitmap font
}

// ====================== BOND CYLINDER RENDERER ======================
// Draws a yellow cylinder between two 3D points, representing a chemical bond.
void drawBond(float x1, float y1, float z1,   // Start point (atom i position)
              float x2, float y2, float z2) { // End point (atom j position)
    float dx = x2-x1, dy = y2-y1, dz = z2-z1;  // Direction vector from atom i to atom j
    float len = sqrtf(dx*dx + dy*dy + dz*dz);   // Bond length: Euclidean distance between endpoints
    if (len < 1e-4f) return;                     // Skip degenerate bonds (atoms at same position)
    float rot_ax = -dy, rot_ay = dx;             // Rotation axis = (0,0,1) × bond_dir = (-dy, dx, 0)
    float angle = (180.0f/M_PI) * acosf(dz/len); // Rotation angle: arccos of z-component of unit bond vector
    if (fabsf(dz/len) > 0.9999f) {              // Degenerate case: bond is nearly parallel to the Z axis
        rot_ax = 1.0f; rot_ay = 0.0f;           // Use X axis as rotation axis (arbitrary in degenerate case)
        angle = (dz < 0) ? 180.0f : 0.0f;       // 180° if pointing -Z, 0° if +Z
    }
    glPushMatrix();                              // Save current modelview matrix
    glTranslatef(x1, y1, z1);                   // Translate to start atom position
    glRotatef(angle, rot_ax, rot_ay, 0.0f);     // Rotate default +Z cylinder to match bond direction
    GLUquadric *q = gluNewQuadric();             // Create GLU quadric object for cylinder generation
    glColor4f(0.95f, 0.85f, 0.1f, 0.85f);      // Bond color: warm yellow, slightly transparent
    gluCylinder(q, 0.07, 0.07, len, 14, 1);    // Draw cylinder: radii 0.07, height=len, 14 subdivisions
    gluDeleteQuadric(q);                        // Free the quadric to avoid memory leak
    glPopMatrix();                              // Restore modelview matrix
}

// ====================== PANEL F: RED FORCE ARROWS ======================
// Draws a red 3D arrow from an atom's position along its force vector.
// The force is F = m·a, already computed from the HF gradient — zero extra cost.
// Arrow length is visually scaled for clarity; direction is physically correct.
void drawForceArrow(float x,  float y,  float z,   // Atom position (arrow origin)
                    float fx, float fy, float fz) { // Force vector components F = m·a (a.u.)
    float flen = sqrtf(fx*fx + fy*fy + fz*fz);     // Force magnitude |F|
    if (flen < 1e-14f) return;                      // Skip atoms with negligible force (avoid degenerate arrow)

    float vscale = 300.0f;                          // Visual scale factor: maps force (a.u.) to OpenGL units
    float ex = x + fx*vscale;                       // Arrow tip x = atom x + scaled force x
    float ey = y + fy*vscale;                       // Arrow tip y = atom y + scaled force y
    float ez = z + fz*vscale;                       // Arrow tip z = atom z + scaled force z
    float alen = sqrtf((ex-x)*(ex-x)+(ey-y)*(ey-y)+(ez-z)*(ez-z)); // Actual arrow length after scaling
    if (alen < 0.02f) return;                       // Skip if arrow would be too short to see

    glDisable(GL_LIGHTING);                         // Disable lighting so arrow appears flat bright red
    glColor3f(1.0f, 0.12f, 0.12f);                 // Bright red color for force arrow shaft
    glLineWidth(2.8f);                              // Thicker line for visibility
    glBegin(GL_LINES);                              // Draw the arrow shaft as a single line segment
    glVertex3f(x, y, z);                           // Arrow origin: atom center
    glVertex3f(ex, ey, ez);                        // Arrow tip: atom center + scaled force
    glEnd();
    glLineWidth(1.0f);                              // Restore default line width

    // --- Cone arrowhead at the tip ---
    float dx = ex-x, dy2 = ey-y, dz2 = ez-z;      // Direction vector of the arrow (same as force direction)
    float norm = sqrtf(dx*dx + dy2*dy2 + dz2*dz2); // Normalize: compute arrow shaft length
    float axr = -dy2, ayr = dx;                    // Rotation axis to align cone with arrow direction
    float ang = (180.0f/M_PI) * acosf(dz2/norm);  // Rotation angle in degrees
    if (fabsf(dz2/norm) > 0.9999f) {              // Degenerate case: arrow nearly parallel to Z
        axr = 1; ayr = 0; ang = (dz2 < 0) ? 180.0f : 0.0f;
    }
    glPushMatrix();                                // Save matrix before cone transform
    glTranslatef(ex, ey, ez);                     // Translate to arrow tip position
    glRotatef(ang, axr, ayr, 0.0f);              // Rotate cone from +Z to arrow direction
    GLUquadric *q = gluNewQuadric();              // Create quadric for cone geometry
    glColor3f(1.0f, 0.25f, 0.1f);               // Slightly darker red for the cone tip
    gluCylinder(q, 0.09f, 0.0f, 0.22f, 12, 1); // Cone: base radius 0.09, apex 0, height 0.22
    gluDeleteQuadric(q);                         // Free quadric memory
    glPopMatrix();                               // Restore matrix
    glEnable(GL_LIGHTING);                       // Re-enable lighting for subsequent 3D rendering
}

// ====================== PANEL O: MOLECULAR ORBITAL DIAGRAM ======================
// Displays a 2D energy-level diagram using orbital energies and occupancies
// already extracted from the converged SCF — zero extra computational cost.
// Occupied MOs are shown in green with spin-up/down arrows; virtual MOs in grey.
void drawOrbitalDiagram() {
    int n = nb_r;               // Number of MOs to display = number of basis functions
    if (n < 1) return;          // Nothing to draw if no data is available yet

    float x0 = 0.62f, y0 = -2.95f; // Bottom-left corner of the orbital diagram panel
    float gw = 1.05f, gh = 2.75f;   // Panel width and height in 2D ortho units

    // --- Panel background ---
    glColor4f(0.02f, 0.06f, 0.18f, 0.93f);  // Dark blue background, almost opaque
    glBegin(GL_QUADS);                        // Filled rectangle for panel background
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0); // Bottom edge
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh); // Top edge
    glEnd();
    glColor3f(0.2f, 0.5f, 0.85f);            // Blue border color
    glBegin(GL_LINE_LOOP);                    // Rectangular border around panel
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();

    // --- Panel title and legend ---
    glColor3f(0.3f, 0.85f, 1.0f);            // Cyan title color
    drawText(x0+0.04f, y0+gh-0.11f, "Orbitals [O]", GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.4f, 0.4f, 0.55f);            // Muted grey for legend text
    drawText(x0+0.04f, y0+gh-0.22f, "occ=green  virt=grey", GLUT_BITMAP_HELVETICA_10);

    // --- Compute energy range for vertical scaling ---
    double emin_ev = orbital_eps_r[0]   * 27.2114; // Lowest orbital energy in eV
    double emax_ev = orbital_eps_r[n-1] * 27.2114; // Highest orbital energy in eV
    double erange  = emax_ev - emin_ev;             // Energy span for normalization
    if (erange < 0.5) erange = 0.5;                // Clamp minimum range to prevent degenerate layout

    float ystart  = y0 + 0.32f;       // Lowest y position for orbital level lines
    float yend    = y0 + gh - 0.32f;  // Highest y position for orbital level lines
    float xcenter = x0 + gw * 0.5f;   // Horizontal center of the panel

    char buf[64]; // String buffer for labels
    for (int i = 0; i < n; i++) {      // Loop over all MOs (sorted by energy)
        double ev = orbital_eps_r[i] * 27.2114; // Orbital energy in eV (1 Ha = 27.2114 eV)
        float yp = ystart + (float)((ev - emin_ev)/erange) * (yend - ystart); // Map energy to y position

        // --- Draw horizontal energy level line ---
        int is_occ = (i < occ_r);               // True if this MO is occupied
        if (is_occ) glColor3f(0.15f, 1.0f, 0.45f); // Green for occupied MOs
        else        glColor3f(0.45f, 0.45f, 0.55f); // Grey for virtual (unoccupied) MOs
        glLineWidth(2.0f);                           // Thicker line for energy level
        glBegin(GL_LINES);
        glVertex2f(xcenter-0.22f, yp); glVertex2f(xcenter+0.22f, yp); // Horizontal level line
        glEnd();
        glLineWidth(1.0f);                           // Restore line width

        // --- Draw electron spin arrows on occupied MOs ---
        if (is_occ) {
            glColor3f(1.0f, 0.95f, 0.2f);           // Yellow for electron spin arrows
            // Spin-up arrow (vertical line + upward triangle)
            glBegin(GL_LINES);                       // Arrow shaft (going up)
            glVertex2f(xcenter-0.10f, yp+0.001f);   // Shaft base
            glVertex2f(xcenter-0.10f, yp+0.09f);    // Shaft tip
            glEnd();
            glBegin(GL_TRIANGLES);                   // Upward-pointing arrowhead
            glVertex2f(xcenter-0.10f, yp+0.12f);    // Triangle apex (top)
            glVertex2f(xcenter-0.13f, yp+0.08f);    // Triangle left base
            glVertex2f(xcenter-0.07f, yp+0.08f);    // Triangle right base
            glEnd();
            // Spin-down arrow (vertical line + downward triangle)
            glBegin(GL_LINES);                       // Arrow shaft (going down)
            glVertex2f(xcenter+0.04f, yp+0.12f);    // Shaft top
            glVertex2f(xcenter+0.04f, yp+0.03f);    // Shaft bottom
            glEnd();
            glBegin(GL_TRIANGLES);                   // Downward-pointing arrowhead
            glVertex2f(xcenter+0.04f, yp+0.001f);   // Triangle apex (bottom)
            glVertex2f(xcenter+0.01f, yp+0.05f);    // Triangle left base
            glVertex2f(xcenter+0.07f, yp+0.05f);    // Triangle right base
            glEnd();
        }

        // --- Energy label on the right ---
        glColor3f(0.75f, 0.85f, 0.95f);             // Light blue for energy values
        sprintf(buf, "%+.2f eV", ev);               // Format energy as signed float in eV
        drawText(xcenter+0.24f, yp-0.025f, buf, GLUT_BITMAP_HELVETICA_10);

        // --- MO index label on the left ---
        glColor3f(0.5f, 0.55f, 0.65f);              // Grey for MO index label
        sprintf(buf, "MO%d", i+1);                  // "MO1", "MO2", etc.
        drawText(x0+0.04f, yp-0.025f, buf, GLUT_BITMAP_HELVETICA_10);
    }

    // --- Central vertical energy axis ---
    glColor3f(0.25f, 0.3f, 0.4f);                   // Dark grey axis line
    glBegin(GL_LINES);
    glVertex2f(xcenter, ystart-0.05f);              // Axis bottom (slightly below lowest level)
    glVertex2f(xcenter, yend+0.05f);               // Axis top (slightly above highest level)
    glEnd();
}

// ====================== PANEL M: MULLIKEN POPULATION ANALYSIS ======================
// Displays gross electron population and net charge for each atom.
// Gross population n_A = Σ_{μ∈A} (PS)_{μμ} (already computed from converged SCF).
// Net charge q_A = Z_A - n_A  (Z=1 for hydrogen).
// Color coding: red = electron-deficient (q>0), blue = electron-rich (q<0).
void drawMulliken() {
    float x0   = -2.95f;                       // Left edge of the Mulliken panel
    float y0   =  0.22f;                       // Bottom edge of the Mulliken panel
    float gw   =  1.5f;                        // Panel width in 2D ortho units
    float rowh =  0.28f;                       // Vertical spacing between atom rows
    float gh   = 0.38f + rowh*(NUM_ATOMS+1);   // Total panel height: header + one row per atom

    // --- Panel background ---
    glColor4f(0.02f, 0.10f, 0.05f, 0.92f);    // Dark green-black background
    glBegin(GL_QUADS);
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();
    glColor3f(0.2f, 0.75f, 0.35f);            // Green border
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();

    // --- Panel title and column headers ---
    glColor3f(0.25f, 1.0f, 0.5f);             // Bright green title
    drawText(x0+0.05f, y0+gh-0.13f, "Mulliken Populations [M]", GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.35f, 0.55f, 0.4f);            // Muted green for column labels
    drawText(x0+0.05f, y0+gh-0.26f, "Atom  Gross Pop   Net Charge", GLUT_BITMAP_HELVETICA_10);

    char buf[64];                              // String buffer for formatted values
    float bar_max = 0.45f;                    // Maximum bar width in ortho units (represents 2 electrons)

    for (int i = 0; i < NUM_ATOMS; i++) {     // Loop over all atoms to display their population data
        float yp    = y0 + gh - 0.38f - rowh*(i+1); // Vertical position for this atom's row
        double gross = 1.0 - mulliken_q_r[i]; // Gross electron population = Z - q_net (Z=1 for H)
        double q_net = mulliken_q_r[i];       // Net Mulliken charge: positive = electron-deficient

        // --- Atom label ---
        glColor3f(0.9f, 0.9f, 0.5f);         // Yellow for atom index label
        sprintf(buf, "H%-2d", i+1);           // e.g. "H1 ", "H2 "
        drawText(x0+0.05f, yp, buf, GLUT_BITMAP_HELVETICA_10);

        // --- Gross population bar (green horizontal bar) ---
        float blen = (float)(gross / 2.0) * bar_max; // Bar length: normalized so 2e = full bar (for H2)
        if (blen < 0)       blen = 0;         // Clamp to zero (no negative bar)
        if (blen > bar_max) blen = bar_max;   // Clamp to maximum bar width
        glColor4f(0.15f, 0.75f, 0.3f, 0.75f); // Transparent green fill for population bar
        glBegin(GL_QUADS);
        glVertex2f(x0+0.38f,      yp-0.01f);  // Bar bottom-left corner
        glVertex2f(x0+0.38f+blen, yp-0.01f);  // Bar bottom-right corner
        glVertex2f(x0+0.38f+blen, yp+0.14f);  // Bar top-right corner
        glVertex2f(x0+0.38f,      yp+0.14f);  // Bar top-left corner
        glEnd();

        // --- Gross population numerical value ---
        glColor3f(0.7f, 0.9f, 0.75f);         // Light green for numerical values
        sprintf(buf, "%.3f", gross);           // Format gross population to 3 decimal places
        drawText(x0+0.42f, yp, buf, GLUT_BITMAP_HELVETICA_10);

        // --- Net charge with color coding ---
        if      (q_net >  0.01f) glColor3f(1.0f, 0.4f, 0.4f); // Red: positive charge (electron-deficient)
        else if (q_net < -0.01f) glColor3f(0.4f, 0.6f, 1.0f); // Blue: negative charge (electron-rich)
        else                     glColor3f(0.7f, 0.85f, 0.7f); // Green: approximately neutral
        sprintf(buf, "%+.3f e", q_net);        // Format net charge with sign, in units of elementary charge
        drawText(x0+0.90f, yp, buf, GLUT_BITMAP_HELVETICA_10);
    }
}

// ====================== PANEL E: ENERGY DECOMPOSITION ======================
// Displays the four HF energy components as labeled bars, all from converged SCF data.
//   T   = Tr(P · T_kin)        electronic kinetic energy    (always positive)
//   Vne = Tr(P · V_ne)         nuclear attraction energy    (always negative)
//   Vee = 1/2 Tr(P · G)        electron-electron repulsion  (always positive)
//   Vnn = nuclear repulsion     geometric term               (always positive)
// Bars are centered and extend left (negative) or right (positive) for intuitive comparison.
void drawEnergyDecomp() {
    float x0 = -2.95f, y0 = -1.58f; // Bottom-left corner of the energy decomposition panel
    float gw = 1.82f,  gh = 1.35f;  // Panel width and height

    // --- Panel background ---
    glColor4f(0.08f, 0.03f, 0.15f, 0.92f); // Dark purple background
    glBegin(GL_QUADS);
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();
    glColor3f(0.6f, 0.3f, 0.85f);          // Purple border
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();

    // --- Panel title ---
    glColor3f(0.75f, 0.45f, 1.0f);         // Light purple for title text
    drawText(x0+0.05f, y0+gh-0.13f, "Energy Decomposition [E]", GLUT_BITMAP_HELVETICA_10);

    double E_total = E_kin_r + E_vne_r + E_vee_r + E_nuc_r; // Reconstruct total energy from components

    // --- Component arrays: labels, values, and RGB colors ---
    const char *labels[5] = {
        "T   (e. kinetic)",   // Electronic kinetic energy
        "Vne (nuc. attr.)",   // Electron-nuclear attraction
        "Vee (e-e repuls.)",  // Electron-electron repulsion
        "Vnn (nuc. repuls.)", // Nuclear-nuclear repulsion
        "E_total"             // Sum of all components
    };
    double vals[5] = { E_kin_r, E_vne_r, E_vee_r, E_nuc_r, E_total }; // Energy values (Ha)
    float colors[5][3] = {
        {0.3f, 0.8f,  1.0f},   // Cyan:   kinetic (positive, always)
        {1.0f, 0.5f,  0.2f},   // Orange: nuclear attraction (negative, stabilizing)
        {1.0f, 0.8f,  0.2f},   // Yellow: electron repulsion (positive, destabilizing)
        {0.6f, 0.9f,  0.6f},   // Green:  nuclear repulsion (positive, destabilizing)
        {0.85f, 0.85f, 0.95f}  // White:  total energy
    };

    // --- Find absolute maximum for bar length normalization ---
    double abs_max = 0.01;   // Minimum scale to prevent division by zero
    for (int i = 0; i < 5; i++)
        if (fabs(vals[i]) > abs_max) abs_max = fabs(vals[i]); // Update scale if this value is larger

    float bar_zone = 0.55f;  // Maximum bar half-width in ortho units (positive or negative direction)
    float rowh     = 0.21f;  // Vertical row spacing
    char buf[80];            // String buffer for formatted values

    for (int i = 0; i < 5; i++) {    // Loop over all five energy components
        float yp = y0 + gh - 0.28f - rowh*(i+1); // Vertical position for this component's row

        // --- Horizontal separator line before the total energy row ---
        if (i == 4) {
            glColor3f(0.4f, 0.4f, 0.5f);   // Grey separator line
            glBegin(GL_LINES);
            glVertex2f(x0+0.03f, yp+rowh-0.02f);   // Left end of separator
            glVertex2f(x0+gw-0.03f, yp+rowh-0.02f); // Right end of separator
            glEnd();
        }

        // --- Component label ---
        glColor3f(colors[i][0], colors[i][1], colors[i][2]); // Use component's assigned color
        drawText(x0+0.05f, yp, labels[i], GLUT_BITMAP_HELVETICA_10); // Render label text

        // --- Bidirectional bar: extends right for positive, left for negative values ---
        float midx  = x0 + 1.12f;                         // Center x position (zero line of the bar chart)
        float blen  = (float)(vals[i]/abs_max) * bar_zone; // Normalized bar length (signed)
        float bx_lo = (blen >= 0) ? midx        : midx+blen; // Left edge of bar rectangle
        float bx_hi = (blen >= 0) ? midx+blen   : midx;      // Right edge of bar rectangle
        if (fabs(blen) > 0.003f) {                          // Only draw bar if it's large enough to see
            glColor4f(colors[i][0], colors[i][1], colors[i][2], 0.55f); // Semi-transparent bar fill
            glBegin(GL_QUADS);
            glVertex2f(bx_lo, yp+0.01f);  glVertex2f(bx_hi, yp+0.01f);  // Bottom edge
            glVertex2f(bx_hi, yp+0.15f);  glVertex2f(bx_lo, yp+0.15f);  // Top edge
            glEnd();
        }

        // --- Numerical value in Hartree ---
        glColor3f(colors[i][0]*0.85f+0.15f,   // Slightly brightened component color for readability
                  colors[i][1]*0.85f+0.15f,
                  colors[i][2]*0.85f+0.15f);
        sprintf(buf, "%+.5f Ha", vals[i]);    // Signed energy value to 5 decimal places
        drawText(x0+gw-0.72f, yp, buf, GLUT_BITMAP_HELVETICA_10);
    }
}

// ====================== REAL-TIME ENERGY GRAPH ======================
// Draws a line graph of the HF energy history in the lower-left area of the HUD.
// Uses the ring buffer ehist_r[] and the count hcount_r for correct ordering.
void drawEnergyGraph(float x0, float y0) {        // Bottom-left corner of graph area
    int n = (hcount_r < HISTORY_LEN) ? hcount_r : HISTORY_LEN; // Number of valid data points
    if (n < 2) return;                             // Need at least 2 points to draw a line
    int start = (hcount_r > HISTORY_LEN) ? (hcount_r % HISTORY_LEN) : 0; // Ring buffer start index

    // --- Find min and max energy for auto-scaling the y axis ---
    double emin = ehist_r[start], emax = emin;    // Initialize range with first value
    for (int i = 0; i < n; i++) {                 // Scan all valid entries
        int idx = (start+i) % HISTORY_LEN;        // Correct ring buffer index
        if (ehist_r[idx] < emin) emin = ehist_r[idx]; // Update minimum
        if (ehist_r[idx] > emax) emax = ehist_r[idx]; // Update maximum
    }
    double er = emax - emin;                      // Energy range for y-axis scaling
    if (er < 1e-12) er = 1e-12;                  // Prevent division by zero if all values are equal

    float gw = 1.65f, gh = 0.58f;                // Graph width and height in ortho units

    // --- Graph background rectangle ---
    glColor4f(0, 0.05f, 0.15f, 0.85f);           // Dark navy background
    glBegin(GL_QUADS);
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();

    // --- Graph border ---
    glColor3f(0.25f, 0.45f, 0.7f);               // Blue border
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0); glVertex2f(x0+gw, y0);
    glVertex2f(x0+gw, y0+gh); glVertex2f(x0, y0+gh);
    glEnd();

    // --- Graph title ---
    glColor3f(0.65f, 0.85f, 1.0f);               // Light blue title
    drawText(x0+0.04f, y0+gh-0.1f, "E(HF) Ha vs step", GLUT_BITMAP_HELVETICA_10);

    // --- Energy curve ---
    glColor3f(0.1f, 1.0f, 0.55f);                // Bright green curve
    glLineWidth(1.6f);                            // Slightly thick line for visibility
    glBegin(GL_LINE_STRIP);                       // Connected line segments for the energy history
    for (int i = 0; i < n; i++) {                 // Loop over all valid data points
        int   idx = (start+i) % HISTORY_LEN;     // Ring buffer index for this point
        float xp  = x0+0.05f + (float)i/(n-1)*(gw-0.1f); // Map step to x position in graph
        float yp  = y0+0.06f + (float)((ehist_r[idx]-emin)/er)*(gh-0.18f); // Map energy to y position
        glVertex2f(xp, yp);                      // Add vertex for this energy data point
    }
    glEnd();
    glLineWidth(1.0f);                            // Restore default line width
}

// ====================== HEADS-UP DISPLAY (HUD) ======================
// Renders all 2D informational overlays using an orthographic projection.
// Called at the end of display() after all 3D geometry is drawn.
void drawHUD() {
    // Switch to 2D orthographic projection covering [-3,3] x [-3,3]
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(-3,3,-3,3);
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity(); // Reset modelview for 2D
    glDisable(GL_LIGHTING);    // Disable 3D lighting so 2D text appears flat
    glDisable(GL_DEPTH_TEST);  // Disable depth test so HUD always renders on top

    char buf[200]; // General-purpose string buffer for formatted output

    // --- Main title ---
    glColor3f(0.3f, 0.85f, 1.0f); // Cyan title color
    drawText(-2.9f, 2.72f, "Ab initio MD  -  Hartree-Fock / STO-3G", GLUT_BITMAP_HELVETICA_18);

    // --- System subtitle ---
    glColor3f(0.5f, 0.5f, 0.62f); // Muted grey-purple subtitle color
    sprintf(buf, "System: %d H atom(s) | %d electrons | STO-3G basis | Velocity Verlet + Langevin", NUM_ATOMS, NUM_ATOMS);
    drawText(-2.9f, 2.47f, buf, GLUT_BITMAP_HELVETICA_12);

    // --- Computing indicator (shown only while SCF is running) ---
    if (computing) {
        glColor3f(1.0f, 0.5f, 0.0f); // Orange warning color
        drawText(-2.9f, 2.22f, "[ COMPUTING SCF... ]", GLUT_BITMAP_HELVETICA_12);
    }

    // --- Right panel: physical observables ---
    float px = 1.12f, py = 2.72f, dy = 0.27f; // Starting position and row spacing for right panel

    glColor3f(1.0f, 0.85f, 0.0f); // Yellow for step counter
    sprintf(buf, "Step    : %d", step_r);
    drawText(px, py, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(0.78f, 0.78f, 0.84f); // Light grey for simulation time
    sprintf(buf, "Time    : %.4f fs", step_r*dt*0.02418884f); // Convert a.u. to femtoseconds
    drawText(px, py-dy, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(0.2f, 1.0f, 0.5f); // Bright green for HF energy
    sprintf(buf, "E(HF)   : %+.6f Ha", E_r);
    drawText(px, py-2*dy, buf, GLUT_BITMAP_HELVETICA_12);
    sprintf(buf, "          %+.4f eV", E_r*27.2114); // Convert Hartree to eV
    drawText(px, py-3*dy, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(1.0f, 0.38f, 0.3f); // Red-orange for instantaneous temperature
    sprintf(buf, "T_inst  : %.1f K", T_r);
    drawText(px, py-4*dy, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(0.65f, 0.3f, 0.22f); // Darker red for target temperature
    sprintf(buf, "T_target: %.0f K  (+/- adjust)", T_target);
    drawText(px, py-5*dy, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(0.4f, 0.7f, 1.0f); // Sky blue for pressure
    sprintf(buf, "Pressure: %+.5f a.u.", P_r);
    drawText(px, py-6*dy, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(1.0f, 0.78f, 0.18f); // Gold for bond distances
    sprintf(buf, "d_min   : %.4f bohr", bond_r);
    drawText(px, py-7*dy, buf, GLUT_BITMAP_HELVETICA_12);
    sprintf(buf, "          %.4f Ang  (H2 eq: 0.74)", bond_r*0.529177f); // Convert bohr to Angstrom
    drawText(px, py-8*dy, buf, GLUT_BITMAP_HELVETICA_12);

    // --- Pairwise distance table (only for small systems) ---
    if (NUM_ATOMS <= 4) {                         // Avoid overflow for large systems
        glColor3f(0.6f, 0.7f, 0.5f);             // Muted green for pair distance table
        float yy = py - 9*dy;                     // Starting y position below bond distance display
        for (int i = 0; i < NUM_ATOMS; i++)       // Outer atom index
            for (int j = i+1; j < NUM_ATOMS; j++) { // Inner atom index (unique pairs only)
                float dd = atomDistance(atoms_r[i], atoms_r[j]); // Current distance between atoms i and j
                sprintf(buf, "d(%d-%d)  : %.3f bohr", i, j, dd); // Format pair distance label
                drawText(px, yy, buf, GLUT_BITMAP_HELVETICA_10); // Small font for table
                yy -= 0.18f;                      // Step down for next pair
            }
    }

    // --- Method information section ---
    glColor3f(0.45f, 0.45f, 0.75f); // Purple section header
    drawText(-2.9f, -1.72f, "[ Quantum Method ]", GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.5f, 0.5f, 0.62f); // Grey for method details
    drawText(-2.9f, -1.89f, "Theory    : Hartree-Fock (RHF, closed-shell)",       GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f, -2.04f, "Basis     : STO-3G (3 Gaussians per AO)",            GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f, -2.19f, "Gradient  : Central finite difference (d=0.001 a.u.)", GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f, -2.34f, "Integrator: Velocity Verlet",                         GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f, -2.49f, "Thermostat: Langevin  |  Barostat: Berendsen",       GLUT_BITMAP_HELVETICA_10);

    // --- Keyboard controls reference ---
    glColor3f(0.38f, 0.38f, 0.48f); // Dark grey for controls hint
    drawText(-2.9f, -2.68f, "W/S: zoom   A/D: pan   Z/X: up/down   +/-: temperature", GLUT_BITMAP_HELVETICA_10);

    // --- Educational panel toggle indicators (lit = active, dim = inactive) ---
    float kx = -2.9f, ky = -2.82f, ksp = 0.62f; // Position and spacing for toggle buttons
    const char *key_labels[4] = {"[O] Orbitals", "[M] Mulliken", "[E] Energy", "[F] Forces"}; // Button labels
    int  *key_flags[4]  = {&show_orbital, &show_mulliken, &show_energy_dec, &show_forces}; // Toggle flags
    float key_colors[4][3] = {
        {0.3f,  0.85f, 1.0f},   // Cyan for orbital panel
        {0.25f, 1.0f,  0.5f},   // Green for Mulliken panel
        {0.75f, 0.4f,  1.0f},   // Purple for energy panel
        {1.0f,  0.25f, 0.25f}   // Red for force panel
    };
    for (int i = 0; i < 4; i++) {                                     // Draw each toggle indicator
        if (*key_flags[i]) glColor3f(key_colors[i][0], key_colors[i][1], key_colors[i][2]); // Active: bright
        else               glColor3f(0.3f, 0.3f, 0.38f);             // Inactive: dim grey
        drawText(kx + ksp*i, ky, key_labels[i], GLUT_BITMAP_HELVETICA_10);
    }

    // --- Always-visible energy history graph ---
    drawEnergyGraph(-2.9f, -1.35f); // Position: lower-left area of HUD

    // --- Educational panels (shown only when their toggle is active) ---
    if (show_orbital)    drawOrbitalDiagram(); // Panel O: MO energy levels
    if (show_mulliken)   drawMulliken();        // Panel M: electron populations
    if (show_energy_dec) drawEnergyDecomp();    // Panel E: energy component bars

    // Restore 3D rendering state
    glEnable(GL_DEPTH_TEST);                    // Re-enable depth testing for 3D geometry
    glEnable(GL_LIGHTING);                      // Re-enable lighting for shaded atoms
    glMatrixMode(GL_PROJECTION); glPopMatrix(); // Restore perspective projection matrix
    glMatrixMode(GL_MODELVIEW);  glPopMatrix(); // Restore 3D camera modelview matrix
}

// ====================== MAIN DISPLAY CALLBACK ======================
// Called every frame by GLUT. Renders the complete scene: 3D molecules + 2D HUD.
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers for fresh frame
    glLoadIdentity();                                    // Reset modelview matrix
    gluLookAt(obs_x, obs_y, obs_z, 0, 0, 0, 0, 1, 0); // Camera: eye at obs, looking at origin, Y-up

    // --- Safely snapshot atom state from render buffer ---
    struct Atom snap[MAX_ATOM]; // Thread-local snapshot; safe to use without holding mutex
    float snap_box;             // Local copy of box size for consistent rendering
    pthread_mutex_lock(&render_mutex);                               // Acquire lock
    memcpy(snap, atoms_r, NUM_ATOMS * sizeof(struct Atom));          // Copy render-side atom data
    snap_box = box_size;                                             // Read current box size
    pthread_mutex_unlock(&render_mutex);                             // Release lock immediately

    // --- Draw simulation box wireframe ---
    glDisable(GL_LIGHTING);                    // Flat color looks better for wireframe box
    {
        float h = snap_box / 2.0f;            // Half-side: box spans [-h, h] in each dimension
        glColor4f(0.35f, 0.55f, 0.8f, 0.35f); glLineWidth(1.2f); // Semi-transparent blue box lines
        glBegin(GL_LINE_LOOP);                // Back face of box (z = -h)
        glVertex3f(-h,-h,-h); glVertex3f(h,-h,-h); glVertex3f(h,h,-h); glVertex3f(-h,h,-h);
        glEnd();
        glBegin(GL_LINE_LOOP);                // Front face of box (z = +h)
        glVertex3f(-h,-h,h); glVertex3f(h,-h,h); glVertex3f(h,h,h); glVertex3f(-h,h,h);
        glEnd();
        glBegin(GL_LINES);                    // Four vertical edges connecting front and back faces
        glVertex3f(-h,-h,-h); glVertex3f(-h,-h,h); // Left-bottom edge
        glVertex3f( h,-h,-h); glVertex3f( h,-h,h); // Right-bottom edge
        glVertex3f( h, h,-h); glVertex3f( h, h,h); // Right-top edge
        glVertex3f(-h, h,-h); glVertex3f(-h, h,h); // Left-top edge
        glEnd();
    }

    // --- Draw bond cylinders between atoms within cutoff ---
    for (int i = 0; i < NUM_ATOMS; i++)                    // Outer atom loop
        for (int j = i+1; j < NUM_ATOMS; j++) {            // Inner atom loop (unique pairs only)
            float dd = atomDistance(snap[i], snap[j]);     // Interatomic distance for this pair
            if (dd < BOND_CUTOFF)                          // Only draw if within bonding cutoff
                drawBond(snap[i].x, snap[i].y, snap[i].z, // Bond from atom i
                         snap[j].x, snap[j].y, snap[j].z); // to atom j
        }

    // --- Panel F: Red force arrows (only when toggled on) ---
    if (show_forces) {                                      // Check if force arrows are enabled
        for (int i = 0; i < NUM_ATOMS; i++) {              // Loop over all atoms
            float fx = snap[i].ax * snap[i].mass;          // Force x = mass × acceleration x (Newton 2nd law)
            float fy = snap[i].ay * snap[i].mass;          // Force y = mass × acceleration y
            float fz = snap[i].az * snap[i].mass;          // Force z = mass × acceleration z
            drawForceArrow(snap[i].x, snap[i].y, snap[i].z, fx, fy, fz); // Draw red arrow for this atom
        }
    }

    glEnable(GL_LIGHTING); // Re-enable lighting before drawing shaded atom spheres

    // --- Draw atom spheres ---
    for (int i = 0; i < NUM_ATOMS; i++) {              // Loop over all atoms
        glColor3f(0.82f, 0.82f, 0.95f);               // Pale blue-white color for hydrogen atoms
        glPushMatrix();                                // Save matrix before atom transform
        glTranslatef(snap[i].x, snap[i].y, snap[i].z); // Translate to atom position
        glutSolidSphere(snap[i].radius, slices, stacks); // Draw shaded sphere with tessellation
        glPopMatrix();                                 // Restore matrix after drawing atom
    }

    drawHUD();         // Render 2D informational overlay on top of the 3D scene
    glutSwapBuffers(); // Swap front/back buffers to display the completed frame (double buffering)
}

// ====================== IDLE CALLBACK ======================
static void idle(void) { glutPostRedisplay(); } // Called when idle: request a new frame (maximizes FPS)

// ====================== ATOM INITIALIZATION ======================
// Places NUM_ATOMS hydrogen atoms at random non-overlapping positions inside the box.
// Assigns Maxwell-Boltzmann velocities at temperature T_target with zero net momentum.
// Computes the initial HF forces so the first MD step has correct accelerations.
void initializePositions() {
    const float kB = 3.1668114e-6f;    // Boltzmann constant in atomic units (Ha/K)
    int placed = 0;                    // Number of atoms successfully placed without overlap

    for (int i = 0; i < NUM_ATOMS; i++) {      // Loop over all atoms to place
        struct Atom *e = &atoms[i];            // Pointer to current atom
        e->mass   = proton_mass;              // Proton mass in a.u. (1836 electron masses)
        e->radius = atom_radius;             // Visual/collision radius for rendering and wall check
        do {                                  // Rejection sampling: retry until no overlap
            e->x = ((float)(rand()%2000)/2000.0f * box_size) - box_size/2; // Random x in [-L/2, L/2]
            e->y = ((float)(rand()%2000)/2000.0f * box_size) - box_size/2; // Random y in [-L/2, L/2]
            e->z = ((float)(rand()%2000)/2000.0f * box_size) - box_size/2; // Random z in [-L/2, L/2]
            int collision = 0;                // Flag: 0 = no overlap with already-placed atoms
            for (int j = 0; j < placed; j++) // Check against all previously placed atoms
                if (atomDistance(*e, atoms[j]) < 1.2f) { collision = 1; break; } // Min spacing: 1.2 bohr
            if (!collision) break;            // Accept position if no overlap found
        } while (1);                          // Retry indefinitely until valid position found
        placed++;                             // Successfully placed one more atom
    }

    // --- Assign Maxwell-Boltzmann initial velocities ---
    for (int i = 0; i < NUM_ATOMS; i++) {              // Loop over all atoms
        struct Atom *e = &atoms[i];                    // Pointer to current atom
        float sigma = sqrtf(kB * T_target / e->mass); // MB standard deviation σ = sqrt(k_B T / m)
        e->vx_old = sigma * (float)gauss_rand();      // v_x ~ N(0, σ²) from Maxwell-Boltzmann distribution
        e->vy_old = sigma * (float)gauss_rand();      // v_y ~ N(0, σ²)
        e->vz_old = sigma * (float)gauss_rand();      // v_z ~ N(0, σ²)
    }

    // --- Remove center-of-mass velocity (zero total momentum) ---
    float v_cm[3] = {0, 0, 0};                        // Center-of-mass velocity accumulator
    for (int i = 0; i < NUM_ATOMS; i++) {              // Accumulate total momentum (mass is equal for all H)
        v_cm[0] += atoms[i].vx_old;                   // Sum x-velocities
        v_cm[1] += atoms[i].vy_old;                   // Sum y-velocities
        v_cm[2] += atoms[i].vz_old;                   // Sum z-velocities
    }
    for (int i = 0; i < NUM_ATOMS; i++) {              // Subtract average velocity from each atom
        atoms[i].vx_old -= v_cm[0] / NUM_ATOMS;       // Remove net x drift
        atoms[i].vy_old -= v_cm[1] / NUM_ATOMS;       // Remove net y drift
        atoms[i].vz_old -= v_cm[2] / NUM_ATOMS;       // Remove net z drift
    }

    // --- Compute initial HF forces (required for first MD step) ---
    compute_hf_accelerations(atoms, NUM_ATOMS, &E_hf_cached); // Full SCF + gradient; also extracts analysis

    // --- Initialize minimum bond distance ---
    min_bond = 1e9f;                               // Start with large sentinel value
    for (int i = 0; i < NUM_ATOMS; i++)            // Loop over all atom pairs
        for (int j = i+1; j < NUM_ATOMS; j++) {    // Unique pairs only
            float d = atomDistance(atoms[i], atoms[j]); // Distance for pair (i,j)
            if (d < min_bond) min_bond = d;        // Update minimum if closer
        }

    // --- Initialize ring buffer with first energy value ---
    energy_history[0] = E_hf_cached;  // First entry in the energy history buffer
    history_count = 1;                // One valid entry in the buffer

    // --- Initialize render buffers with initial state ---
    memcpy(atoms_r, atoms, sizeof(atoms));                     // Render atoms = initial atoms
    E_r    = E_hf_cached;                                     // Render energy = initial energy
    bond_r = min_bond;                                        // Render bond = initial minimum bond
    memcpy(orbital_eps_r, g_orbital_eps, sizeof(g_orbital_eps)); // Render orbital energies
    memcpy(mo_coeff_r,    g_mo_coeff,    sizeof(g_mo_coeff));    // Render MO coefficients
    memcpy(mulliken_q_r,  g_mulliken_q,  sizeof(g_mulliken_q));  // Render Mulliken charges
    E_kin_r = g_E_kin; E_vne_r = g_E_vne;                    // Render energy components
    E_vee_r = g_E_vee; E_nuc_r = g_E_nuc;                    // Render energy components
    nb_r    = g_nb;    occ_r   = g_occ;                       // Render basis metadata
}

// ====================== OPENGL INITIALIZATION ======================
void init_gl(void) {
    glClearColor(0.04f, 0.04f, 0.09f, 0.0f);         // Background: very dark navy blue
    glEnable(GL_DEPTH_TEST);                           // Enable depth buffer for correct 3D occlusion
    glEnable(GL_BLEND);                               // Enable alpha blending for transparent elements
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Standard alpha blend: out = src·α + dst·(1-α)
}

// ====================== WINDOW RESHAPE CALLBACK ======================
void reshape(int w, int h) {                           // Called when the window is resized
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);          // Update viewport to cover entire window
    glMatrixMode(GL_PROJECTION); glLoadIdentity();     // Reset projection matrix
    gluPerspective(60, (GLfloat)w/(GLfloat)h, 0.1, 50.0); // 60° FOV perspective, near=0.1, far=50 bohr
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();      // Reset modelview matrix
}

// ====================== KEYBOARD CALLBACK ======================
void key(unsigned char k, int x, int y) {   // Called by GLUT on every key press
    switch (k) {
        // --- Camera controls ---
        case 'w': obs_z -= 0.2; break;       // W: zoom in (move camera toward scene)
        case 's': obs_z += 0.2; break;       // S: zoom out (move camera away from scene)
        case 'a': obs_x -= 0.2; break;       // A: pan camera left
        case 'd': obs_x += 0.2; break;       // D: pan camera right
        case 'x': obs_y -= 0.2; break;       // X: pan camera down
        case 'z': obs_y += 0.2; break;       // Z: pan camera up

        // --- Thermostat controls ---
        case '+':                              // +: increase target temperature by 50 K
            T_target += 50.0f;
            printf("Target temperature: %.0f K\n", T_target);
            break;
        case '-':                              // -: decrease target temperature by 50 K (minimum 50 K)
            T_target = (T_target > 50) ? T_target - 50 : 50;
            printf("Target temperature: %.0f K\n", T_target);
            break;

        // --- Educational panel toggles (zero extra SCF cost) ---
        case 'o': case 'O':                   // O/o: toggle molecular orbital diagram
            show_orbital = !show_orbital;
            printf("[Panel] Orbital diagram: %s\n", show_orbital ? "ON" : "OFF");
            break;
        case 'm': case 'M':                   // M/m: toggle Mulliken population analysis
            show_mulliken = !show_mulliken;
            printf("[Panel] Mulliken populations: %s\n", show_mulliken ? "ON" : "OFF");
            break;
        case 'e': case 'E':                   // E/e: toggle energy decomposition
            show_energy_dec = !show_energy_dec;
            printf("[Panel] Energy decomposition: %s\n", show_energy_dec ? "ON" : "OFF");
            break;
        case 'f': case 'F':                   // F/f: toggle red force arrows in 3D view
            show_forces = !show_forces;
            printf("[Panel] Force arrows: %s\n", show_forces ? "ON" : "OFF");
            break;

        case 27: exit(0); // ESC: cleanly terminate the program
    }
    glutPostRedisplay(); // Request a new frame after any state change
}

// ====================== LIGHTING PARAMETERS ======================
const GLfloat light_ambient[]   = {0.1f, 0.1f, 0.1f, 1.0f}; // Ambient: dim 10% white background light
const GLfloat light_diffuse[]   = {1, 1, 1, 1};              // Diffuse: full white directional light
const GLfloat light_specular[]  = {1, 1, 1, 1};              // Specular: full white highlight
const GLfloat light_position[]  = {2, 5, 5, 0};              // Directional light (w=0) from upper-right-front

const GLfloat mat_ambient[]     = {0.7f, 0.7f, 0.7f, 1};    // Material ambient: 70% ambient reflectance
const GLfloat mat_diffuse[]     = {0.8f, 0.8f, 0.8f, 1};    // Material diffuse: 80% diffuse reflectance
const GLfloat mat_specular[]    = {1, 1, 1, 1};              // Material specular: full specular reflectance
const GLfloat high_shininess[]  = {100};                     // Shininess 100: tight, glossy specular highlight

// ====================== PROGRAM ENTRY POINT ======================
int main(int argc, char **argv) {
    // --- Parse optional command-line argument: number of atoms ---
    if (argc > 1) {                                          // Check if argument was provided
        int n = atoi(argv[1]);                               // Convert argument string to integer
        printf("Received N = %d\n", n);                     // Echo parsed value to terminal
        if (n % 2 != 0) {                                   // RHF requires even electron count
            fprintf(stderr, "Error: RHF requires an even number of atoms/electrons.\n");
            exit(EXIT_FAILURE);                             // Abort with failure code
        }
        if (n > 0 && n <= MAX_ATOM)                         // Check valid range
            NUM_ATOMS = n;                                  // Override default atom count
        else
            printf("Invalid atom count. Using default = %d\n", NUM_ATOMS);
    }

    srand(time(NULL)); // Seed RNG with current time for different initial conditions each run

    // --- Startup banner ---
    printf("================================================\n");
    printf("  AIMD — Hartree-Fock / STO-3G  (v1.1 Educational)\n");
    printf("  %d hydrogen atom(s) | %d electrons\n", NUM_ATOMS, NUM_ATOMS);
    printf("  Educational panels: O=Orbitals  M=Mulliken  E=Energy  F=Forces\n");
    printf("  Physics runs in a separate pthread thread\n");
    printf("  dt = %.4f a.u. = %.6f fs\n", dt, dt*0.02418884f);
    printf("================================================\n");

    printf("Computing initial HF forces...\n");              // Warn user: first SCF may be slow
    initializePositions();                                   // Place atoms and compute initial HF forces
    printf("Done. E_initial = %+.8f Ha\n\n", E_hf_cached); // Report initial total energy

    // --- Launch physics background thread ---
    pthread_t physics_thread;                               // Thread handle for the MD engine
    pthread_create(&physics_thread, NULL, physics_loop, NULL); // Start physics_loop() in background

    // --- Initialize GLUT and create window ---
    glutInit(&argc, argv);                                  // Initialize GLUT (must be called before any other GLUT)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // Double-buffered RGB window with depth buffer
    glutInitWindowSize(1150, 820);                          // Initial window size in pixels (W × H)
    glutInitWindowPosition(50, 50);                        // Initial window position on screen
    glutCreateWindow("Ab initio MD  -  HF/STO-3G  |  O M E F = Educational Panels"); // Window title

    // --- Configure OpenGL state ---
    init_gl();                                             // Set clear color, depth test, blending
    glutDisplayFunc(display);                              // Register display callback
    glutReshapeFunc(reshape);                              // Register reshape callback
    glutKeyboardFunc(key);                                 // Register keyboard callback
    glutIdleFunc(idle);                                    // Register idle callback (continuous redraw)

    // --- Enable 3D lighting ---
    glEnable(GL_DEPTH_TEST);  glDepthFunc(GL_LESS);        // Draw only closer fragments
    glEnable(GL_LIGHT0);                                   // Enable the single directional light source
    glEnable(GL_NORMALIZE);                                // Auto-normalize normals after scaling
    glEnable(GL_COLOR_MATERIAL);                           // Allow glColor to set material properties
    glEnable(GL_LIGHTING);                                 // Enable OpenGL lighting model

    // --- Set light and material properties ---
    glLightfv(GL_LIGHT0, GL_AMBIENT,   light_ambient);    // Ambient component of light 0
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   light_diffuse);    // Diffuse component of light 0
    glLightfv(GL_LIGHT0, GL_SPECULAR,  light_specular);   // Specular component of light 0
    glLightfv(GL_LIGHT0, GL_POSITION,  light_position);   // Direction/position of light 0

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);    // Front-face ambient material
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);    // Front-face diffuse material
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);   // Front-face specular material
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess); // Front-face shininess exponent

    glutMainLoop(); // Enter GLUT event loop: blocks here forever, dispatching render/input callbacks
    return 0;       // Unreachable; included for C standard compliance
}

