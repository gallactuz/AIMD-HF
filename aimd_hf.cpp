// ============================================================
// Ab initio Molecular Dynamics (AIMD)
// Method: Hartree-Fock (RHF) / STO-3G basis set
// Real-time visualization with OpenGL
//
// Author  : Anderson Aparecido do Espirito Santo
// Date    : 2026
//
// Description:
//   Ab initio molecular dynamics for H_n, He_n and HeH+ systems.
//   Full SCF Hartree-Fock for electronic energy and forces.
//   Educational panels extracted at zero extra SCF cost:
//     O → Molecular orbital diagram (energies + occupancy)
//     M → Mulliken population analysis (gross pop + net charge)
//     E → Energy decomposition (T + V_ne + V_ee + V_nn)
//     F → Red force arrows on each atom (3D)
//     V → 3D semi-transparent orbital volume cloud (new)
//
// Compile:
//   Linux : g++ aimd_hf.cpp -o aimd_hf -lGL -lGLU -lglut -lm -lpthread
//
// Controls:
//   W/S : zoom in/out       A/D : camera left/right
//   Z/X : camera up/down    +/- : temperature ±50 K
//   O   : orbital diagram   M   : Mulliken populations
//   E   : energy decomp     F   : force arrows
//   V   : 3D orbital volume (semi-transparent MO lobes)
//   ESC : exit
//
// Usage:
//   ./aimd_hf h 2          → 2 hydrogen atoms (H2)
//   ./aimd_hf h 4          → 4 hydrogen atoms
//   ./aimd_hf heh+         → 1 HeH+ pair  (2 atoms, 2 e⁻, charge +1)
//   ./aimd_hf heh+ 2       → 2 HeH+ pairs (4 atoms, 4 e⁻, charge +2)
//   ./aimd_hf he 2         → 2 He atom    (4 e⁻, neutral)
//   ./aimd_hf he 4         → 4 He atoms   (8 e⁻, neutral)
// ============================================================

#include <GL/glut.h>   // OpenGL Utility Toolkit — windowing, input, rendering
#include <pthread.h>   // POSIX threads — physics runs on a separate thread
#include <math.h>      // Standard math functions (sqrt, exp, erf, etc.)
#include <stdlib.h>    // Memory allocation, rand(), exit()
#include <stdio.h>     // printf, fprintf, sprintf
#include <string.h>    // memcpy, memset, strcpy, strcmp
#include <time.h>      // time() used to seed the random number generator

// ====================== SIMULATION PARAMETERS ======================
int NUM_ATOMS     = 2;    // Total number of atoms in the simulation (updated from CLI)
int NUM_ELECTRONS = 2;    // Total number of electrons (must be even for closed-shell RHF)
#define dt            2.0f   // MD time step in atomic units (1 a.u. ≈ 0.02419 fs)
#define BOND_CUTOFF   2.5f   // Distance threshold (bohr) below which a bond cylinder is drawn

#define MAX_ATOM      12              // Maximum number of atoms supported
#define NP            3               // Number of Gaussian primitives per STO-3G basis function
#define MAX_BASIS     (MAX_ATOM * NP) // Maximum total number of Gaussian primitives
#define HISTORY_LEN   300             // Number of energy values kept in the rolling history buffer

// ====================== MOLECULE TYPE ======================
//  0 = H_n   (pure hydrogen clusters)
//  1 = HeH+  (helium hydride cation pairs)
//  2 = He_n  (pure helium clusters)
static int molecule_type = 0;    // Default: hydrogen system
static int N_heh_pairs   = 1;    // Number of HeH+ pairs (only relevant when molecule_type == 1)

static int    g_Z[MAX_ATOM];              // Nuclear charge (atomic number) for each atom
static double g_a_exp[MAX_ATOM][NP];      // STO-3G Gaussian exponents for each atom's basis functions
static char   g_atom_symbol[MAX_ATOM][4]; // Chemical symbol string for each atom (e.g., "H", "He")

// STO-3G contraction coefficients (same for all atoms in minimal basis)
static const double STO3G_d[NP]    = {0.15432897, 0.53532814, 0.44463454};
// STO-3G Gaussian exponents for hydrogen (Z=1)
static const double STO3G_H_a[NP]  = {3.42525091, 0.62391373, 0.16885540};
// STO-3G Gaussian exponents for helium (Z=2)
static const double STO3G_He_a[NP] = {6.36242139, 1.15892300, 0.31364979};

// ====================== CAMERA GLOBALS ======================
GLdouble obs_x = 0, obs_y = 0, obs_z = 7.0; // Camera (observer) position in 3D space
static int   slices        = 20;    // Longitude subdivisions for sphere rendering
static int   stacks        = 20;    // Latitude subdivisions for sphere rendering
static float atom_radius   = 0.28f; // Default visual radius for hydrogen atoms
static float proton_mass   = 1836.0f; // Proton mass in atomic units (m_e = 1)
static float box_size      = 5.0f;    // Edge length of the cubic simulation box (bohr)
static float T_target      = 300.0f;  // Target temperature for the Langevin thermostat (K)
static float target_pressure = 0.001f; // Target pressure for the Berendsen barostat (a.u.)

// ====================== ATOM STRUCTURE ======================
struct Atom {
    float x, y, z;               // Current position (bohr)
    float vx_old, vy_old, vz_old; // Velocity at the previous half-step (Velocity Verlet)
    float vx, vy, vz;            // Current full-step velocity (informational)
    float ax, ay, az;            // Current acceleration = Force / mass (bohr/a.u.²)
    float mass, radius;          // Atomic mass (a.u.) and visual sphere radius
};

// ====================== PHYSICS-SIDE BUFFERS ======================
// These variables are written by the physics thread and protected by render_mutex
static struct Atom atoms[MAX_ATOM];     // Live atom states used during MD integration
static double E_hf_cached = 0.0;       // Most recent Hartree-Fock total energy (Hartree)
static float  T_inst      = 0.0f;      // Instantaneous kinetic temperature (K)
static float  P_inst      = 0.0f;      // Instantaneous virial pressure (a.u.)
static float  min_bond    = 0.0f;      // Shortest interatomic distance in current step (bohr)
static double energy_history[HISTORY_LEN]; // Circular buffer of past HF energies for the graph
static int    history_count = 0;        // Total number of energy entries recorded
static int    md_step       = 0;        // Current MD step counter

// Arrays filled during SCF analysis (populated when do_analysis == 1)
static double g_orbital_eps[MAX_BASIS];         // MO orbital energies (Hartree)
static double g_mo_coeff[MAX_BASIS][MAX_BASIS]; // MO coefficient matrix C (AO → MO)
static double g_mulliken_q[MAX_ATOM];           // Mulliken net atomic charges
static double g_E_kin = 0.0;  // Electronic kinetic energy component T (Hartree)
static double g_E_vne = 0.0;  // Nuclear-electron attraction energy V_ne (Hartree)
static double g_E_vee = 0.0;  // Electron-electron repulsion energy V_ee (Hartree)
static double g_E_nuc = 0.0;  // Nuclear-nuclear repulsion energy V_nn (Hartree)
static int    g_nb  = 2;      // Number of basis functions (= NUM_ATOMS for STO-3G s-only)
static int    g_occ = 1;      // Number of occupied MOs (= NUM_ELECTRONS / 2)

// ====================== RENDER-SIDE BUFFERS ======================
// Copies of physics data, safely read by the render thread after mutex lock
static struct Atom atoms_r[MAX_ATOM]; // Snapshot of atom positions for rendering
static double E_r     = 0.0;  // HF energy copy for HUD display
static float  T_r     = 0.0f; // Temperature copy for HUD display
static float  P_r     = 0.0f; // Pressure copy for HUD display
static float  bond_r  = 0.0f; // Minimum bond length copy for HUD display
static int    step_r  = 0;    // MD step counter copy for HUD display
static double ehist_r[HISTORY_LEN]; // Energy history copy for the graph panel
static int    hcount_r = 0;   // Energy history entry count copy
static int    computing = 0;  // Flag: 1 while the SCF is running (shows spinner in HUD)

// Analysis data copies for the educational panels
static double orbital_eps_r[MAX_BASIS];         // Orbital energies for panel O
static double mo_coeff_r[MAX_BASIS][MAX_BASIS]; // MO coefficients for panel O
static double mulliken_q_r[MAX_ATOM];           // Mulliken charges for panel M
static double E_kin_r = 0.0, E_vne_r = 0.0;    // Energy components for panel E
static double E_vee_r = 0.0, E_nuc_r = 0.0;    // Energy components for panel E
static int    nb_r = 2, occ_r = 1;             // Basis/occupancy copies for panel O

// Toggle flags for the educational overlay panels
static int show_orbital    = 0; // 1 = show molecular orbital diagram (key O)
static int show_mulliken   = 0; // 1 = show Mulliken population panel (key M)
static int show_energy_dec = 0; // 1 = show energy decomposition panel (key E)
static int show_forces     = 0; // 1 = draw force arrows on atoms (key F)
static int show_volume     = 0; // 1 = render 3D semi-transparent orbital cloud (key V)

// Mutex protecting all render-side copies above
static pthread_mutex_t render_mutex = PTHREAD_MUTEX_INITIALIZER;

// ====================== BOX-MULLER GAUSSIAN RANDOM ======================
// Returns a single normally distributed random number (mean=0, std=1)
// Uses the Box-Muller transform; caches the second variate for efficiency
double gauss_rand() {
    static int    has_spare = 0; // Flag: a previously generated spare value is available
    static double spare;         // Cached second normal variate from the last call
    if (has_spare) { has_spare = 0; return spare; } // Return the cached spare
    has_spare = 1;
    double u, v, s;
    do {
        // Draw two uniform random numbers in [-1, 1]
        u = (rand()/(double)RAND_MAX)*2.0 - 1.0;
        v = (rand()/(double)RAND_MAX)*2.0 - 1.0;
        s = u*u + v*v; // Squared radius; must be inside unit circle and non-zero
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0*log(s)/s); // Box-Muller scaling factor
    spare = v*s;             // Store second variate for the next call
    return u*s;              // Return the first variate
}

// ====================== HF INTEGRAL HELPERS ======================

// Boys function F0(t) = (1/2) sqrt(π/t) erf(sqrt(t))
// Used in the evaluation of electron repulsion integrals over Gaussians
double F0(double t) {
    if (t < 1e-12) return 1.0; // Limiting value as t → 0
    return 0.5*sqrt(M_PI/t)*erf(sqrt(t));
}

// Normalization constant for a spherical s-type Gaussian with exponent alpha
// N = (2α/π)^(3/4)
double norm_gauss(double alpha) {
    return pow(2.0*alpha/M_PI, 0.75);
}

// Squared Euclidean distance between 3D points A and B
double dist2(const double A[3], const double B[3]) {
    double dx=A[0]-B[0], dy=A[1]-B[1], dz=A[2]-B[2];
    return dx*dx+dy*dy+dz*dz;
}

// Euclidean distance between two Atom structs (float positions)
float atomDistance(struct Atom a, struct Atom b) {
    return sqrtf((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}

// ====================== JACOBI EIGENVALUE DIAGONALIZATION ======================
// Diagonalizes a real symmetric matrix A (n×n) using Jacobi rotations.
// On output: V contains eigenvectors (columns), eig contains eigenvalues.
void jacobi(int n, double A[MAX_BASIS][MAX_BASIS],
            double V[MAX_BASIS][MAX_BASIS], double eig[MAX_BASIS]) {
    // Initialize V as the identity matrix (eigenvector accumulator)
    memset(V, 0, sizeof(double)*MAX_BASIS*MAX_BASIS);
    for (int i = 0; i < n; i++) V[i][i] = 1.0;

    for (int iter = 0; iter < 200; iter++) { // Maximum 200 Jacobi sweeps
        // Find the off-diagonal element with the largest absolute value
        int p=0, q=1;
        double maxval = fabs(A[p][q]);
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                if (fabs(A[i][j]) > maxval) { maxval=fabs(A[i][j]); p=i; q=j; }

        if (maxval < 1e-10) break; // Convergence: all off-diagonal elements are negligible

        // Compute the rotation angle θ that zeros out A[p][q]
        double theta = 0.5*atan2(2.0*A[p][q], A[q][q]-A[p][p]);
        double c=cos(theta), s=sin(theta); // Cosine and sine of the rotation angle

        // Update the two diagonal elements after the Jacobi rotation
        double App = c*c*A[p][p]-2*s*c*A[p][q]+s*s*A[q][q];
        double Aqq = s*s*A[p][p]+2*s*c*A[p][q]+c*c*A[q][q];
        A[p][p]=App; A[q][q]=Aqq; A[p][q]=A[q][p]=0.0; // Zero the pivot element

        // Update remaining rows/columns affected by the rotation
        for (int k = 0; k < n; k++) {
            if (k==p||k==q) continue; // Skip the pivot rows
            double Akp=c*A[k][p]-s*A[k][q], Akq=s*A[k][p]+c*A[k][q];
            A[k][p]=A[p][k]=Akp; A[k][q]=A[q][k]=Akq;
        }
        // Accumulate the rotation into the eigenvector matrix V
        for (int k = 0; k < n; k++) {
            double vkp=c*V[k][p]-s*V[k][q], vkq=s*V[k][p]+c*V[k][q];
            V[k][p]=vkp; V[k][q]=vkq;
        }
    }
    // Extract eigenvalues from the (now diagonal) matrix A
    for (int i = 0; i < n; i++) eig[i]=A[i][i];
}

// ====================== HARTREE-FOCK SCF ENERGY ======================
// Computes the RHF total energy for the given atomic configuration.
// If do_analysis == 1, also fills the educational analysis arrays (MOs, Mulliken, etc.).
double compute_hf_energy(const double R_atom_in[MAX_ATOM][3],
                         int NB, int do_analysis) {
    // Local copy of atomic positions to avoid aliasing issues
    double R_atom[MAX_ATOM][3];
    for (int i = 0; i < NB; i++)
        for (int k = 0; k < 3; k++)
            R_atom[i][k] = R_atom_in[i][k];

    // STO-3G contraction coefficients (shared across all atom types)
    double d_coef[NP];
    for (int p = 0; p < NP; p++) d_coef[p] = STO3G_d[p];

    // Flatten basis: index primitive gaussians as (atom * NP + primitive)
    double a_prim[MAX_BASIS];        // Exponent of each primitive Gaussian
    double R_prim[MAX_BASIS][3];     // Center of each primitive Gaussian (= its atom center)
    for (int A = 0; A < NB; A++)
        for (int p = 0; p < NP; p++) {
            int idx = A*NP+p;
            a_prim[idx] = g_a_exp[A][p]; // Copy atom's STO-3G exponent
            for (int k = 0; k < 3; k++) R_prim[idx][k] = R_atom[A][k]; // Assign atom center
        }

    // ---- One-electron integrals ----
    double S[MAX_BASIS][MAX_BASIS]     = {{0}}; // Overlap matrix S_ij = <i|j>
    double T_kin[MAX_BASIS][MAX_BASIS] = {{0}}; // Kinetic energy matrix T_ij = <i|-½∇²|j>
    double Vne[MAX_BASIS][MAX_BASIS]   = {{0}}; // Nuclear attraction matrix V_ij = <i|Σ -Z/r|j>
    double Hcore[MAX_BASIS][MAX_BASIS];          // Core Hamiltonian H = T + V_ne

    for (int i = 0; i < NB; i++)
        for (int j = i; j < NB; j++) { // Exploit symmetry: only compute upper triangle
            for (int p = 0; p < NP; p++)
                for (int q = 0; q < NP; q++) {
                    int ip=i*NP+p, jq=j*NP+q; // Flat primitive indices
                    double al1=a_prim[ip], al2=a_prim[jq]; // Exponents of the two primitives
                    double N1=norm_gauss(al1), N2=norm_gauss(al2); // Normalization constants
                    double gam=al1+al2;           // Sum of exponents (combined exponent)
                    double ratio=al1*al2/gam;     // Product / sum — appears in overlap formula
                    double Rab2=dist2(R_prim[ip], R_prim[jq]); // Squared distance between centers
                    double Rp[3]; // Gaussian product center (weighted average of the two centers)
                    for (int k = 0; k < 3; k++)
                        Rp[k]=(al1*R_prim[ip][k]+al2*R_prim[jq][k])/gam;

                    // Overlap integral between two s-type Gaussians
                    double Sij=pow(M_PI/gam,1.5)*exp(-ratio*Rab2);
                    // Kinetic energy integral via the Gaussian kinetic energy formula
                    double Tij=ratio*(3.0-2.0*ratio*Rab2)*Sij;

                    // Nuclear attraction integral: sum over all nuclei
                    double Vij=0.0;
                    for (int A2 = 0; A2 < NB; A2++) {
                        double rPA2=dist2(Rp, R_atom[A2]); // Squared dist from product center to nucleus A2
                        // Contribution of nucleus A2 with charge Z[A2] via the Boys function
                        Vij -= (double)g_Z[A2]*(2.0*M_PI/gam)*exp(-ratio*Rab2)*F0(gam*rPA2);
                    }

                    // Accumulate with contraction coefficients and normalization
                    double cc=d_coef[p]*d_coef[q]*N1*N2;
                    S[i][j]     += cc*Sij;
                    T_kin[i][j] += cc*Tij;
                    Vne[i][j]   += cc*Vij;
                }
            // Copy upper triangle to lower triangle (symmetric matrices)
            S[j][i]     = S[i][j];
            T_kin[j][i] = T_kin[i][j];
            Vne[j][i]   = Vne[i][j];
        }

    // Build core Hamiltonian: H_core = T + V_ne
    for (int i = 0; i < NB; i++)
        for (int j = 0; j < NB; j++)
            Hcore[i][j] = T_kin[i][j]+Vne[i][j];

    // ---- Nuclear repulsion energy (classical, no electron involvement) ----
    double E_nuc = 0.0;
    for (int A = 0; A < NB; A++)
        for (int B = A+1; B < NB; B++) // Sum over unique pairs
            E_nuc += (double)(g_Z[A]*g_Z[B]) / sqrt(dist2(R_atom[A], R_atom[B])); // Z_A*Z_B / R_AB

    // ---- Two-electron repulsion integrals (i j | k l) ----
    // (ij|kl) = ∫∫ φ_i(r1) φ_j(r1) (1/r12) φ_k(r2) φ_l(r2) dr1 dr2
    static double eri[MAX_BASIS][MAX_BASIS][MAX_BASIS][MAX_BASIS];
    memset(eri, 0, sizeof(eri)); // Zero out the ERI tensor
    for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
    for (int k=0;k<NB;k++) for (int l=0;l<NB;l++) {
        double val=0.0;
        for (int p=0;p<NP;p++) for (int q=0;q<NP;q++)
        for (int r=0;r<NP;r++) for (int s=0;s<NP;s++) {
            int ip=i*NP+p,jq=j*NP+q,kr=k*NP+r,ls=l*NP+s; // Flat primitive indices
            double a1=a_prim[ip],a2=a_prim[jq],a3=a_prim[kr],a4=a_prim[ls]; // Exponents
            double N1=norm_gauss(a1),N2=norm_gauss(a2),N3=norm_gauss(a3),N4=norm_gauss(a4);
            double g1=a1+a3, g2=a2+a4, g=g1+g2; // Combined exponents for each electron pair
            double Rp2[3],Rq2[3]; // Product centers for electron 1 (Rp2) and electron 2 (Rq2)
            for (int dd=0;dd<3;dd++) {
                Rp2[dd]=(a1*R_prim[ip][dd]+a3*R_prim[kr][dd])/g1; // Product center P
                Rq2[dd]=(a2*R_prim[jq][dd]+a4*R_prim[ls][dd])/g2; // Product center Q
            }
            double rab2=dist2(R_prim[ip],R_prim[kr]); // |r_i - r_k|^2
            double rcd2=dist2(R_prim[jq],R_prim[ls]); // |r_j - r_l|^2
            double rpq2=dist2(Rp2,Rq2);               // |P - Q|^2 (inter-center distance)
            double pref=2.0*pow(M_PI,2.5)/(g1*g2*sqrt(g)); // Prefactor from the ERI formula
            double boys_arg=(g1*g2/g)*rpq2; // Argument of the Boys function
            // Accumulate primitive contribution with contraction coefficients
            val+=d_coef[p]*d_coef[q]*d_coef[r]*d_coef[s]*N1*N2*N3*N4*pref
                *exp(-a1*a3/g1*rab2-a2*a4/g2*rcd2)*F0(boys_arg);
        }
        eri[i][j][k][l]=val; // Store contracted ERI element
    }

    // ---- Löwdin orthogonalization: compute X = S^{-1/2} ----
    // This transforms the generalized eigenvalue problem F C = S C ε into standard form
    double Scopy[MAX_BASIS][MAX_BASIS], U[MAX_BASIS][MAX_BASIS], eigS[MAX_BASIS];
    memcpy(Scopy, S, sizeof(S));         // Copy S before diagonalization (jacobi modifies in place)
    jacobi(NB, Scopy, U, eigS);          // Diagonalize S → eigenvectors U, eigenvalues eigS

    // Build X = U * diag(1/sqrt(eigS)) * U^T (the symmetric orthogonalizer)
    double X[MAX_BASIS][MAX_BASIS]={{0}};
    for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
        for (int k=0;k<NB;k++)
            if (eigS[k]>1e-10) // Skip near-zero eigenvalues (linear dependence)
                X[i][j]+=U[i][k]*(1.0/sqrt(eigS[k]))*U[j][k];

    // ---- SCF loop (Self-Consistent Field iteration) ----
    double P[MAX_BASIS][MAX_BASIS]={{0}};  // Density matrix P_ij = 2 Σ_occ C_ia C_ja
    double E_old=0.0, damp=0.6, Etot=0.0; // Previous energy, DIIS-like damping factor, total energy
    int occ     = NUM_ELECTRONS/2;         // Number of doubly-occupied MOs
    int max_iter = 120, iter;              // SCF iteration limit and counter
    // Final converged quantities (updated each iteration, kept after convergence)
    double C_final[MAX_BASIS][MAX_BASIS]={{0}};
    double eps_final[MAX_BASIS]={0};
    double P_final[MAX_BASIS][MAX_BASIS]={{0}};
    double G_final[MAX_BASIS][MAX_BASIS]={{0}};

    for (iter=0; iter<max_iter; iter++) {
        // Build the two-electron (Coulomb + exchange) contribution G from density matrix P
        double G[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int p2=0;p2<NB;p2++) for (int q=0;q<NB;q++)
                // G_ij = Σ P_pq [ (iq|jp) - 0.5*(iq|pj) ]  (Coulomb minus half Exchange)
                G[i][j]+=P[p2][q]*(eri[i][q][j][p2]-0.5*eri[i][q][p2][j]);

        // Fock matrix F = H_core + G
        double F_mat[MAX_BASIS][MAX_BASIS];
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            F_mat[i][j]=Hcore[i][j]+G[i][j];

        // Transform Fock matrix to orthogonal basis: F' = X^T F X
        double Fp[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int k=0;k<NB;k++) for (int l=0;l<NB;l++)
                Fp[i][j]+=X[k][i]*F_mat[k][l]*X[l][j];

        // Diagonalize F' to get MO coefficients C' and orbital energies ε
        double Acopy[MAX_BASIS][MAX_BASIS], Cp[MAX_BASIS][MAX_BASIS], eps[MAX_BASIS];
        memcpy(Acopy, Fp, sizeof(Fp));
        jacobi(NB, Acopy, Cp, eps); // Solve F' C' = C' diag(ε)

        // Sort MOs by ascending orbital energy (bubble sort on eigenvalues)
        for (int i=0;i<NB-1;i++) for (int jj=i+1;jj<NB;jj++)
            if (eps[jj]<eps[i]) {
                double tmp=eps[i]; eps[i]=eps[jj]; eps[jj]=tmp;
                for (int k=0;k<NB;k++) { tmp=Cp[k][i]; Cp[k][i]=Cp[k][jj]; Cp[k][jj]=tmp; }
            }

        // Back-transform MO coefficients to AO basis: C = X * C'
        double C[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int k=0;k<NB;k++) C[i][j]+=X[i][k]*Cp[k][j];

        // Build new density matrix from occupied MOs: P_new = 2 * C_occ * C_occ^T
        double Pnew[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int k=0;k<occ;k++) Pnew[i][j]+=2.0*C[i][k]*C[j][k];

        // Compute electronic energy: E_elec = 0.5 * Tr[P (H_core + F)]
        double Eelec=0.0;
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            Eelec+=0.5*Pnew[i][j]*(Hcore[i][j]+F_mat[i][j]);
        Etot=Eelec+E_nuc; // Total energy = electronic + nuclear repulsion

        // Convergence check: energy change and density matrix change
        double deltaE=fabs(Etot-E_old), deltaP=0.0;
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            deltaP+=fabs(Pnew[i][j]-P[i][j]); // Sum of absolute density matrix changes

        if (deltaE<1e-8 && deltaP<1e-6) { // SCF converged
            memcpy(C_final,   C,    sizeof(C));
            memcpy(eps_final, eps,  sizeof(eps));
            memcpy(P_final,   Pnew, sizeof(Pnew));
            memcpy(G_final,   G,    sizeof(G));
            break; // Exit the SCF loop
        }

        // Adaptive damping: increase mixing of old density to improve convergence
        if (iter>8)  damp=0.35; // Increase damping after 8 iterations
        if (iter>20) damp=0.20; // Further increase damping after 20 iterations

        // Damp density matrix: P = (1-α)*P_old + α*P_new
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            P[i][j]=(1.0-damp)*P[i][j]+damp*Pnew[i][j];
        E_old=Etot; // Update previous energy for next convergence check

        // Keep final arrays up to date (used if max_iter is reached without convergence)
        memcpy(C_final,   C,    sizeof(C));
        memcpy(eps_final, eps,  sizeof(eps));
        memcpy(P_final,   Pnew, sizeof(Pnew));
        memcpy(G_final,   G,    sizeof(G));
    }

    // Print convergence status (only once per MD step to avoid log flooding)
    static int last_step_reported=-1;
    if (md_step!=last_step_reported) {
        if (iter>=max_iter-1)
            printf("Warning: SCF did not converge after %d iterations\n", max_iter);
        else if (iter>30)
            printf("SCF converged in %d iterations\n", iter+1);
        last_step_reported=md_step;
    }

    // ====================== EDUCATIONAL ANALYSIS ======================
    // Populated only when do_analysis == 1 (main energy call, not finite-difference calls)
    if (do_analysis) {
        g_nb=NB; g_occ=occ; // Store basis size and occupancy for the panels

        // Copy orbital energies and MO coefficients for panel O and volume renderer
        for (int i=0;i<NB;i++) {
            g_orbital_eps[i]=eps_final[i];
            for (int j=0;j<NB;j++) g_mo_coeff[i][j]=C_final[i][j];
        }

        // Mulliken gross population for atom A: q_A = Z_A - Σ_ν P_Aν * S_νA
        for (int A=0;A<NB;A++) {
            double gross=0.0;
            for (int nu=0;nu<NB;nu++) gross+=P_final[A][nu]*S[nu][A]; // Gross population
            g_mulliken_q[A]=(double)g_Z[A]-gross; // Net charge = nuclear charge minus gross pop
        }

        // Energy decomposition: kinetic, nuclear attraction, electron repulsion
        double Ekin_comp=0.0, Evne_comp=0.0, Evee_comp=0.0;
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++) {
            Ekin_comp+=P_final[i][j]*T_kin[i][j]; // T = Tr[P * T_kin]
            Evne_comp+=P_final[i][j]*Vne[i][j];   // V_ne = Tr[P * V_ne]
            Evee_comp+=0.5*P_final[i][j]*G_final[i][j]; // V_ee = 0.5 * Tr[P * G]
        }
        g_E_kin=Ekin_comp; g_E_vne=Evne_comp;
        g_E_vee=Evee_comp; g_E_nuc=E_nuc; // Store all four components globally
    }

    return Etot; // Return total Hartree-Fock energy in Hartree
}

// ====================== HF FORCES VIA FINITE DIFFERENCE ======================
// Computes atomic accelerations using central finite differences on the HF energy.
// Also stores the energy from the central (unperturbed) point in *E_out.
void compute_hf_accelerations(struct Atom *atoms_in, int NB, double *E_out) {
    // Extract current atomic positions into a plain array
    double R[MAX_ATOM][3];
    for (int i=0;i<NB;i++) {
        R[i][0]=atoms_in[i].x; R[i][1]=atoms_in[i].y; R[i][2]=atoms_in[i].z;
    }
    const double delta=0.001; // Finite-difference step size (bohr)

    // Compute energy at the current geometry (with full analysis)
    *E_out=compute_hf_energy(R, NB, 1);

    // Loop over every atom and every Cartesian direction
    for (int i=0;i<NB;i++) {
        double Rs[3]={R[i][0],R[i][1],R[i][2]}, Ep, Em; // Save original position

        // Force along x: F_x = -(E(x+δ) - E(x-δ)) / (2δ)
        R[i][0]=Rs[0]+delta; Ep=compute_hf_energy(R,NB,0); // Forward displacement
        R[i][0]=Rs[0]-delta; Em=compute_hf_energy(R,NB,0); // Backward displacement
        R[i][0]=Rs[0]; atoms_in[i].ax=-(Ep-Em)/(2.0*delta)/atoms_in[i].mass; // a = F/m

        // Force along y
        R[i][1]=Rs[1]+delta; Ep=compute_hf_energy(R,NB,0);
        R[i][1]=Rs[1]-delta; Em=compute_hf_energy(R,NB,0);
        R[i][1]=Rs[1]; atoms_in[i].ay=-(Ep-Em)/(2.0*delta)/atoms_in[i].mass;

        // Force along z
        R[i][2]=Rs[2]+delta; Ep=compute_hf_energy(R,NB,0);
        R[i][2]=Rs[2]-delta; Em=compute_hf_energy(R,NB,0);
        R[i][2]=Rs[2]; atoms_in[i].az=-(Ep-Em)/(2.0*delta)/atoms_in[i].mass;
    }
}

// ====================== BOUNDARY CONDITIONS ======================
// Reflects atoms off the walls of the cubic simulation box (elastic collision).
void applyBoundaryConditions(struct Atom *e) {
    float h=box_size/2.0f; // Half-length of the box edge

    // Check each face of the box and reflect velocity if the atom crosses it
    if (e->x+e->radius> h){e->x= h-e->radius; e->vx_old=-e->vx_old;} // +x wall
    if (e->x-e->radius<-h){e->x=-h+e->radius; e->vx_old=-e->vx_old;} // -x wall
    if (e->y+e->radius> h){e->y= h-e->radius; e->vy_old=-e->vy_old;} // +y wall
    if (e->y-e->radius<-h){e->y=-h+e->radius; e->vy_old=-e->vy_old;} // -y wall
    if (e->z+e->radius> h){e->z= h-e->radius; e->vz_old=-e->vz_old;} // +z wall
    if (e->z-e->radius<-h){e->z=-h+e->radius; e->vz_old=-e->vz_old;} // -z wall
}

// ====================== LANGEVIN THERMOSTAT ======================
// Applies a Langevin thermostat: scales velocities with exponential damping
// and adds Gaussian random kicks to maintain the target temperature T_desired.
// Returns the instantaneous kinetic temperature before the thermostat is applied.
float controlTemperature(float T_desired) {
    const float kB=3.1668114e-6f; // Boltzmann constant in atomic units (Hartree/K)

    // Compute instantaneous kinetic energy K = Σ ½mv²
    float K=0.0f;
    for (int i=0;i<NUM_ATOMS;i++) {
        struct Atom *e=&atoms[i];
        K+=0.5f*e->mass*(e->vx_old*e->vx_old+e->vy_old*e->vy_old+e->vz_old*e->vz_old);
    }
    // Instantaneous temperature from equipartition: T = 2K / (3 N kB)
    float T_inst_local=(2.0f*K)/(3.0f*NUM_ATOMS*kB);

    // Langevin coefficients: c1 = e^(-γ dt), c2 = sqrt((1-c1²) kB T / m)
    float c1=expf(-0.1f*dt);                        // Damping coefficient (friction)
    float c2b=sqrtf((1.0f-c1*c1)*kB*T_desired);     // Noise amplitude (before mass scaling)

    // Apply Langevin dynamics: v_new = c1*v + c2 * η  (η ~ N(0,1))
    for (int i=0;i<NUM_ATOMS;i++) {
        struct Atom *e=&atoms[i];
        float c2=c2b/sqrtf(e->mass); // Scale noise by 1/sqrt(m)
        e->vx_old=c1*e->vx_old+c2*(float)gauss_rand(); // Damp + random kick in x
        e->vy_old=c1*e->vy_old+c2*(float)gauss_rand(); // Damp + random kick in y
        e->vz_old=c1*e->vz_old+c2*(float)gauss_rand(); // Damp + random kick in z
    }
    return T_inst_local; // Return pre-thermostat temperature for logging/display
}

// ====================== VIRIAL PRESSURE ======================
// Estimates the instantaneous pressure using the virial theorem:
// P = (2K + W) / (3V),  where W = Σ r_i · F_i is the virial
float computePressure() {
    float K=0.0f, virial=0.0f;
    for (int i=0;i<NUM_ATOMS;i++) {
        struct Atom *e=&atoms[i];
        K+=0.5f*e->mass*(e->vx_old*e->vx_old+e->vy_old*e->vy_old+e->vz_old*e->vz_old); // Kinetic energy
        virial+=e->x*e->mass*e->ax+e->y*e->mass*e->ay+e->z*e->mass*e->az; // Virial W = r · F
    }
    float V=box_size*box_size*box_size; // Volume of the cubic simulation box
    return (2.0f*K+virial)/(3.0f*V);   // Pressure from the virial theorem
}

// ====================== BERENDSEN BAROSTAT ======================
// Rescales the box size and all atomic positions to drive pressure toward target.
// Uses Berendsen coupling: μ = 1 - β_T * (dt/τ_P) * (P_target - P_current)
void rescaleBox(float P_current) {
    float scale=1.0f-0.0005f*(dt/5.0f)*(target_pressure-P_current); // Scaling factor μ
    box_size*=scale; // Resize the simulation box
    for (int i=0;i<NUM_ATOMS;i++) { atoms[i].x*=scale; atoms[i].y*=scale; atoms[i].z*=scale; } // Scale positions
}

// ====================== MD STEP ======================
// Performs one complete Velocity Verlet MD step:
//   1. Half-step velocity update
//   2. Langevin thermostat
//   3. Full-step position update + boundary conditions
//   4. HF energy and force evaluation
//   5. Second half-step velocity update
//   6. Pressure calculation + Berendsen barostat
//   7. Bookkeeping (bond lengths, energy history, console output)
void stepMD() {
    // --- First half-step: v(t + dt/2) = v(t) + a(t)*dt/2 ---
    for (int i=0;i<NUM_ATOMS;i++) {
        atoms[i].vx_old+=0.5f*atoms[i].ax*dt;
        atoms[i].vy_old+=0.5f*atoms[i].ay*dt;
        atoms[i].vz_old+=0.5f*atoms[i].az*dt;
    }

    T_inst=controlTemperature(T_target); // Apply Langevin thermostat and get current T

    // --- Full-step position update: r(t+dt) = r(t) + v(t+dt/2)*dt ---
    for (int i=0;i<NUM_ATOMS;i++) {
        atoms[i].x+=atoms[i].vx_old*dt;
        atoms[i].y+=atoms[i].vy_old*dt;
        atoms[i].z+=atoms[i].vz_old*dt;
        applyBoundaryConditions(&atoms[i]); // Reflect atoms that left the box
    }

    // --- Evaluate HF energy and forces at the new positions ---
    compute_hf_accelerations(atoms, NUM_ATOMS, &E_hf_cached);

    // --- Second half-step: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2 ---
    for (int i=0;i<NUM_ATOMS;i++) {
        atoms[i].vx_old+=0.5f*atoms[i].ax*dt;
        atoms[i].vy_old+=0.5f*atoms[i].ay*dt;
        atoms[i].vz_old+=0.5f*atoms[i].az*dt;
    }

    P_inst=computePressure(); // Compute virial pressure
    rescaleBox(P_inst);       // Apply Berendsen barostat to maintain target pressure

    // Find the shortest interatomic distance (for bond display and logging)
    min_bond=1e9f;
    for (int i=0;i<NUM_ATOMS;i++)
        for (int j=i+1;j<NUM_ATOMS;j++) {
            float d=atomDistance(atoms[i],atoms[j]);
            if (d<min_bond) min_bond=d;
        }

    // Store current energy in the rolling circular buffer
    energy_history[history_count%HISTORY_LEN]=E_hf_cached;
    history_count++; md_step++; // Advance counters

    // Print detailed step summary to the console
    printf("\n=== STEP %d (t=%.4f fs) ===\n", md_step, md_step*dt*0.02418884f);
    printf("  E(HF)    : %+.8f Ha  (%+.4f eV)\n", E_hf_cached, E_hf_cached*27.2114);
    printf("  E_kin    : %+.6f Ha   E_Vne : %+.6f Ha\n", g_E_kin, g_E_vne);
    printf("  E_Vee    : %+.6f Ha   E_Vnn : %+.6f Ha\n", g_E_vee, g_E_nuc);
    printf("  T_inst   : %.2f K  (target: %.0f K)\n", T_inst, T_target);
    printf("  Pressure : %+.6f a.u.\n", P_inst);
    printf("  d_min    : %.4f bohr = %.4f Ang\n", min_bond, min_bond*0.529177f);
    for (int i=0;i<NUM_ATOMS;i++)
        printf("  Mulliken q[%s%d] : %+.4f e\n", g_atom_symbol[i], i+1, g_mulliken_q[i]);
}

// ====================== PHYSICS THREAD ======================
// Entry point for the background physics thread.
// Runs the MD loop continuously, updating physics state and copying to render buffers.
void* physics_loop(void* arg) {
    (void)arg; // Suppress unused parameter warning
    while (1) {
        // Signal to the render thread that computation is in progress
        pthread_mutex_lock(&render_mutex); computing=1; pthread_mutex_unlock(&render_mutex);

        stepMD(); // Execute one complete MD step (blocking, takes significant time)

        // Lock the mutex and copy all physics data to the render-side buffers
        pthread_mutex_lock(&render_mutex);
        memcpy(atoms_r, atoms, sizeof(atoms));  // Copy atom states
        E_r=E_hf_cached; T_r=T_inst; P_r=P_inst; bond_r=min_bond; step_r=md_step;
        memcpy(ehist_r, energy_history, sizeof(energy_history)); hcount_r=history_count;
        memcpy(orbital_eps_r, g_orbital_eps, sizeof(g_orbital_eps)); // Copy MO energies
        memcpy(mo_coeff_r,    g_mo_coeff,    sizeof(g_mo_coeff));    // Copy MO coefficients
        memcpy(mulliken_q_r,  g_mulliken_q,  sizeof(g_mulliken_q));  // Copy Mulliken charges
        E_kin_r=g_E_kin; E_vne_r=g_E_vne; E_vee_r=g_E_vee; E_nuc_r=g_E_nuc; // Copy energy components
        nb_r=g_nb; occ_r=g_occ; computing=0; // Copy basis/occupancy info and clear computing flag
        pthread_mutex_unlock(&render_mutex);
    }
    return NULL;
}

// ====================== 2D TEXT HELPER ======================
// Draws a string at the given 2D screen-space coordinates using a GLUT bitmap font.
void drawText(float x, float y, const char *str, void *font) {
    glRasterPos2f(x, y);                           // Set raster position for text
    for (int i=0; str[i]; i++) glutBitmapCharacter(font, str[i]); // Draw each character
}

// ====================== BOND CYLINDER ======================
// Draws a cylinder between two 3D points to represent a chemical bond.
void drawBond(float x1,float y1,float z1, float x2,float y2,float z2) {
    float dx=x2-x1,dy=y2-y1,dz=z2-z1;           // Bond vector
    float len=sqrtf(dx*dx+dy*dy+dz*dz);           // Bond length
    if (len<1e-4f) return;                         // Skip degenerate (zero-length) bonds

    // Compute rotation axis and angle to align the cylinder with the bond vector
    float rot_ax=-dy, rot_ay=dx;
    float angle=(180.0f/M_PI)*acosf(dz/len);
    if (fabsf(dz/len)>0.9999f){rot_ax=1;rot_ay=0;angle=(dz<0)?180.0f:0.0f;} // Handle vertical bond edge case

    glPushMatrix();
    glTranslatef(x1,y1,z1);             // Move to the start atom
    glRotatef(angle,rot_ax,rot_ay,0.0f); // Rotate to align with bond direction
    GLUquadric *q=gluNewQuadric();
    glColor4f(0.95f,0.85f,0.1f,0.85f);  // Golden-yellow color for bond
    gluCylinder(q,0.07,0.07,len,14,1);  // Draw cylinder with radius 0.07, length = bond length
    gluDeleteQuadric(q);
    glPopMatrix();
}

// ====================== FORCE ARROWS ======================
// Draws a 3D arrow from an atom's position in the direction of the force vector.
void drawForceArrow(float x,float y,float z,float fx,float fy,float fz) {
    float flen=sqrtf(fx*fx+fy*fy+fz*fz);   // Magnitude of the force
    if (flen<1e-14f) return;                // Skip negligible forces

    float vscale=300.0f; // Scale factor to convert force magnitude into visible arrow length
    float ex=x+fx*vscale,ey=y+fy*vscale,ez=z+fz*vscale; // Arrow tip position

    float alen=sqrtf((ex-x)*(ex-x)+(ey-y)*(ey-y)+(ez-z)*(ez-z)); // Arrow shaft length
    if (alen<0.02f) return; // Skip if arrow is too short to be visible

    glDisable(GL_LIGHTING); // Disable lighting so the arrow color is unaffected

    // Draw the arrow shaft as a red line
    glColor3f(1.0f,0.12f,0.12f); glLineWidth(2.8f);
    glBegin(GL_LINES); glVertex3f(x,y,z); glVertex3f(ex,ey,ez); glEnd();
    glLineWidth(1.0f);

    // Compute rotation to align the arrowhead cone with the force direction
    float dx=ex-x,dy2=ey-y,dz2=ez-z;
    float norm=sqrtf(dx*dx+dy2*dy2+dz2*dz2);
    float axr=-dy2,ayr=dx,ang=(180.0f/M_PI)*acosf(dz2/norm);
    if (fabsf(dz2/norm)>0.9999f){axr=1;ayr=0;ang=(dz2<0)?180.0f:0.0f;} // Edge case

    // Draw the arrowhead as a cone at the tip
    glPushMatrix(); glTranslatef(ex,ey,ez); glRotatef(ang,axr,ayr,0.0f);
    GLUquadric *q=gluNewQuadric();
    glColor3f(1.0f,0.25f,0.1f);
    gluCylinder(q,0.09f,0.0f,0.22f,12,1); // Cone: base radius 0.09, apex at top
    gluDeleteQuadric(q); glPopMatrix();
    glEnable(GL_LIGHTING); // Re-enable lighting for subsequent geometry
}

// ====================== ORBITAL VOLUME RENDERER (KEY V) ======================
// Renders all occupied molecular orbitals as semi-transparent 3D point clouds
// directly in the 3D viewport alongside the atoms.
//
// Algorithm:
//   1. Build a 20×20×20 regular grid centred on the molecule.
//   2. At every grid point r, evaluate ψ_k(r) = Σ_A C[A][k] · χ_A(r)
//      where χ_A is the contracted STO-3G s-type Gaussian centred on atom A.
//   3. Colour each point by the sign of ψ (positive lobe = cool, negative = warm)
//      and set alpha proportional to |ψ|² so dense regions appear more opaque.
//   4. Depth writes are disabled (glDepthMask GL_FALSE) so the cloud never
//      occludes the atom spheres that were already written to the depth buffer.
//
// Cost: 20³ × occ × nb × NP  evaluations of exp() per frame.
//   H₂ (occ=1, nb=2, NP=3): ≈ 24 000 exp() calls — negligible.
//   6 occ MOs, 12 atoms    : ≈ 1.7 M exp() calls — still real-time on modern HW.
void drawOrbitalVolume() {
    int nb  = nb_r;   // Number of contracted basis functions (one per atom)
    int occ = occ_r;  // Number of doubly-occupied molecular orbitals
    if (nb < 1 || occ < 1) return; // Guard: no SCF data available yet

    // Compute the geometric centroid of the molecule from render-side atom positions
    float cx=0.0f, cy=0.0f, cz=0.0f;
    for (int i=0; i<nb; i++) {
        cx += atoms_r[i].x; // Accumulate x coordinate of atom i
        cy += atoms_r[i].y; // Accumulate y coordinate of atom i
        cz += atoms_r[i].z; // Accumulate z coordinate of atom i
    }
    cx/=nb; cy/=nb; cz/=nb; // Divide by atom count to obtain centroid

    // Grid parameters: 20×20×20 = 8 000 sample points per frame
    const int   NG   = 20;    // Number of grid points per Cartesian dimension
    const float GEXT = 2.8f;  // Half-extent of the evaluation grid in bohr (grid spans ±GEXT)
    float step = 2.0f*GEXT/(NG-1); // Uniform spacing between adjacent grid points (bohr)

    // Per-MO colour palette: [pos_R, pos_G, pos_B, neg_R, neg_G, neg_B]
    // Positive lobe (ψ > 0) → cool colours; negative lobe (ψ < 0) → warm colours
    static const float mo_col[6][6] = {
        {0.20f,0.45f,1.00f,  1.00f,0.25f,0.20f}, // MO 1: blue / red
        {0.20f,0.90f,0.30f,  1.00f,0.65f,0.10f}, // MO 2: green / orange
        {0.10f,0.85f,0.95f,  0.90f,0.20f,0.90f}, // MO 3: cyan / magenta
        {1.00f,0.95f,0.20f,  0.60f,0.20f,0.90f}, // MO 4: yellow / purple
        {1.00f,1.00f,1.00f,  0.50f,0.50f,0.55f}, // MO 5: white / grey
        {0.20f,0.85f,0.75f,  1.00f,0.45f,0.65f}, // MO 6: teal / pink
    };

    glDisable(GL_LIGHTING);                        // Cloud is self-coloured, not lit by the scene light
    glEnable(GL_BLEND);                            // Enable alpha-blending for translucency
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); // Standard over-compositing blend equation
    glDepthMask(GL_FALSE);                         // Disable depth writes so cloud never occludes atoms
    glPointSize(6.0f);                             // Larger point sprites give a smoother cloud look

    glBegin(GL_POINTS); // Batch all cloud vertices in a single draw call for efficiency

    for (int ix=0; ix<NG; ix++) {
        float rx = cx + (-GEXT + ix*step); // X coordinate of this grid point (world space)
        for (int iy=0; iy<NG; iy++) {
            float ry = cy + (-GEXT + iy*step); // Y coordinate of this grid point
            for (int iz=0; iz<NG; iz++) {
                float rz = cz + (-GEXT + iz*step); // Z coordinate of this grid point

                // Evaluate every occupied MO at this single grid point
                for (int k=0; k<occ; k++) {

                    // ψ_k(r) = Σ_A C[A][k] · χ_A(r)
                    double psi = 0.0; // Accumulated MO amplitude at grid point r
                    for (int A=0; A<nb; A++) {
                        double cAk = mo_coeff_r[A][k]; // MO coefficient: AO A contributes to MO k
                        // Displacement vector from nucleus A to the current grid point
                        double dxA = rx - atoms_r[A].x;
                        double dyA = ry - atoms_r[A].y;
                        double dzA = rz - atoms_r[A].z;
                        double r2A = dxA*dxA + dyA*dyA + dzA*dzA; // |r − R_A|² (squared distance)
                        // χ_A(r) = Σ_p d_p · N(α_p) · exp(−α_p · |r−R_A|²)   (contracted STO-3G)
                        double chi = 0.0;
                        for (int p=0; p<NP; p++) {
                            double alp = g_a_exp[A][p]; // STO-3G exponent for atom A, primitive p
                            // Primitive Gaussian contribution: contraction coeff × norm × Gaussian
                            chi += STO3G_d[p] * norm_gauss(alp) * exp(-alp*r2A);
                        }
                        psi += cAk * chi; // Accumulate AO contribution: ψ += C_Ak · χ_A
                    }

                    float psi2 = (float)(psi*psi); // Electron probability density: ρ = |ψ|²
                    if (psi2 < 4e-4f) continue;    // Skip negligible-density regions (reduce clutter)

                    // Alpha proportional to density, hard-capped at 0.38 to stay translucent
                    float alpha = (psi2*14.0f < 0.38f) ? psi2*14.0f : 0.38f;

                    // Select colour pair for this MO (cycle table if occ > 6)
                    int ci = k % 6;
                    if (psi > 0.0)
                        glColor4f(mo_col[ci][0],mo_col[ci][1],mo_col[ci][2],alpha); // Positive lobe
                    else
                        glColor4f(mo_col[ci][3],mo_col[ci][4],mo_col[ci][5],alpha); // Negative lobe

                    glVertex3f(rx, ry, rz); // Emit coloured point into the GL_POINTS batch
                }
            }
        }
    }

    glEnd();              // Flush all GL_POINTS geometry to the GPU
    glPointSize(1.0f);    // Restore default point size for subsequent geometry
    glDepthMask(GL_TRUE); // Re-enable depth writes so opaque geometry renders correctly
    glEnable(GL_LIGHTING); // Restore Phong lighting for atom spheres and bond cylinders
}

// ====================== PANEL O: ORBITAL DIAGRAM ======================
// Draws the molecular orbital energy diagram on screen (toggled with key O).
void drawOrbitalDiagram() {
    int n=nb_r; if (n<1) return; // Nothing to draw if no basis functions

    // Panel geometry: bottom-left corner and dimensions
    float x0=0.62f, y0=-2.95f, gw=1.05f, gh=2.75f;

    // Draw dark background panel
    glColor4f(0.02f,0.06f,0.18f,0.93f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Draw panel border
    glColor3f(0.2f,0.5f,0.85f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Panel title and legend
    glColor3f(0.3f,0.85f,1.0f);
    drawText(x0+0.04f,y0+gh-0.11f,"Orbitals [O]",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.4f,0.4f,0.55f);
    drawText(x0+0.04f,y0+gh-0.22f,"occ=green  virt=grey",GLUT_BITMAP_HELVETICA_10);

    // Map orbital energies (eV) to vertical positions within the panel
    double emin_ev=orbital_eps_r[0]*27.2114, emax_ev=orbital_eps_r[n-1]*27.2114;
    double erange=emax_ev-emin_ev; if (erange<0.5) erange=0.5; // Minimum range to avoid collapse
    float ystart=y0+0.32f, yend=y0+gh-0.32f, xcenter=x0+gw*0.5f;
    char buf[64];

    for (int i=0;i<n;i++) {
        double ev=orbital_eps_r[i]*27.2114; // Orbital energy in electron volts
        // Map energy to y position (linear interpolation within panel height)
        float yp=ystart+(float)((ev-emin_ev)/erange)*(yend-ystart);
        int is_occ=(i<occ_r); // True if this MO is occupied in the ground state

        // Draw horizontal level line: green for occupied, grey for virtual
        if (is_occ) glColor3f(0.15f,1.0f,0.45f); else glColor3f(0.45f,0.45f,0.55f);
        glLineWidth(2.0f);
        glBegin(GL_LINES); glVertex2f(xcenter-0.22f,yp); glVertex2f(xcenter+0.22f,yp); glEnd();
        glLineWidth(1.0f);

        // Draw electron arrows (↑↓) for occupied orbitals
        if (is_occ) {
            glColor3f(1.0f,0.95f,0.2f); // Yellow arrows for electrons

            // Up arrow: shaft + triangle tip
            glBegin(GL_LINES); glVertex2f(xcenter-0.10f,yp+0.001f); glVertex2f(xcenter-0.10f,yp+0.09f); glEnd();
            glBegin(GL_TRIANGLES); glVertex2f(xcenter-0.10f,yp+0.12f); glVertex2f(xcenter-0.13f,yp+0.08f); glVertex2f(xcenter-0.07f,yp+0.08f); glEnd();

            // Down arrow: inverted shaft + triangle tip
            glBegin(GL_LINES); glVertex2f(xcenter+0.04f,yp+0.12f); glVertex2f(xcenter+0.04f,yp+0.03f); glEnd();
            glBegin(GL_TRIANGLES); glVertex2f(xcenter+0.04f,yp+0.001f); glVertex2f(xcenter+0.01f,yp+0.05f); glVertex2f(xcenter+0.07f,yp+0.05f); glEnd();
        }

        // Energy label on the right side of the level
        glColor3f(0.75f,0.85f,0.95f);
        sprintf(buf,"%+.2f eV",ev); drawText(xcenter+0.24f,yp-0.025f,buf,GLUT_BITMAP_HELVETICA_10);

        // MO label on the left side of the level
        glColor3f(0.5f,0.55f,0.65f);
        sprintf(buf,"MO%d",i+1); drawText(x0+0.04f,yp-0.025f,buf,GLUT_BITMAP_HELVETICA_10);
    }

    // Draw vertical energy axis
    glColor3f(0.25f,0.3f,0.4f);
    glBegin(GL_LINES); glVertex2f(xcenter,ystart-0.05f); glVertex2f(xcenter,yend+0.05f); glEnd();
}

// ====================== PANEL M: MULLIKEN ======================
// Draws the Mulliken population analysis panel (toggled with key M).
void drawMulliken() {
    float x0=-2.95f, y0=0.92f, gw=1.5f, rowh=0.28f;
    float gh=0.38f+rowh*(NUM_ATOMS+1); // Panel height scales with number of atoms

    // Background panel
    glColor4f(0.02f,0.10f,0.05f,0.92f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Border
    glColor3f(0.2f,0.75f,0.35f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Title and column headers
    glColor3f(0.25f,1.0f,0.5f);
    drawText(x0+0.05f,y0+gh-0.13f,"Mulliken Populations [M]",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.35f,0.55f,0.4f);
    drawText(x0+0.05f,y0+gh-0.26f,"Atom  Gross Pop   Net Charge",GLUT_BITMAP_HELVETICA_10);

    char buf[64];
    float bar_max=0.45f; // Maximum bar width for the gross population bar

    for (int i=0;i<NUM_ATOMS;i++) {
        float yp=y0+gh-0.38f-rowh*(i+1); // Vertical position for this atom's row
        double gross=(double)g_Z[i]-mulliken_q_r[i]; // Gross population = Z - net charge
        double q_net=mulliken_q_r[i];                 // Net (Mulliken) atomic charge

        // Atom label (e.g., "H1", "He2")
        glColor3f(0.9f,0.9f,0.5f);
        sprintf(buf,"%s%d",g_atom_symbol[i],i+1);
        drawText(x0+0.05f,yp,buf,GLUT_BITMAP_HELVETICA_10);

        // Green bar showing gross population relative to nuclear charge
        float blen=(float)(gross/(double)g_Z[i])*bar_max;
        if (blen<0) blen=0; if (blen>bar_max) blen=bar_max; // Clamp to panel width
        glColor4f(0.15f,0.75f,0.3f,0.75f);
        glBegin(GL_QUADS);
        glVertex2f(x0+0.38f,yp-0.01f); glVertex2f(x0+0.38f+blen,yp-0.01f);
        glVertex2f(x0+0.38f+blen,yp+0.14f); glVertex2f(x0+0.38f,yp+0.14f);
        glEnd();

        // Gross population numeric value
        glColor3f(0.7f,0.9f,0.75f);
        sprintf(buf,"%.3f",gross); drawText(x0+0.42f,yp,buf,GLUT_BITMAP_HELVETICA_10);

        // Net charge: red for positive (electron-deficient), blue for negative (electron-rich)
        if (q_net>0.01f) glColor3f(1.0f,0.4f,0.4f);
        else if (q_net<-0.01f) glColor3f(0.4f,0.6f,1.0f);
        else glColor3f(0.7f,0.85f,0.7f); // Near-neutral: grey-green
        sprintf(buf,"%+.3f e",q_net); drawText(x0+0.90f,yp,buf,GLUT_BITMAP_HELVETICA_10);
    }
}

// ====================== PANEL E: ENERGY DECOMPOSITION ======================
// Draws the energy decomposition panel breaking E_total into T, V_ne, V_ee, V_nn (key E).
void drawEnergyDecomp() {
    float x0=-2.95f,y0=-0.60f,gw=1.82f,gh=1.35f;

    // Background panel
    glColor4f(0.08f,0.03f,0.15f,0.92f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Border
    glColor3f(0.6f,0.3f,0.85f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Title
    glColor3f(0.75f,0.45f,1.0f);
    drawText(x0+0.05f,y0+gh-0.13f,"Energy Decomposition [E]",GLUT_BITMAP_HELVETICA_10);

    // Total energy and its four components
    double E_total=E_kin_r+E_vne_r+E_vee_r+E_nuc_r;
    const char *labels[5]={"T   (e. kinetic)","Vne (nuc. attr.)","Vee (e-e repuls.)","Vnn (nuc. repuls.)","E_total"};
    double vals[5]={E_kin_r,E_vne_r,E_vee_r,E_nuc_r,E_total};
    float colors[5][3]={{0.3f,0.8f,1.0f},{1.0f,0.5f,0.2f},{1.0f,0.8f,0.2f},{0.6f,0.9f,0.6f},{0.85f,0.85f,0.95f}};

    // Find the largest component for relative bar scaling
    double abs_max=0.01;
    for (int i=0;i<5;i++) if (fabs(vals[i])>abs_max) abs_max=fabs(vals[i]);

    float bar_zone=0.55f, rowh=0.21f;
    char buf[80];

    for (int i=0;i<5;i++) {
        float yp=y0+gh-0.28f-rowh*(i+1); // Vertical position for this row

        // Draw separator line above the total energy row
        if (i==4){
            glColor3f(0.4f,0.4f,0.5f);
            glBegin(GL_LINES); glVertex2f(x0+0.03f,yp+rowh-0.02f); glVertex2f(x0+gw-0.03f,yp+rowh-0.02f); glEnd();
        }

        // Component label
        glColor3f(colors[i][0],colors[i][1],colors[i][2]);
        drawText(x0+0.05f,yp,labels[i],GLUT_BITMAP_HELVETICA_10);

        // Horizontal bar proportional to the component's value (diverging: +right, -left)
        float midx=x0+1.12f, blen=(float)(vals[i]/abs_max)*bar_zone;
        float bx_lo=(blen>=0)?midx:midx+blen, bx_hi=(blen>=0)?midx+blen:midx;
        if (fabs(blen)>0.003f){
            glColor4f(colors[i][0],colors[i][1],colors[i][2],0.55f);
            glBegin(GL_QUADS);
            glVertex2f(bx_lo,yp+0.01f); glVertex2f(bx_hi,yp+0.01f);
            glVertex2f(bx_hi,yp+0.15f); glVertex2f(bx_lo,yp+0.15f);
            glEnd();
        }

        // Numeric value in Hartree
        glColor3f(colors[i][0]*0.85f+0.15f,colors[i][1]*0.85f+0.15f,colors[i][2]*0.85f+0.15f);
        sprintf(buf,"%+.5f Ha",vals[i]);
        drawText(x0+gw-0.72f,yp,buf,GLUT_BITMAP_HELVETICA_10);
    }
}

// ====================== ENERGY GRAPH ======================
// Draws a small line chart of the HF total energy history at position (x0, y0).
void drawEnergyGraph(float x0, float y0) {
    int n=(hcount_r<HISTORY_LEN)?hcount_r:HISTORY_LEN; // Number of points to plot
    if (n<2) return; // Need at least two points for a line

    int start=(hcount_r>HISTORY_LEN)?(hcount_r%HISTORY_LEN):0; // Index of oldest entry

    // Find min and max energy for scaling the y-axis
    double emin=ehist_r[start], emax=emin;
    for (int i=0;i<n;i++){int idx=(start+i)%HISTORY_LEN; if(ehist_r[idx]<emin)emin=ehist_r[idx]; if(ehist_r[idx]>emax)emax=ehist_r[idx];}
    double er=emax-emin; if(er<1e-12) er=1e-12; // Prevent division by zero if energy is flat

    float gw=1.65f, gh=0.58f; // Graph panel dimensions

    // Background
    glColor4f(0,0.05f,0.15f,0.85f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Border
    glColor3f(0.25f,0.45f,0.7f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Graph label
    glColor3f(0.65f,0.85f,1.0f);
    drawText(x0+0.04f,y0+gh-0.1f,"E(HF) Ha vs step",GLUT_BITMAP_HELVETICA_10);

    // Draw the energy line chart
    glColor3f(0.1f,1.0f,0.55f); glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (int i=0;i<n;i++){
        int idx=(start+i)%HISTORY_LEN; // Circular buffer index
        float xp=x0+0.05f+(float)i/(n-1)*(gw-0.1f);            // X position (left to right = oldest to newest)
        float yp=y0+0.06f+(float)((ehist_r[idx]-emin)/er)*(gh-0.18f); // Y position scaled to [emin, emax]
        glVertex2f(xp,yp);
    }
    glEnd(); glLineWidth(1.0f);
}

// ====================== HUD ======================
// Renders all 2D overlay elements: title, status readouts, key hints, and optional panels.
void drawHUD() {
    // Switch to orthographic 2D projection for HUD rendering
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(-3,3,-3,3);
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity();
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); // Disable 3D lighting and depth test for 2D overlay
    char buf[256];

    // Main title
    glColor3f(0.3f,0.85f,1.0f);
    drawText(-2.9f,2.72f,"Ab initio MD  -  Hartree-Fock / STO-3G",GLUT_BITMAP_HELVETICA_18);

    // Build system description string based on molecule type
    char sys_name[64];
    if (molecule_type == 1) {
        if (N_heh_pairs == 1)
            strcpy(sys_name, "HeH+  (1 pair, charge +1)");
        else
            sprintf(sys_name, "%d x HeH+  (%d atoms, charge +%d)", N_heh_pairs, NUM_ATOMS, N_heh_pairs);
    } else if (molecule_type == 2) {
        sprintf(sys_name, "%d He atom(s)  (neutral)", NUM_ATOMS);
    } else {
        sprintf(sys_name, "%d H atom(s)", NUM_ATOMS);
    }

    // Subtitle: system info
    glColor3f(0.5f,0.5f,0.62f);
    sprintf(buf,"System: %s | %d electrons | STO-3G | Velocity Verlet + Langevin", sys_name, NUM_ELECTRONS);
    drawText(-2.9f,2.47f,buf,GLUT_BITMAP_HELVETICA_12);

    // SCF computing indicator
    if (computing){glColor3f(1.0f,0.5f,0.0f); drawText(-2.9f,2.22f,"[ COMPUTING SCF... ]",GLUT_BITMAP_HELVETICA_12);}

    // Right-side status readouts
    float px=1.12f,py=2.72f,dy=0.27f;
    glColor3f(1.0f,0.85f,0.0f); sprintf(buf,"Step    : %d",step_r); drawText(px,py,buf,GLUT_BITMAP_HELVETICA_12);
    glColor3f(0.78f,0.78f,0.84f); sprintf(buf,"Time    : %.4f fs",step_r*dt*0.02418884f); drawText(px,py-dy,buf,GLUT_BITMAP_HELVETICA_12);
    glColor3f(0.2f,1.0f,0.5f); sprintf(buf,"E(HF)   : %+.6f Ha",E_r); drawText(px,py-2*dy,buf,GLUT_BITMAP_HELVETICA_12);
    sprintf(buf,"          %+.4f eV",E_r*27.2114); drawText(px,py-3*dy,buf,GLUT_BITMAP_HELVETICA_12); // eV conversion
    glColor3f(1.0f,0.38f,0.3f); sprintf(buf,"T_inst  : %.1f K",T_r); drawText(px,py-4*dy,buf,GLUT_BITMAP_HELVETICA_12);
    glColor3f(0.65f,0.3f,0.22f); sprintf(buf,"T_target: %.0f K  (+/- adjust)",T_target); drawText(px,py-5*dy,buf,GLUT_BITMAP_HELVETICA_12);
    glColor3f(0.4f,0.7f,1.0f); sprintf(buf,"Pressure: %+.5f a.u.",P_r); drawText(px,py-6*dy,buf,GLUT_BITMAP_HELVETICA_12);
    glColor3f(1.0f,0.78f,0.18f); sprintf(buf,"d_min   : %.4f bohr",bond_r); drawText(px,py-7*dy,buf,GLUT_BITMAP_HELVETICA_12);

    // Reference bond length label for the current molecule type
    const char *ref;
    if      (molecule_type==1) ref="HeH+ eq: ~1.46 bohr";
    else if (molecule_type==2) ref="He-He (van der Waals)";
    else                       ref="H2 eq: 0.74 Ang";
    sprintf(buf,"          %.4f Ang  (%s)", bond_r*0.529177f, ref); // Convert bohr to Angstrom
    drawText(px,py-8*dy,buf,GLUT_BITMAP_HELVETICA_12);

    // Per-pair distances (shown only for small systems ≤ 4 atoms)
    if (NUM_ATOMS<=4){
        glColor3f(0.6f,0.7f,0.5f);
        float yy=py-9*dy;
        for (int i=0;i<NUM_ATOMS;i++) for (int j=i+1;j<NUM_ATOMS;j++){
            float dd=atomDistance(atoms_r[i],atoms_r[j]); // Current interatomic distance
            sprintf(buf,"d(%s%d-%s%d): %.3f bohr",g_atom_symbol[i],i+1,g_atom_symbol[j],j+1,dd);
            drawText(px,yy,buf,GLUT_BITMAP_HELVETICA_10); yy-=0.18f;
        }
    }

    // Bottom-left: quantum method legend
    glColor3f(0.45f,0.45f,0.75f); drawText(-2.9f,-1.72f,"[ Quantum Method ]",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.5f,0.5f,0.62f);
    drawText(-2.9f,-1.89f,"Theory    : Hartree-Fock (RHF, closed-shell)",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.04f,"Basis     : STO-3G (3 Gaussians per AO)",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.19f,"Gradient  : Central finite difference (d=0.001 a.u.)",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.34f,"Integrator: Velocity Verlet",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.49f,"Thermostat: Langevin  |  Barostat: Berendsen",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.38f,0.38f,0.48f);
    // Updated controls hint now includes V for orbital volume toggle
    drawText(-2.9f,-2.68f,"W/S:zoom  A/D:pan  Z/X:up/dn  +/-:temp  V:vol.orb",GLUT_BITMAP_HELVETICA_10);

    // Panel toggle key indicators at the bottom — now 5 buttons (O M E F V)
    float kx=-2.9f, ky=-2.82f, ksp=0.50f; // Slightly tighter spacing to fit the extra V button
    const char *key_labels[5]={"[O] Orbitals","[M] Mulliken","[E] Energy","[F] Forces","[V] Vol.Orb"};
    int  *key_flags[5]={&show_orbital,&show_mulliken,&show_energy_dec,&show_forces,&show_volume};
    float key_colors[5][3]={
        {0.3f,0.85f,1.0f},  // Cyan for orbital diagram (O)
        {0.25f,1.0f,0.5f},  // Green for Mulliken panel (M)
        {0.75f,0.4f,1.0f},  // Purple for energy panel (E)
        {1.0f,0.25f,0.25f}, // Red for force arrows (F)
        {0.8f,0.7f,1.0f}    // Lavender for orbital volume cloud (V)
    };
    for (int i=0;i<5;i++){
        if (*key_flags[i]) glColor3f(key_colors[i][0],key_colors[i][1],key_colors[i][2]); // Bright when active
        else glColor3f(0.3f,0.3f,0.38f); // Dim grey when inactive
        drawText(kx+ksp*i,ky,key_labels[i],GLUT_BITMAP_HELVETICA_10);
    }

    drawEnergyGraph(-2.9f,-1.35f); // Always-visible energy history plot

    // Conditionally draw educational panels based on toggle flags
    if (show_orbital)    drawOrbitalDiagram(); // Panel O: MO energy level diagram
    if (show_mulliken)   drawMulliken();        // Panel M: Mulliken charge analysis
    if (show_energy_dec) drawEnergyDecomp();    // Panel E: four-component energy breakdown

    // Restore 3D rendering state
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glMatrixMode(GL_MODELVIEW);  glPopMatrix();
}

// ====================== DISPLAY ======================
// Main OpenGL display callback: clears the framebuffer, renders 3D scene, then HUD.
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
    glLoadIdentity();
    gluLookAt(obs_x,obs_y,obs_z,0,0,0,0,1,0); // Position camera at (obs_x, obs_y, obs_z) looking at origin

    // Take a thread-safe snapshot of current atom states
    struct Atom snap[MAX_ATOM]; float snap_box;
    pthread_mutex_lock(&render_mutex);
    memcpy(snap,atoms_r,NUM_ATOMS*sizeof(struct Atom)); snap_box=box_size;
    pthread_mutex_unlock(&render_mutex);

    // ---- Draw simulation box wireframe ----
    glDisable(GL_LIGHTING);
    {
        float h=snap_box/2.0f; // Half-edge length of the box
        glColor4f(0.35f,0.55f,0.8f,0.35f); glLineWidth(1.2f);
        // Bottom face
        glBegin(GL_LINE_LOOP); glVertex3f(-h,-h,-h); glVertex3f(h,-h,-h); glVertex3f(h,h,-h); glVertex3f(-h,h,-h); glEnd();
        // Top face
        glBegin(GL_LINE_LOOP); glVertex3f(-h,-h,h);  glVertex3f(h,-h,h);  glVertex3f(h,h,h);  glVertex3f(-h,h,h);  glEnd();
        // Vertical edges connecting top and bottom faces
        glBegin(GL_LINES);
        glVertex3f(-h,-h,-h); glVertex3f(-h,-h,h); glVertex3f(h,-h,-h); glVertex3f(h,-h,h);
        glVertex3f(h,h,-h);   glVertex3f(h,h,h);   glVertex3f(-h,h,-h); glVertex3f(-h,h,h);
        glEnd();
    }

    // ---- Draw bonds between nearby atoms ----
    for (int i=0;i<NUM_ATOMS;i++) for (int j=i+1;j<NUM_ATOMS;j++){
        float dd=atomDistance(snap[i],snap[j]);
        if (dd<BOND_CUTOFF) drawBond(snap[i].x,snap[i].y,snap[i].z,snap[j].x,snap[j].y,snap[j].z);
    }

    // ---- Draw force arrows if enabled ----
    if (show_forces) for (int i=0;i<NUM_ATOMS;i++){
        float fx=snap[i].ax*snap[i].mass, fy=snap[i].ay*snap[i].mass, fz=snap[i].az*snap[i].mass; // F = ma
        drawForceArrow(snap[i].x,snap[i].y,snap[i].z,fx,fy,fz);
    }

    // ---- Draw atoms as shaded spheres (opaque — written to depth buffer) ----
    glEnable(GL_LIGHTING);
    for (int i=0;i<NUM_ATOMS;i++){
        if (g_Z[i]==2) glColor3f(0.95f,0.90f,0.35f); // He: warm yellow
        else           glColor3f(0.82f,0.82f,0.95f);  // H: pale blue-white
        glPushMatrix();
        glTranslatef(snap[i].x,snap[i].y,snap[i].z);  // Move to atom position
        glutSolidSphere(snap[i].radius,slices,stacks); // Draw sphere (writes depth)
        glPopMatrix();
    }

    // ---- Panel V: 3D orbital volume cloud (rendered AFTER atoms so depth test works) ----
    // Atoms are already in the depth buffer; the cloud tests against them but never overwrites,
    // so orbital lobes appear around the nuclei without obscuring them.
    if (show_volume) drawOrbitalVolume();

    drawHUD();         // Draw 2D overlay on top of the 3D scene
    glutSwapBuffers(); // Present the rendered frame (double buffering)
}

// Idle callback: continuously request redraws to keep the display updated
static void idle(void){glutPostRedisplay();}

// ====================== INITIALIZE POSITIONS ======================
// Places atoms at their initial positions, assigns velocities from the
// Maxwell-Boltzmann distribution, and computes the first set of HF forces.
void initializePositions() {
    const float kB = 3.1668114e-6f; // Boltzmann constant (atomic units)

    // Expand box proportionally for larger systems to avoid overcrowding
    if (NUM_ATOMS > 4) box_size = 5.0f + (NUM_ATOMS - 4) * 1.5f;

    if (molecule_type == 1) {
        // ---- Place N HeH+ pairs on a regular grid ----
        // Each pair: He at -0.73 bohr, H at +0.73 bohr from the pair center
        float spacing = 5.0f; // Distance between adjacent pair centers (bohr)

        for (int pair = 0; pair < N_heh_pairs; pair++) {
            int he_idx = pair * 2;     // Index of the He atom in this pair
            int h_idx  = pair * 2 + 1; // Index of the H atom in this pair

            // Default: distribute pairs in a single row along X, centered at origin
            float cx = (pair - (N_heh_pairs - 1) / 2.0f) * spacing;
            float cy = 0.0f;
            float cz = 0.0f;

            // For more than 3 pairs, switch to a 2D grid layout
            if (N_heh_pairs > 3) {
                int cols = (int)ceilf(sqrtf((float)N_heh_pairs)); // Number of columns in the grid
                int row  = pair / cols; // Row index of this pair
                int col  = pair % cols; // Column index of this pair
                cx = (col - (cols - 1) / 2.0f) * spacing; // X offset
                cy = (row - (int)(N_heh_pairs / cols) / 2.0f) * spacing; // Y offset
                cz = 0.0f;
            }

            // Place He slightly left and H slightly right of the pair center
            atoms[he_idx].x = cx - 0.73f; atoms[he_idx].y = cy; atoms[he_idx].z = cz;
            atoms[h_idx].x  = cx + 0.73f; atoms[h_idx].y  = cy; atoms[h_idx].z  = cz;

            atoms[he_idx].mass   = 4.0f * proton_mass; // He-4 mass
            atoms[he_idx].radius = 0.35f;               // Slightly larger visual radius for He
            atoms[h_idx].mass    = proton_mass;          // Proton mass for H
            atoms[h_idx].radius  = atom_radius;          // Default hydrogen radius
        }

    } else if (molecule_type == 2) {
        // ---- Place N He atoms at random non-overlapping positions ----
        int placed = 0;
        for (int i = 0; i < NUM_ATOMS; i++) {
            struct Atom *e = &atoms[i];
            e->mass   = 4.0f * proton_mass;
            e->radius = 0.35f;
            do {
                // Random position within the simulation box
                e->x = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                e->y = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                e->z = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                // Check for overlap with already-placed atoms
                int col = 0;
                for (int j = 0; j < placed; j++)
                    if (atomDistance(*e, atoms[j]) < 2.5f) { col = 1; break; }
                if (!col) break; // Accept position if no overlap
            } while (1);
            placed++;
        }

    } else {
        // ---- Place N H atoms at random non-overlapping positions ----
        int placed = 0;
        for (int i = 0; i < NUM_ATOMS; i++) {
            struct Atom *e = &atoms[i];
            e->mass   = proton_mass;
            e->radius = atom_radius;
            do {
                e->x = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                e->y = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                e->z = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                int col = 0;
                for (int j = 0; j < placed; j++)
                    if (atomDistance(*e, atoms[j]) < 1.2f) { col = 1; break; }
                if (!col) break;
            } while (1);
            placed++;
        }
    }

    // ---- Assign Maxwell-Boltzmann velocities ----
    // For each atom: σ = sqrt(kB*T/m), v_i ~ N(0, σ)
    for (int i = 0; i < NUM_ATOMS; i++) {
        struct Atom *e = &atoms[i];
        float sigma = sqrtf(kB * T_target / e->mass); // Thermal speed scale
        e->vx_old = sigma * (float)gauss_rand(); // Gaussian-distributed velocity component
        e->vy_old = sigma * (float)gauss_rand();
        e->vz_old = sigma * (float)gauss_rand();
    }

    // ---- Zero the net center-of-mass momentum ----
    float v_cm[3] = {0,0,0};
    for (int i = 0; i < NUM_ATOMS; i++) {
        v_cm[0] += atoms[i].vx_old;
        v_cm[1] += atoms[i].vy_old;
        v_cm[2] += atoms[i].vz_old;
    }
    for (int i = 0; i < NUM_ATOMS; i++) {
        atoms[i].vx_old -= v_cm[0]/NUM_ATOMS; // Subtract mean velocity in x
        atoms[i].vy_old -= v_cm[1]/NUM_ATOMS; // Subtract mean velocity in y
        atoms[i].vz_old -= v_cm[2]/NUM_ATOMS; // Subtract mean velocity in z
    }

    // ---- Compute initial HF forces before the first MD step ----
    compute_hf_accelerations(atoms, NUM_ATOMS, &E_hf_cached);

    // ---- Compute initial minimum bond length ----
    min_bond = 1e9f;
    for (int i = 0; i < NUM_ATOMS; i++)
        for (int j = i+1; j < NUM_ATOMS; j++) {
            float d = atomDistance(atoms[i], atoms[j]);
            if (d < min_bond) min_bond = d;
        }

    // ---- Initialize render buffers with starting values ----
    energy_history[0] = E_hf_cached; history_count = 1;
    memcpy(atoms_r, atoms, sizeof(atoms)); E_r = E_hf_cached; bond_r = min_bond;
    memcpy(orbital_eps_r, g_orbital_eps, sizeof(g_orbital_eps));
    memcpy(mo_coeff_r,    g_mo_coeff,    sizeof(g_mo_coeff));
    memcpy(mulliken_q_r,  g_mulliken_q,  sizeof(g_mulliken_q));
    E_kin_r = g_E_kin; E_vne_r = g_E_vne; E_vee_r = g_E_vee; E_nuc_r = g_E_nuc;
    nb_r = g_nb; occ_r = g_occ;
}

// ====================== OPENGL INIT ======================
// Sets global OpenGL state: background color, depth test, alpha blending.
void init_gl(void){
    glClearColor(0.04f,0.04f,0.09f,0.0f); // Very dark blue background
    glEnable(GL_DEPTH_TEST); // Enable depth buffering for correct 3D occlusion
    glEnable(GL_BLEND);      // Enable alpha blending for transparent panels
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); // Standard over-compositing blend
}

// Window reshape callback: updates the viewport and projection matrix on resize.
void reshape(int w,int h){
    glViewport(0,0,(GLsizei)w,(GLsizei)h);          // Map rendering to the full window
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluPerspective(60,(GLfloat)w/(GLfloat)h,0.1,50.0); // 60° FOV perspective projection
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

// Keyboard callback: handles all key presses for camera, temperature, and panel toggles.
void key(unsigned char k,int x,int y){
    switch(k){
        case 'w': obs_z-=0.2; break; // Zoom in (move camera closer)
        case 's': obs_z+=0.2; break; // Zoom out (move camera farther)
        case 'a': obs_x-=0.2; break; // Pan camera left
        case 'd': obs_x+=0.2; break; // Pan camera right
        case 'x': obs_y-=0.2; break; // Move camera down
        case 'z': obs_y+=0.2; break; // Move camera up
        case '+': T_target+=50.0f; printf("T_target: %.0f K\n",T_target); break; // Increase target temperature
        case '-': T_target=(T_target>50)?T_target-50:50; printf("T_target: %.0f K\n",T_target); break; // Decrease target temperature (min 50 K)
        case 'o': case 'O': show_orbital=!show_orbital; printf("[O] Orbitals: %s\n",show_orbital?"ON":"OFF"); break;
        case 'm': case 'M': show_mulliken=!show_mulliken; printf("[M] Mulliken: %s\n",show_mulliken?"ON":"OFF"); break;
        case 'e': case 'E': show_energy_dec=!show_energy_dec; printf("[E] Energy: %s\n",show_energy_dec?"ON":"OFF"); break;
        case 'f': case 'F': show_forces=!show_forces; printf("[F] Forces: %s\n",show_forces?"ON":"OFF"); break;
        case 'v': case 'V': show_volume=!show_volume; printf("[V] Orbital Volume: %s\n",show_volume?"ON":"OFF"); break; // Toggle 3D orbital cloud rendering
        case 27: exit(0); // ESC key: exit the application
    }
    glutPostRedisplay(); // Request a redraw after any key press
}

// OpenGL lighting and material properties
const GLfloat light_ambient[]  = {0.1f,0.1f,0.1f,1}; // Low ambient light (dark shadow)
const GLfloat light_diffuse[]  = {1,1,1,1};           // Full white diffuse light
const GLfloat light_specular[] = {1,1,1,1};           // Full white specular highlights
const GLfloat light_position[] = {2,5,5,0};           // Directional light from upper-right
const GLfloat mat_ambient[]    = {0.7f,0.7f,0.7f,1}; // Material ambient reflectance
const GLfloat mat_diffuse[]    = {0.8f,0.8f,0.8f,1}; // Material diffuse reflectance
const GLfloat mat_specular[]   = {1,1,1,1};           // Material specular reflectance (shiny)
const GLfloat high_shininess[] = {100};               // High shininess exponent for glossy look

// ====================== MAIN ======================
int main(int argc, char **argv) {

    // ---- Parse command-line arguments ----
    if (argc > 1) {

        // ---- HeH+ [N_pairs] ----
        if (strcmp(argv[1],"heh+")==0 || strcmp(argv[1],"HeH+")==0 ||
            strcmp(argv[1],"heh")==0  || strcmp(argv[1],"HEH+")==0) {

            molecule_type = 1;  // Set molecule type to HeH+
            N_heh_pairs   = 1;  // Default: one HeH+ pair

            if (argc > 2) {
                int np = atoi(argv[2]); // Parse number of pairs from second argument
                if (np < 1) { fprintf(stderr,"Error: number of HeH+ pairs must be >= 1.\n"); exit(EXIT_FAILURE); }
                if (np*2 > MAX_ATOM) {
                    fprintf(stderr,"Error: %d pairs → %d atoms exceeds MAX_ATOM=%d.\n", np, np*2, MAX_ATOM);
                    exit(EXIT_FAILURE);
                }
                N_heh_pairs = np;
            }

            NUM_ATOMS     = 2 * N_heh_pairs; // Each pair contributes 2 atoms
            NUM_ELECTRONS = 2 * N_heh_pairs; // Each pair contributes 2 electrons (He:2, H:1, charge:-1 → net 2)

            printf("Molecule  : %d x HeH+  (helium hydride cation)\n", N_heh_pairs);
            printf("Atoms     : %d  (%d He + %d H)\n", NUM_ATOMS, N_heh_pairs, N_heh_pairs);
            printf("Electrons : %d  (closed-shell RHF, occ=%d)\n", NUM_ELECTRONS, NUM_ELECTRONS/2);
            printf("Charge    : +%d\n", N_heh_pairs);

        // ---- He [N_atoms] ----
        } else if (strcmp(argv[1],"he")==0 || strcmp(argv[1],"He")==0 || strcmp(argv[1],"HE")==0) {

            molecule_type = 2; // Set molecule type to helium
            int n = 1;
            if (argc > 2) {
                n = atoi(argv[2]); // Parse number of He atoms
                if (n < 1) { fprintf(stderr,"Error: number of He atoms must be >= 1.\n"); exit(EXIT_FAILURE); }
                if (n > MAX_ATOM) {
                    fprintf(stderr,"Error: %d atoms exceeds MAX_ATOM=%d.\n", n, MAX_ATOM);
                    exit(EXIT_FAILURE);
                }
            }
            NUM_ATOMS     = n;
            NUM_ELECTRONS = 2 * n; // Each He atom has 2 electrons

            printf("Molecule  : %d He atom(s)  (neutral)\n", NUM_ATOMS);
            printf("Electrons : %d  (closed-shell RHF, occ=%d)\n", NUM_ELECTRONS, NUM_ELECTRONS/2);

        // ---- H [N_atoms] or plain number (backward compatible) ----
        } else {
            int n;
            if (strcmp(argv[1],"h")==0 || strcmp(argv[1],"H")==0) {
                n = 2;                       // Default H2 if no count given
                if (argc > 2) n = atoi(argv[2]);
            } else {
                n = atoi(argv[1]);           // Direct count: ./aimd_hf 4 (backward compatible)
            }

            if (n <= 0) {
                // Unrecognized argument: print usage and exit
                fprintf(stderr,"Error: unrecognized argument '%s'.\n", argv[1]);
                fprintf(stderr,"Usage:\n");
                fprintf(stderr,"  ./aimd_hf [N]          → N hydrogen atoms\n");
                fprintf(stderr,"  ./aimd_hf h  [N]       → N hydrogen atoms\n");
                fprintf(stderr,"  ./aimd_hf heh+ [N]     → N HeH+ pairs\n");
                fprintf(stderr,"  ./aimd_hf he  [N]      → N He atoms\n");
                exit(EXIT_FAILURE);
            }
            if (n % 2 != 0) {
                // RHF requires an even electron count (closed-shell constraint)
                fprintf(stderr,"Error: RHF requires an even number of electrons. Got %d H atoms.\n", n);
                exit(EXIT_FAILURE);
            }
            if (n > MAX_ATOM) {
                fprintf(stderr,"Error: %d atoms exceeds MAX_ATOM=%d.\n", n, MAX_ATOM);
                exit(EXIT_FAILURE);
            }
            NUM_ATOMS     = n;
            NUM_ELECTRONS = n; // Each H atom contributes 1 electron

            printf("Molecule  : %d hydrogen atom(s)\n", NUM_ATOMS);
            printf("Electrons : %d  (closed-shell RHF, occ=%d)\n", NUM_ELECTRONS, NUM_ELECTRONS/2);
        }

    } else {
        // Default: H2 molecule (2 hydrogen atoms, 2 electrons)
        NUM_ATOMS     = 2;
        NUM_ELECTRONS = 2;
        printf("Molecule  : H2  (default)\n");
    }

    // ---- Assign per-atom STO-3G basis parameters ----
    if (molecule_type == 1) {
        // HeH+ pairs: alternate He (even indices) and H (odd indices)
        for (int pair = 0; pair < N_heh_pairs; pair++) {
            int hi = pair * 2;     // He index
            int li = pair * 2 + 1; // H index
            g_Z[hi] = 2;           // Helium nuclear charge
            for (int p = 0; p < NP; p++) g_a_exp[hi][p] = STO3G_He_a[p]; // He STO-3G exponents
            strcpy(g_atom_symbol[hi], "He");
            g_Z[li] = 1;           // Hydrogen nuclear charge
            for (int p = 0; p < NP; p++) g_a_exp[li][p] = STO3G_H_a[p];  // H STO-3G exponents
            strcpy(g_atom_symbol[li], "H");
        }
    } else if (molecule_type == 2) {
        // All helium atoms
        for (int i = 0; i < NUM_ATOMS; i++) {
            g_Z[i] = 2;
            for (int p = 0; p < NP; p++) g_a_exp[i][p] = STO3G_He_a[p];
            strcpy(g_atom_symbol[i], "He");
        }
    } else {
        // All hydrogen atoms
        for (int i = 0; i < NUM_ATOMS; i++) {
            g_Z[i] = 1;
            for (int p = 0; p < NP; p++) g_a_exp[i][p] = STO3G_H_a[p];
            strcpy(g_atom_symbol[i], "H");
        }
    }

    srand(time(NULL)); // Seed the random number generator with the current time

    // Print simulation header
    printf("================================================\n");
    printf("  AIMD — Hartree-Fock / STO-3G  (v1.1)\n");
    if      (molecule_type == 1)
        printf("  %d x HeH+  |  %d atoms  |  %d electrons  |  charge +%d\n",
               N_heh_pairs, NUM_ATOMS, NUM_ELECTRONS, N_heh_pairs);
    else if (molecule_type == 2)
        printf("  %d He atom(s)  |  %d electrons  |  neutral\n", NUM_ATOMS, NUM_ELECTRONS);
    else
        printf("  %d H atom(s)  |  %d electrons\n", NUM_ATOMS, NUM_ELECTRONS);
    printf("  Panels: O=Orbitals  M=Mulliken  E=Energy  F=Forces  V=Vol.Orb\n");
    printf("  dt = %.4f a.u. = %.6f fs\n", dt, dt*0.02418884f); // Print time step in both unit systems
    printf("================================================\n");

    // Compute initial positions and HF forces
    printf("Computing initial HF forces...\n");
    initializePositions();
    printf("Done. E_initial = %+.8f Ha  (%+.4f eV)\n\n", E_hf_cached, E_hf_cached*27.2114);

    // Print reference energy for validation against literature values
    if (molecule_type == 1 && N_heh_pairs == 1)
        printf("HeH+ STO-3G reference energy ≈ -2.8418 Ha\n\n");
    else if (molecule_type == 2 && NUM_ATOMS == 1)
        printf("He STO-3G reference energy ≈ -2.8077 Ha\n\n");

    // Start the physics thread (runs MD loop in parallel with rendering)
    pthread_t physics_thread;
    pthread_create(&physics_thread, NULL, physics_loop, NULL);

    // ---- Initialize GLUT and OpenGL window ----
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH); // Double-buffered RGB with depth buffer
    glutInitWindowSize(1150, 820);   // Initial window size in pixels
    glutInitWindowPosition(50, 50);  // Initial window position on screen

    // Build a descriptive window title based on the molecule type
    char title[256];
    if (molecule_type == 1)
        sprintf(title,"Ab initio MD  -  HF/STO-3G  |  %dx HeH+  |  O M E F V = Panels", N_heh_pairs);
    else if (molecule_type == 2)
        sprintf(title,"Ab initio MD  -  HF/STO-3G  |  %d He  |  O M E F V = Panels", NUM_ATOMS);
    else
        sprintf(title,"Ab initio MD  -  HF/STO-3G  |  %dH  |  O M E F V = Panels", NUM_ATOMS);
    glutCreateWindow(title);

    // Register GLUT callbacks
    init_gl();
    glutDisplayFunc(display);  // Called to render each frame
    glutReshapeFunc(reshape);  // Called when window is resized
    glutKeyboardFunc(key);     // Called on key press
    glutIdleFunc(idle);        // Called when no events are pending (triggers continuous redraw)

    // ---- Configure lighting ----
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);      // Standard depth testing
    glEnable(GL_LIGHT0);     glEnable(GL_NORMALIZE);    // Enable light source 0 and auto-normalize normals
    glEnable(GL_COLOR_MATERIAL); glEnable(GL_LIGHTING);  // Allow glColor to drive material color

    // Set light source properties
    glLightfv(GL_LIGHT0, GL_AMBIENT,   light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR,  light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION,  light_position);

    // Set front-face material properties
    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    glutMainLoop(); // Enter the GLUT event loop (blocks until window is closed)
    return 0;
}
