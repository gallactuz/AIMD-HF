// ============================================================================
// AIMD-HF — Ab Initio Molecular Dynamics with Real-Time Visualization
// ============================================================================
//
// Method:
//   Restricted Hartree–Fock (RHF)
//   Minimal Basis Set: STO-3G
//
// Description:
//   Real-time ab initio molecular dynamics for small systems (H_n, He_n, HeH+)
//   using full SCF Hartree–Fock at every MD step.
//
//   The program couples:
//     • Electronic structure (SCF)
//     • Nuclear dynamics (classical MD)
//     • Interactive OpenGL visualization
//
//   All analysis quantities are extracted at zero additional SCF cost.
//
// ---------------------------------------------------------------------------
// Educational / Analysis Panels
// ---------------------------------------------------------------------------
//
//   O → Molecular Orbital Diagram
//        - Orbital energies (eV)
//        - Occupation (HOMO / LUMO labeling)
//
//   M → Mulliken Population Analysis
//        - Gross atomic populations
//        - Net atomic charges
//
//   E → Energy Decomposition
//        - T        (electronic kinetic)
//        - V_ne     (electron–nuclear)
//        - V_ee     (electron–electron)
//        - V_nn     (nuclear repulsion)
//
//   F → Nuclear Forces (3D vectors)
//
//   V → Total Electron Density (3D volume cloud)
//   H → HOMO density only
//   L → LUMO density only (if available in basis)
//
//   D → Electric Dipole Moment
//        - Vector (3D)
//        - Magnitude (Debye)
//
//   I → Statistics Panel
//        - Step and time (fs)
//        - Total HF energy (Ha, eV)
//        - Temperature (instantaneous / target)
//        - Pressure (a.u.)
//        - Minimum distance (bohr, Å)
//        - Reference bond (system-dependent)
//        - HOMO–LUMO gap (Ha, eV)
//        - Virial ratio 2<T>/|V| (ideal = 1.000)
//
//   P → Pause / Resume dynamics
//   ENTER → Single MD step (when paused)
//   Q → Deterministic reset (same random seed)
//
// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------
//
//   Linux:
//     g++ aimd_hf.cpp -o aimd_hf -lGL -lGLU -lglut -lm -lpthread
//
// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
//
//   ./aimd_hf h 2        → H2
//   ./aimd_hf h 4        → H4
//   ./aimd_hf he 2       → He2
//   ./aimd_hf he 4       → He4
//   ./aimd_hf heh+       → HeH+
//   ./aimd_hf heh+ 2     → (HeH+)2
//
// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
//
//   Camera:
//     W/S → zoom in/out
//     A/D → rotate left/right
//     Z/X → move up/down
//
//   Simulation:
//     +/- → temperature ±50 K
//     P   → pause/resume
//     ENTER → single step
//     Q   → reset (reproducible)
//
//   Visualization:
//     O → orbital diagram
//     M → Mulliken analysis
//     E → energy decomposition
//     F → force vectors
//     V → total density
//     H → HOMO
//     L → LUMO
//     D → dipole moment
//     I → statistics panel
//
//   System:
//     ESC → exit
//
// ---------------------------------------------------------------------------
//
// Author:
//   Anderson Aparecido do Espirito Santo
//
// Year:
//   2026
//
// ============================================================================


#include <GL/glut.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// ====================== SIMULATION PARAMETERS ======================
int NUM_ATOMS     = 2;
int NUM_ELECTRONS = 2;
#define dt            2.0f
#define BOND_CUTOFF   2.0f

#define MAX_ATOM      12
#define NP            3
#define MAX_BASIS     (MAX_ATOM * NP)
#define HISTORY_LEN   300

static int molecule_type = 0;
static int N_heh_pairs   = 1;

static int    g_Z[MAX_ATOM];
static double g_a_exp[MAX_ATOM][NP];
static char   g_atom_symbol[MAX_ATOM][4];

static const double STO3G_d[NP]    = {0.15432897, 0.53532814, 0.44463454};
static const double STO3G_H_a[NP]  = {3.42525091, 0.62391373, 0.16885540};
static const double STO3G_He_a[NP] = {6.36242139, 1.15892300, 0.31364979};

// ====================== CAMERA GLOBALS ======================
GLdouble obs_x = 0, obs_y = 0, obs_z = 7.0;
static int   slices        = 20;
static int   stacks        = 20;
static float atom_radius   = 0.28f;
static float proton_mass   = 1836.0f;
static float box_size      = 5.0f;
static float T_target      = 300.0f;
static float target_pressure = 0.001f;

// ====================== ATOM STRUCTURE ======================
struct Atom {
    float x, y, z;
    float vx_old, vy_old, vz_old;
    float vx, vy, vz;
    float ax, ay, az;
    float mass, radius;
};

// ====================== PHYSICS-SIDE BUFFERS ======================
static struct Atom atoms[MAX_ATOM];
static double E_hf_cached = 0.0;
static float  T_inst      = 0.0f;
static float  P_inst      = 0.0f;
static float  min_bond    = 0.0f;
static double energy_history[HISTORY_LEN];
static int    history_count = 0;
static int    md_step       = 0;

static double g_orbital_eps[MAX_BASIS];
static double g_mo_coeff[MAX_BASIS][MAX_BASIS];
static double g_mulliken_q[MAX_ATOM];
static double g_E_kin = 0.0;
static double g_E_vne = 0.0;
static double g_E_vee = 0.0;
static double g_E_nuc = 0.0;
static int    g_nb  = 2;
static int    g_occ = 1;

// --- NEW: dipole moment (atomic units) computed each SCF ---
static double g_dipole[3]       = {0.0, 0.0, 0.0};
// --- NEW: HOMO-LUMO gap and Virial ratio ---
static double g_homo_lumo_gap   = 0.0;   // in Hartree
static double g_virial_ratio    = 0.0;   // 2<T>/|<V>|

// ====================== RENDER-SIDE BUFFERS ======================
static struct Atom atoms_r[MAX_ATOM];
static double E_r     = 0.0;
static float  T_r     = 0.0f;
static float  P_r     = 0.0f;
static float  bond_r  = 0.0f;
static int    step_r  = 0;
static double ehist_r[HISTORY_LEN];
static int    hcount_r = 0;
static int    computing = 0;

static double orbital_eps_r[MAX_BASIS];
static double mo_coeff_r[MAX_BASIS][MAX_BASIS];
static double mulliken_q_r[MAX_ATOM];
static double E_kin_r = 0.0, E_vne_r = 0.0;
static double E_vee_r = 0.0, E_nuc_r = 0.0;
static int    nb_r = 2, occ_r = 1;

// --- NEW render-side: dipole, gap, virial ---
static double dipole_r[3]       = {0.0, 0.0, 0.0};
static double homo_lumo_gap_r   = 0.0;
static double virial_ratio_r    = 0.0;

// Toggle flags (original)
static int show_orbital    = 0;
static int show_mulliken   = 0;
static int show_energy_dec = 0;
static int show_forces     = 0;


// --- NEW toggle flags ---
static int show_dipole     = 0;  // D
static int show_stats      = 0;  // I
static int show_homo_lumo  = 0;  // L  (HOMO/LUMO only in volume cloud)
static int white_background = 0; // 0 = preto, 1 = branco



// --- NEW: Pause / single-step control ---
static volatile int  md_paused    = 0;  // 1 = physics thread is paused
static volatile int  md_step_once = 0;  // 1 = take exactly one step then pause again

// --- NEW: reproducible reset ---
static unsigned int  initial_seed = 0;  // saved at startup for Q-reset

static pthread_mutex_t render_mutex = PTHREAD_MUTEX_INITIALIZER;
// Condition variable so the physics thread can sleep cheaply while paused
static pthread_cond_t  pause_cond   = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t pause_mutex  = PTHREAD_MUTEX_INITIALIZER;

// ====================== BOX-MULLER GAUSSIAN RANDOM ======================
double gauss_rand() {
    static int    has_spare = 0;
    static double spare;
    if (has_spare) { has_spare = 0; return spare; }
    has_spare = 1;
    double u, v, s;
    do {
        u = (rand()/(double)RAND_MAX)*2.0 - 1.0;
        v = (rand()/(double)RAND_MAX)*2.0 - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0*log(s)/s);
    spare = v*s;
    return u*s;
}

// ====================== HF INTEGRAL HELPERS ======================
double F0(double t) {
    if (t < 1e-12) return 1.0;
    return 0.5*sqrt(M_PI/t)*erf(sqrt(t));
}

double norm_gauss(double alpha) {
    return pow(2.0*alpha/M_PI, 0.75);
}

double dist2(const double A[3], const double B[3]) {
    double dx=A[0]-B[0], dy=A[1]-B[1], dz=A[2]-B[2];
    return dx*dx+dy*dy+dz*dz;
}

float atomDistance(struct Atom a, struct Atom b) {
    return sqrtf((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}

// ====================== JACOBI EIGENVALUE DIAGONALIZATION ======================
void jacobi(int n, double A[MAX_BASIS][MAX_BASIS],
            double V[MAX_BASIS][MAX_BASIS], double eig[MAX_BASIS]) {
    memset(V, 0, sizeof(double)*MAX_BASIS*MAX_BASIS);
    for (int i = 0; i < n; i++) V[i][i] = 1.0;

    for (int iter = 0; iter < 200; iter++) {
        int p=0, q=1;
        double maxval = fabs(A[p][q]);
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                if (fabs(A[i][j]) > maxval) { maxval=fabs(A[i][j]); p=i; q=j; }

        if (maxval < 1e-10) break;

        double theta = 0.5*atan2(2.0*A[p][q], A[q][q]-A[p][p]);
        double c=cos(theta), s=sin(theta);

        double App = c*c*A[p][p]-2*s*c*A[p][q]+s*s*A[q][q];
        double Aqq = s*s*A[p][p]+2*s*c*A[p][q]+c*c*A[q][q];
        A[p][p]=App; A[q][q]=Aqq; A[p][q]=A[q][p]=0.0;

        for (int k = 0; k < n; k++) {
            if (k==p||k==q) continue;
            double Akp=c*A[k][p]-s*A[k][q], Akq=s*A[k][p]+c*A[k][q];
            A[k][p]=A[p][k]=Akp; A[k][q]=A[q][k]=Akq;
        }
        for (int k = 0; k < n; k++) {
            double vkp=c*V[k][p]-s*V[k][q], vkq=s*V[k][p]+c*V[k][q];
            V[k][p]=vkp; V[k][q]=vkq;
        }
    }
    for (int i = 0; i < n; i++) eig[i]=A[i][i];
}

// ====================== HARTREE-FOCK SCF ENERGY ======================
double compute_hf_energy(const double R_atom_in[MAX_ATOM][3],
                         int NB, int do_analysis) {
    double R_atom[MAX_ATOM][3];
    for (int i = 0; i < NB; i++)
        for (int k = 0; k < 3; k++)
            R_atom[i][k] = R_atom_in[i][k];

    double d_coef[NP];
    for (int p = 0; p < NP; p++) d_coef[p] = STO3G_d[p];

    double a_prim[MAX_BASIS];
    double R_prim[MAX_BASIS][3];
    for (int A = 0; A < NB; A++)
        for (int p = 0; p < NP; p++) {
            int idx = A*NP+p;
            a_prim[idx] = g_a_exp[A][p];
            for (int k = 0; k < 3; k++) R_prim[idx][k] = R_atom[A][k];
        }

    // ---- One-electron integrals ----
    double S[MAX_BASIS][MAX_BASIS]     = {{0}};
    double T_kin[MAX_BASIS][MAX_BASIS] = {{0}};
    double Vne[MAX_BASIS][MAX_BASIS]   = {{0}};
    // NEW: dipole integrals <i|r_k|j> for k=0,1,2
    double Dip[3][MAX_BASIS][MAX_BASIS];
    memset(Dip, 0, sizeof(Dip));
    double Hcore[MAX_BASIS][MAX_BASIS];

    for (int i = 0; i < NB; i++)
        for (int j = i; j < NB; j++) {
            for (int p = 0; p < NP; p++)
                for (int q = 0; q < NP; q++) {
                    int ip=i*NP+p, jq=j*NP+q;
                    double al1=a_prim[ip], al2=a_prim[jq];
                    double N1=norm_gauss(al1), N2=norm_gauss(al2);
                    double gam=al1+al2;
                    double ratio=al1*al2/gam;
                    double Rab2=dist2(R_prim[ip], R_prim[jq]);
                    double Rp[3];
                    for (int k = 0; k < 3; k++)
                        Rp[k]=(al1*R_prim[ip][k]+al2*R_prim[jq][k])/gam;

                    double Sij=pow(M_PI/gam,1.5)*exp(-ratio*Rab2);
                    double Tij=ratio*(3.0-2.0*ratio*Rab2)*Sij;

                    double Vij=0.0;
                    for (int A2 = 0; A2 < NB; A2++) {
                        double rPA2=dist2(Rp, R_atom[A2]);
                        Vij -= (double)g_Z[A2]*(2.0*M_PI/gam)*exp(-ratio*Rab2)*F0(gam*rPA2);
                    }

                    double cc=d_coef[p]*d_coef[q]*N1*N2;
                    S[i][j]     += cc*Sij;
                    T_kin[i][j] += cc*Tij;
                    Vne[i][j]   += cc*Vij;
                    // Dipole integrals: <g_p|r_k|g_q> = Rp[k] * Sij  (s-type Gaussians)
                    for (int k = 0; k < 3; k++)
                        Dip[k][i][j] += cc * Rp[k] * Sij;
                }
            S[j][i]     = S[i][j];
            T_kin[j][i] = T_kin[i][j];
            Vne[j][i]   = Vne[i][j];
            for (int k = 0; k < 3; k++) Dip[k][j][i] = Dip[k][i][j];
        }

    for (int i = 0; i < NB; i++)
        for (int j = 0; j < NB; j++)
            Hcore[i][j] = T_kin[i][j]+Vne[i][j];

    // ---- Nuclear repulsion ----
    double E_nuc = 0.0;
    for (int A = 0; A < NB; A++)
        for (int B = A+1; B < NB; B++)
            E_nuc += (double)(g_Z[A]*g_Z[B]) / sqrt(dist2(R_atom[A], R_atom[B]));

    // ---- Two-electron repulsion integrals ----
    static double eri[MAX_BASIS][MAX_BASIS][MAX_BASIS][MAX_BASIS];
    memset(eri, 0, sizeof(eri));
    for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
    for (int k=0;k<NB;k++) for (int l=0;l<NB;l++) {
        double val=0.0;
        for (int p=0;p<NP;p++) for (int q=0;q<NP;q++)
        for (int r=0;r<NP;r++) for (int s=0;s<NP;s++) {
            int ip=i*NP+p,jq=j*NP+q,kr=k*NP+r,ls=l*NP+s;
            double a1=a_prim[ip],a2=a_prim[jq],a3=a_prim[kr],a4=a_prim[ls];
            double N1=norm_gauss(a1),N2=norm_gauss(a2),N3=norm_gauss(a3),N4=norm_gauss(a4);
            double g1=a1+a3, g2=a2+a4, g=g1+g2;
            double Rp2[3],Rq2[3];
            for (int dd=0;dd<3;dd++) {
                Rp2[dd]=(a1*R_prim[ip][dd]+a3*R_prim[kr][dd])/g1;
                Rq2[dd]=(a2*R_prim[jq][dd]+a4*R_prim[ls][dd])/g2;
            }
            double rab2=dist2(R_prim[ip],R_prim[kr]);
            double rcd2=dist2(R_prim[jq],R_prim[ls]);
            double rpq2=dist2(Rp2,Rq2);
            double pref=2.0*pow(M_PI,2.5)/(g1*g2*sqrt(g));
            double boys_arg=(g1*g2/g)*rpq2;
            val+=d_coef[p]*d_coef[q]*d_coef[r]*d_coef[s]*N1*N2*N3*N4*pref
                *exp(-a1*a3/g1*rab2-a2*a4/g2*rcd2)*F0(boys_arg);
        }
        eri[i][j][k][l]=val;
    }

    // ---- Löwdin orthogonalization X = S^{-1/2} ----
    double Scopy[MAX_BASIS][MAX_BASIS], U[MAX_BASIS][MAX_BASIS], eigS[MAX_BASIS];
    memcpy(Scopy, S, sizeof(S));
    jacobi(NB, Scopy, U, eigS);

    double X[MAX_BASIS][MAX_BASIS]={{0}};
    for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
        for (int k=0;k<NB;k++)
            if (eigS[k]>1e-10)
                X[i][j]+=U[i][k]*(1.0/sqrt(eigS[k]))*U[j][k];

    // ---- SCF loop ----
    double P[MAX_BASIS][MAX_BASIS]={{0}};
    double E_old=0.0, damp=0.6, Etot=0.0;
    int occ     = NUM_ELECTRONS/2;
    int max_iter = 120, iter;
    double C_final[MAX_BASIS][MAX_BASIS]={{0}};
    double eps_final[MAX_BASIS]={0};
    double P_final[MAX_BASIS][MAX_BASIS]={{0}};
    double G_final[MAX_BASIS][MAX_BASIS]={{0}};

    for (iter=0; iter<max_iter; iter++) {
        double G[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int p2=0;p2<NB;p2++) for (int q=0;q<NB;q++)
                G[i][j]+=P[p2][q]*(eri[i][q][j][p2]-0.5*eri[i][q][p2][j]);

        double F_mat[MAX_BASIS][MAX_BASIS];
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            F_mat[i][j]=Hcore[i][j]+G[i][j];

        double Fp[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int k=0;k<NB;k++) for (int l=0;l<NB;l++)
                Fp[i][j]+=X[k][i]*F_mat[k][l]*X[l][j];

        double Acopy[MAX_BASIS][MAX_BASIS], Cp[MAX_BASIS][MAX_BASIS], eps[MAX_BASIS];
        memcpy(Acopy, Fp, sizeof(Fp));
        jacobi(NB, Acopy, Cp, eps);

        for (int i=0;i<NB-1;i++) for (int jj=i+1;jj<NB;jj++)
            if (eps[jj]<eps[i]) {
                double tmp=eps[i]; eps[i]=eps[jj]; eps[jj]=tmp;
                for (int k=0;k<NB;k++) { tmp=Cp[k][i]; Cp[k][i]=Cp[k][jj]; Cp[k][jj]=tmp; }
            }

        double C[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int k=0;k<NB;k++) C[i][j]+=X[i][k]*Cp[k][j];

        double Pnew[MAX_BASIS][MAX_BASIS]={{0}};
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            for (int k=0;k<occ;k++) Pnew[i][j]+=2.0*C[i][k]*C[j][k];

        double Eelec=0.0;
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            Eelec+=0.5*Pnew[i][j]*(Hcore[i][j]+F_mat[i][j]);
        Etot=Eelec+E_nuc;

        double deltaE=fabs(Etot-E_old), deltaP=0.0;
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            deltaP+=fabs(Pnew[i][j]-P[i][j]);

        if (deltaE<1e-8 && deltaP<1e-6) {
            memcpy(C_final,   C,    sizeof(C));
            memcpy(eps_final, eps,  sizeof(eps));
            memcpy(P_final,   Pnew, sizeof(Pnew));
            memcpy(G_final,   G,    sizeof(G));
            break;
        }

        if (iter>8)  damp=0.35;
        if (iter>20) damp=0.20;

        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++)
            P[i][j]=(1.0-damp)*P[i][j]+damp*Pnew[i][j];
        E_old=Etot;

        memcpy(C_final,   C,    sizeof(C));
        memcpy(eps_final, eps,  sizeof(eps));
        memcpy(P_final,   Pnew, sizeof(Pnew));
        memcpy(G_final,   G,    sizeof(G));
    }

    static int last_step_reported=-1;
    if (md_step!=last_step_reported) {
        if (iter>=max_iter-1)
            printf("Warning: SCF did not converge after %d iterations\n", max_iter);
        else if (iter>30)
            printf("SCF converged in %d iterations\n", iter+1);
        last_step_reported=md_step;
    }

    // ====================== EDUCATIONAL ANALYSIS ======================
    if (do_analysis) {
        g_nb=NB; g_occ=occ;

        for (int i=0;i<NB;i++) {
            g_orbital_eps[i]=eps_final[i];
            for (int j=0;j<NB;j++) g_mo_coeff[i][j]=C_final[i][j];
        }

        for (int A=0;A<NB;A++) {
            double gross=0.0;
            for (int nu=0;nu<NB;nu++) gross+=P_final[A][nu]*S[nu][A];
            g_mulliken_q[A]=(double)g_Z[A]-gross;
        }

        double Ekin_comp=0.0, Evne_comp=0.0, Evee_comp=0.0;
        for (int i=0;i<NB;i++) for (int j=0;j<NB;j++) {
            Ekin_comp+=P_final[i][j]*T_kin[i][j];
            Evne_comp+=P_final[i][j]*Vne[i][j];
            Evee_comp+=0.5*P_final[i][j]*G_final[i][j];
        }
        g_E_kin=Ekin_comp; g_E_vne=Evne_comp;
        g_E_vee=Evee_comp; g_E_nuc=E_nuc;

        // ---- NEW: Electric dipole moment ----
        // μ_k = Σ_A Z_A * R_A[k]  -  Σ_{ij} P_ij * <i|r_k|j>
        for (int k = 0; k < 3; k++) {
            double nuc_contrib = 0.0;
            for (int A = 0; A < NB; A++)
                nuc_contrib += (double)g_Z[A] * R_atom[A][k];
            double elec_contrib = 0.0;
            for (int i = 0; i < NB; i++)
                for (int j = 0; j < NB; j++)
                    elec_contrib += P_final[i][j] * Dip[k][i][j];
            g_dipole[k] = nuc_contrib - elec_contrib;
        }

        // ---- NEW: HOMO-LUMO gap ----
        if (occ < NB)
            g_homo_lumo_gap = eps_final[occ] - eps_final[occ-1]; // LUMO - HOMO
        else
            g_homo_lumo_gap = 0.0;

        // ---- NEW: Electronic Virial ratio  2<T_e>/|<V_e>| ----
        // V_e = V_ne + V_ee + V_nn
        double V_total = Evne_comp + Evee_comp + E_nuc;
        if (fabs(V_total) > 1e-12)
            g_virial_ratio = 2.0 * Ekin_comp / fabs(V_total);
        else
            g_virial_ratio = 0.0;
    }

    return Etot;
}

// ====================== HF FORCES VIA FINITE DIFFERENCE ======================
void compute_hf_accelerations(struct Atom *atoms_in, int NB, double *E_out) {
    double R[MAX_ATOM][3];
    for (int i=0;i<NB;i++) {
        R[i][0]=atoms_in[i].x; R[i][1]=atoms_in[i].y; R[i][2]=atoms_in[i].z;
    }
    const double delta=0.001;

    *E_out=compute_hf_energy(R, NB, 1);

    for (int i=0;i<NB;i++) {
        double Rs[3]={R[i][0],R[i][1],R[i][2]}, Ep, Em;

        R[i][0]=Rs[0]+delta; Ep=compute_hf_energy(R,NB,0);
        R[i][0]=Rs[0]-delta; Em=compute_hf_energy(R,NB,0);
        R[i][0]=Rs[0]; atoms_in[i].ax=-(Ep-Em)/(2.0*delta)/atoms_in[i].mass;

        R[i][1]=Rs[1]+delta; Ep=compute_hf_energy(R,NB,0);
        R[i][1]=Rs[1]-delta; Em=compute_hf_energy(R,NB,0);
        R[i][1]=Rs[1]; atoms_in[i].ay=-(Ep-Em)/(2.0*delta)/atoms_in[i].mass;

        R[i][2]=Rs[2]+delta; Ep=compute_hf_energy(R,NB,0);
        R[i][2]=Rs[2]-delta; Em=compute_hf_energy(R,NB,0);
        R[i][2]=Rs[2]; atoms_in[i].az=-(Ep-Em)/(2.0*delta)/atoms_in[i].mass;
    }
}

// ====================== BOUNDARY CONDITIONS ======================
void applyBoundaryConditions(struct Atom *e) {
    float h=box_size/2.0f;
    if (e->x+e->radius> h){e->x= h-e->radius; e->vx_old=-e->vx_old;}
    if (e->x-e->radius<-h){e->x=-h+e->radius; e->vx_old=-e->vx_old;}
    if (e->y+e->radius> h){e->y= h-e->radius; e->vy_old=-e->vy_old;}
    if (e->y-e->radius<-h){e->y=-h+e->radius; e->vy_old=-e->vy_old;}
    if (e->z+e->radius> h){e->z= h-e->radius; e->vz_old=-e->vz_old;}
    if (e->z-e->radius<-h){e->z=-h+e->radius; e->vz_old=-e->vz_old;}
}

// ====================== LANGEVIN THERMOSTAT ======================
float controlTemperature(float T_desired) {
    const float kB=3.1668114e-6f;
    float K=0.0f;
    for (int i=0;i<NUM_ATOMS;i++) {
        struct Atom *e=&atoms[i];
        K+=0.5f*e->mass*(e->vx_old*e->vx_old+e->vy_old*e->vy_old+e->vz_old*e->vz_old);
    }
    float T_inst_local=(2.0f*K)/(3.0f*NUM_ATOMS*kB);
    float c1=expf(-0.1f*dt);
    float c2b=sqrtf((1.0f-c1*c1)*kB*T_desired);
    for (int i=0;i<NUM_ATOMS;i++) {
        struct Atom *e=&atoms[i];
        float c2=c2b/sqrtf(e->mass);
        e->vx_old=c1*e->vx_old+c2*(float)gauss_rand();
        e->vy_old=c1*e->vy_old+c2*(float)gauss_rand();
        e->vz_old=c1*e->vz_old+c2*(float)gauss_rand();
    }
    return T_inst_local;
}

// ====================== VIRIAL PRESSURE ======================
float computePressure() {
    float K=0.0f, virial=0.0f;
    for (int i=0;i<NUM_ATOMS;i++) {
        struct Atom *e=&atoms[i];
        K+=0.5f*e->mass*(e->vx_old*e->vx_old+e->vy_old*e->vy_old+e->vz_old*e->vz_old);
        virial+=e->x*e->mass*e->ax+e->y*e->mass*e->ay+e->z*e->mass*e->az;
    }
    float V=box_size*box_size*box_size;
    return (2.0f*K+virial)/(3.0f*V);
}

// ====================== BERENDSEN BAROSTAT ======================
void rescaleBox(float P_current) {
    float scale=1.0f-0.0005f*(dt/5.0f)*(target_pressure-P_current);
    box_size*=scale;
    for (int i=0;i<NUM_ATOMS;i++) { atoms[i].x*=scale; atoms[i].y*=scale; atoms[i].z*=scale; }
}

// ====================== MD STEP ======================
void stepMD() {
    for (int i=0;i<NUM_ATOMS;i++) {
        atoms[i].vx_old+=0.5f*atoms[i].ax*dt;
        atoms[i].vy_old+=0.5f*atoms[i].ay*dt;
        atoms[i].vz_old+=0.5f*atoms[i].az*dt;
    }
    T_inst=controlTemperature(T_target);
    for (int i=0;i<NUM_ATOMS;i++) {
        atoms[i].x+=atoms[i].vx_old*dt;
        atoms[i].y+=atoms[i].vy_old*dt;
        atoms[i].z+=atoms[i].vz_old*dt;
        applyBoundaryConditions(&atoms[i]);
    }
    compute_hf_accelerations(atoms, NUM_ATOMS, &E_hf_cached);
    for (int i=0;i<NUM_ATOMS;i++) {
        atoms[i].vx_old+=0.5f*atoms[i].ax*dt;
        atoms[i].vy_old+=0.5f*atoms[i].ay*dt;
        atoms[i].vz_old+=0.5f*atoms[i].az*dt;
    }
    P_inst=computePressure();
    rescaleBox(P_inst);

    min_bond=1e9f;
    for (int i=0;i<NUM_ATOMS;i++)
        for (int j=i+1;j<NUM_ATOMS;j++) {
            float d=atomDistance(atoms[i],atoms[j]);
            if (d<min_bond) min_bond=d;
        }

    energy_history[history_count%HISTORY_LEN]=E_hf_cached;
    history_count++; md_step++;

    printf("\n=== STEP %d (t=%.4f fs) ===\n", md_step, md_step*dt*0.02418884f);
    printf("  E(HF)    : %+.8f Ha  (%+.4f eV)\n", E_hf_cached, E_hf_cached*27.2114);
    printf("  E_kin    : %+.6f Ha   E_Vne : %+.6f Ha\n", g_E_kin, g_E_vne);
    printf("  E_Vee    : %+.6f Ha   E_Vnn : %+.6f Ha\n", g_E_vee, g_E_nuc);
    printf("  T_inst   : %.2f K  (target: %.0f K)\n", T_inst, T_target);
    printf("  Pressure : %+.6f a.u.\n", P_inst);
    printf("  d_min    : %.4f bohr = %.4f Ang\n", min_bond, min_bond*0.529177f);
    // NEW: print dipole each step
    double dip_D[3]; for (int k=0;k<3;k++) dip_D[k]=g_dipole[k]*2.5418;
    printf("  Dipole   : (%.4f, %.4f, %.4f) D  |μ|=%.4f D\n",
           dip_D[0], dip_D[1], dip_D[2],
           sqrt(dip_D[0]*dip_D[0]+dip_D[1]*dip_D[1]+dip_D[2]*dip_D[2]));
    printf("  HOMO-LUMO: %.4f Ha = %.4f eV\n", g_homo_lumo_gap, g_homo_lumo_gap*27.2114);
    printf("  Virial   : 2<T>/|<V>| = %.6f  (ideal=1.000)\n", g_virial_ratio);
    for (int i=0;i<NUM_ATOMS;i++)
        printf("  Mulliken q[%s%d] : %+.4f e\n", g_atom_symbol[i], i+1, g_mulliken_q[i]);
}

// ====================== PHYSICS THREAD ======================
void* physics_loop(void* arg) {
    (void)arg;
    while (1) {
        // NEW: honour pause flag — sleep cheaply on the condition variable
        pthread_mutex_lock(&pause_mutex);
        while (md_paused && !md_step_once)
            pthread_cond_wait(&pause_cond, &pause_mutex);
        pthread_mutex_unlock(&pause_mutex);

        pthread_mutex_lock(&render_mutex); computing=1; pthread_mutex_unlock(&render_mutex);

        stepMD();

        pthread_mutex_lock(&render_mutex);
        memcpy(atoms_r, atoms, sizeof(atoms));
        E_r=E_hf_cached; T_r=T_inst; P_r=P_inst; bond_r=min_bond; step_r=md_step;
        memcpy(ehist_r, energy_history, sizeof(energy_history)); hcount_r=history_count;
        memcpy(orbital_eps_r, g_orbital_eps, sizeof(g_orbital_eps));
        memcpy(mo_coeff_r,    g_mo_coeff,    sizeof(g_mo_coeff));
        memcpy(mulliken_q_r,  g_mulliken_q,  sizeof(g_mulliken_q));
        E_kin_r=g_E_kin; E_vne_r=g_E_vne; E_vee_r=g_E_vee; E_nuc_r=g_E_nuc;
        nb_r=g_nb; occ_r=g_occ;
        // NEW: copy dipole, gap, virial
        for (int k=0;k<3;k++) dipole_r[k] = g_dipole[k];
        homo_lumo_gap_r = g_homo_lumo_gap;
        virial_ratio_r  = g_virial_ratio;
        computing=0;
        pthread_mutex_unlock(&render_mutex);

        // NEW: if single-step mode, go back to paused after one step
        if (md_step_once) {
            pthread_mutex_lock(&pause_mutex);
            md_step_once = 0;
            pthread_mutex_unlock(&pause_mutex);
        }
    }
    return NULL;
}

// ====================== 2D TEXT HELPER ======================
void drawText(float x, float y, const char *str, void *font) {
    glRasterPos2f(x, y);
    for (int i=0; str[i]; i++) glutBitmapCharacter(font, str[i]);
}

// ====================== BOND CYLINDER ======================
void drawBond(float x1,float y1,float z1, float x2,float y2,float z2) {
    float dx=x2-x1,dy=y2-y1,dz=z2-z1;
    float len=sqrtf(dx*dx+dy*dy+dz*dz);
    if (len<1e-4f) return;

    float rot_ax=-dy, rot_ay=dx;
    float angle=(180.0f/M_PI)*acosf(dz/len);
    if (fabsf(dz/len)>0.9999f){rot_ax=1;rot_ay=0;angle=(dz<0)?180.0f:0.0f;}

    glPushMatrix();
    glTranslatef(x1,y1,z1);
    glRotatef(angle,rot_ax,rot_ay,0.0f);
    GLUquadric *q=gluNewQuadric();
    glColor4f(0.95f,0.85f,0.1f,0.85f);
    gluCylinder(q,0.07,0.07,len,14,1);
    gluDeleteQuadric(q);
    glPopMatrix();
}

// ====================== FORCE ARROWS ======================
void drawForceArrow(float x,float y,float z,float fx,float fy,float fz) {
    float flen=sqrtf(fx*fx+fy*fy+fz*fz);
    if (flen<1e-14f) return;

    float vscale=300.0f;
    float ex=x+fx*vscale,ey=y+fy*vscale,ez=z+fz*vscale;

    float alen=sqrtf((ex-x)*(ex-x)+(ey-y)*(ey-y)+(ez-z)*(ez-z));
    if (alen<0.02f) return;

    glDisable(GL_LIGHTING);
    glColor3f(1.0f,0.12f,0.12f); glLineWidth(2.8f);
    glBegin(GL_LINES); glVertex3f(x,y,z); glVertex3f(ex,ey,ez); glEnd();
    glLineWidth(1.0f);

    float dx=ex-x,dy2=ey-y,dz2=ez-z;
    float norm=sqrtf(dx*dx+dy2*dy2+dz2*dz2);
    float axr=-dy2,ayr=dx,ang=(180.0f/M_PI)*acosf(dz2/norm);
    if (fabsf(dz2/norm)>0.9999f){axr=1;ayr=0;ang=(dz2<0)?180.0f:0.0f;}

    glPushMatrix(); glTranslatef(ex,ey,ez); glRotatef(ang,axr,ayr,0.0f);
    GLUquadric *q=gluNewQuadric();
    glColor3f(1.0f,0.25f,0.1f);
    gluCylinder(q,0.09f,0.0f,0.22f,12,1);
    gluDeleteQuadric(q); glPopMatrix();
    glEnable(GL_LIGHTING);
}

// ====================== DIPOLE 3D ARROW ======================
// Draws the electric dipole vector in 3D space.
// The arrow is rendered as a cylinder + cone (robust thickness).
// The origin is shifted backward along the dipole direction so the arrow
// does not overlap with the molecule.
// Length is scaled proportionally to dipole magnitude.

void drawDipoleArrow3D() {
    double dip[3];
    int nb;

    // Thread-safe copy of dipole and atom count
    pthread_mutex_lock(&render_mutex);
    for (int k=0;k<3;k++) dip[k]=dipole_r[k];
    nb=nb_r;
    pthread_mutex_unlock(&render_mutex);

    // Dipole magnitude (atomic units)
    double mag = sqrt(dip[0]*dip[0] + dip[1]*dip[1] + dip[2]*dip[2]);
    if (mag < 1e-5) return;

    // Compute molecular centroid
    float cx=0.f, cy=0.f, cz=0.f;
    for (int i=0;i<nb;i++){
        cx += atoms_r[i].x;
        cy += atoms_r[i].y;
        cz += atoms_r[i].z;
    }
    cx/=nb; cy/=nb; cz/=nb;

    // Normalize dipole direction
    float nx = (float)(dip[0] / mag);
    float ny = (float)(dip[1] / mag);
    float nz = (float)(dip[2] / mag);

    // Shift origin backward so arrow does not overlap molecule
    float offset = 1.2f;  // adjust if needed
    float sx = cx - nx * offset;
    float sy = cy - ny * offset;
    float sz = cz - nz * offset;

    // Arrow length scaling (controls visibility)
    float scale = 2.0f;
    float ex = sx + (float)(dip[0] * scale);
    float ey = sy + (float)(dip[1] * scale);
    float ez = sz + (float)(dip[2] * scale);

    // Direction vector
    float dx = ex - sx;
    float dy = ey - sy;
    float dz = ez - sz;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len < 0.05f) return;

    glDisable(GL_LIGHTING);

    // ================= BODY (CYLINDER) =================
    GLUquadric *q = gluNewQuadric();

    // Compute rotation to align cylinder with dipole vector
    float ax = -dy;
    float ay = dx;
    float angle = (180.0f/M_PI) * acosf(dz / len);

    if (fabsf(dz/len) > 0.9999f) {
        ax = 1.0f; ay = 0.0f;
        angle = (dz < 0) ? 180.0f : 0.0f;
    }

    glPushMatrix();
    glTranslatef(sx, sy, sz);
    glRotatef(angle, ax, ay, 0.0f);

    // Strong red color for visibility
    glColor3f(1.0f, 0.5f, 0.0f);

    // Cylinder body (real thickness control)
    gluCylinder(q, 0.1f, 0.1f, len, 16, 1);

    // ================= ARROWHEAD =================
    glTranslatef(0.0f, 0.0f, len);

    // Slightly larger cone for clear direction
    gluCylinder(q, 0.28f, 0.0f, 0.45f, 18, 1);

    glPopMatrix();
    gluDeleteQuadric(q);

    glEnable(GL_LIGHTING);
}




// ====================== ORBITAL VOLUME RENDERER (KEY V / H / L) ======================
void drawOrbitalVolume() {
    int nb = nb_r;
    int occ = occ_r;
    if (nb < 1 || occ < 1) return;

    // Nova lógica unificada (0=off, 1=V, 2=H, 3=L)
    if (show_homo_lumo == 0) return;

    // Centro da molécula
    float cx=0.0f, cy=0.0f, cz=0.0f;
    for (int i=0; i<nb; i++) {
        cx += atoms_r[i].x;
        cy += atoms_r[i].y;
        cz += atoms_r[i].z;
    }
    cx /= nb; cy /= nb; cz /= nb;

    const int NG = 60;
    const float GEXT = 4.0f;
    float step = 2.0f * GEXT / (NG - 1);

    static const float homo_col[2][6] = {
        {1.00f,0.55f,0.05f,  0.90f,0.15f,0.05f}, // HOMO (+/-)
        {0.10f,0.90f,1.00f,  0.05f,0.30f,1.00f}  // LUMO (+/-)
    };

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glPointSize(3.0f);
    glBegin(GL_POINTS);

    for (int ix = 0; ix < NG; ix++) {
        for (int iy = 0; iy < NG; iy++) {
            for (int iz = 0; iz < NG; iz++) {

                float jitter = step * 0.15f;
                float rx = cx + (-GEXT + ix * step) + ((rand() / (float)RAND_MAX - 0.5f) * jitter);
                float ry = cy + (-GEXT + iy * step) + ((rand() / (float)RAND_MAX - 0.5f) * jitter);
                float rz = cz + (-GEXT + iz * step) + ((rand() / (float)RAND_MAX - 0.5f) * jitter);

                if (show_homo_lumo == 1) {
                    // =========================================================
                    // MODO 1 → DENSIDADE TOTAL (V)
                    // =========================================================
                    double rho = 0.0;
                    for (int k = 0; k < occ; k++) {
                        double psi = 0.0;
                        for (int A = 0; A < nb; A++) {
                            double cAk = mo_coeff_r[A][k];
                            double dx = rx - atoms_r[A].x;
                            double dy = ry - atoms_r[A].y;
                            double dz = rz - atoms_r[A].z;
                            double r2 = dx*dx + dy*dy + dz*dz;

                            double chi = 0.0;
                            for (int p = 0; p < NP; p++) {
                                double alp = g_a_exp[A][p];
                                chi += STO3G_d[p] * norm_gauss(alp) * exp(-alp * r2);
                            }
                            psi += cAk * chi;
                        }
                        rho += psi * psi;
                    }

                    float rho_f = (float)rho;
                    if (rho_f < 5e-3f) continue;

                    float alpha = rho_f * 30.0f;
                    if (alpha > 0.7f) alpha = 0.7f;
                    if (alpha < 0.02f) continue;

                    glColor4f(0.8f, 0.9f, 1.0f, alpha);
                    glVertex3f(rx, ry, rz);
                }
                else if (show_homo_lumo == 2) {
                    // =========================================================
                    // MODO 2 → HOMO only
                    // =========================================================
                    int k_homo = occ - 1;
                    double psi = 0.0;
                    for (int A = 0; A < nb; A++) {
                        double cAk = mo_coeff_r[A][k_homo];
                        double dx = rx - atoms_r[A].x;
                        double dy = ry - atoms_r[A].y;
                        double dz = rz - atoms_r[A].z;
                        double r2 = dx*dx + dy*dy + dz*dz;

                        double chi = 0.0;
                        for (int p = 0; p < NP; p++) {
                            double alp = g_a_exp[A][p];
                            chi += STO3G_d[p]*norm_gauss(alp)*exp(-alp*r2);
                        }
                        psi += cAk * chi;
                    }

                    float psi2 = (float)(psi*psi);
                    if (psi2 > 5e-4f) {
                        float alpha = psi2 * 25.0f;
                        if (alpha > 0.6f) alpha = 0.6f;

                        if (psi > 0.0)
                            glColor4f(homo_col[0][0],homo_col[0][1],homo_col[0][2],alpha);
                        else
                            glColor4f(homo_col[0][3],homo_col[0][4],homo_col[0][5],alpha);

                        glVertex3f(rx, ry, rz);
                    }
                }
                else if (show_homo_lumo == 3) {
                    // =========================================================
                    // MODO 3 → LUMO only
                    // =========================================================
                    int k_lumo = occ; 
                    if (k_lumo >= nb) continue;

                    double psi = 0.0;
                    for (int A = 0; A < nb; A++) {
                        double cAk = mo_coeff_r[A][k_lumo];
                        double dx = rx - atoms_r[A].x;
                        double dy = ry - atoms_r[A].y;
                        double dz = rz - atoms_r[A].z;
                        double r2 = dx*dx + dy*dy + dz*dz;

                        double chi = 0.0;
                        for (int p = 0; p < NP; p++) {
                            double alp = g_a_exp[A][p];
                            chi += STO3G_d[p] * norm_gauss(alp) * exp(-alp * r2);
                        }
                        psi += cAk * chi;
                    }

                    float psi2 = (float)(psi * psi);
                    if (psi2 > 5e-4f) {
                        float alpha = psi2 * 25.0f;
                        if (alpha > 0.6f) alpha = 0.6f;

                        if (psi > 0.0)
                            glColor4f(homo_col[1][0], homo_col[1][1], homo_col[1][2], alpha);
                        else
                            glColor4f(homo_col[1][3], homo_col[1][4], homo_col[1][5], alpha);

                        glVertex3f(rx, ry, rz);
                    }
                }
            }
        }
    }

    glEnd();
    glPointSize(1.0f);
    glDepthMask(GL_TRUE);
    glEnable(GL_LIGHTING);
}

// ====================== PANEL O: ORBITAL DIAGRAM ======================
void drawOrbitalDiagram() {
    int n=nb_r; if (n<1) return;
    float x0=1.92f, y0=-2.58f, gw=1.05f, gh=2.75f;
    glColor4f(0.02f,0.06f,0.18f,0.93f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.2f,0.5f,0.85f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.3f,0.85f,1.0f);
    drawText(x0+0.04f,y0+gh-0.11f,"Orbitals [O]",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.4f,0.4f,0.55f);
    drawText(x0+0.04f,y0+gh-0.22f,"occ=green  virt=grey",GLUT_BITMAP_HELVETICA_10);

    double emin_ev=orbital_eps_r[0]*27.2114, emax_ev=orbital_eps_r[n-1]*27.2114;
    double erange=emax_ev-emin_ev; if (erange<0.5) erange=0.5;
    float ystart=y0+0.32f, yend=y0+gh-0.32f, xcenter=x0+gw*0.5f;
    char buf[64];

    for (int i=0;i<n;i++) {
        double ev=orbital_eps_r[i]*27.2114;
        float yp=ystart+(float)((ev-emin_ev)/erange)*(yend-ystart);
        int is_occ=(i<occ_r);
        if (is_occ) glColor3f(0.15f,1.0f,0.45f); else glColor3f(0.45f,0.45f,0.55f);
        glLineWidth(2.0f);
        glBegin(GL_LINES); glVertex2f(xcenter-0.22f,yp); glVertex2f(xcenter+0.22f,yp); glEnd();
        glLineWidth(1.0f);
        if (is_occ) {
            glColor3f(1.0f,0.95f,0.2f);
            glBegin(GL_LINES); glVertex2f(xcenter-0.10f,yp+0.001f); glVertex2f(xcenter-0.10f,yp+0.09f); glEnd();
            glBegin(GL_TRIANGLES); glVertex2f(xcenter-0.10f,yp+0.12f); glVertex2f(xcenter-0.13f,yp+0.08f); glVertex2f(xcenter-0.07f,yp+0.08f); glEnd();
            glBegin(GL_LINES); glVertex2f(xcenter+0.04f,yp+0.12f); glVertex2f(xcenter+0.04f,yp+0.03f); glEnd();
            glBegin(GL_TRIANGLES); glVertex2f(xcenter+0.04f,yp+0.001f); glVertex2f(xcenter+0.01f,yp+0.05f); glVertex2f(xcenter+0.07f,yp+0.05f); glEnd();
        }
        // NEW: highlight HOMO/LUMO labels
        if (i == occ_r-1)      { glColor3f(1.0f,0.6f,0.1f); sprintf(buf,"HOMO"); drawText(x0+0.04f,yp+0.03f,buf,GLUT_BITMAP_HELVETICA_10); }
        else if (i == occ_r)   { glColor3f(0.1f,0.85f,1.0f); sprintf(buf,"LUMO"); drawText(x0+0.04f,yp+0.03f,buf,GLUT_BITMAP_HELVETICA_10); }

        glColor3f(0.75f,0.85f,0.95f);
        sprintf(buf,"%+.2f eV",ev); drawText(xcenter+0.24f,yp-0.025f,buf,GLUT_BITMAP_HELVETICA_10);
        glColor3f(0.5f,0.55f,0.65f);
        sprintf(buf,"MO%d",i+1); drawText(x0+0.27f,yp-0.025f,buf,GLUT_BITMAP_HELVETICA_10);
    }
    glColor3f(0.25f,0.3f,0.4f);
    glBegin(GL_LINES); glVertex2f(xcenter,ystart-0.05f); glVertex2f(xcenter,yend+0.05f); glEnd();
}

// ====================== PANEL M: MULLIKEN ======================
void drawMulliken() {
    float x0=-2.95f, y0=0.42f, gw=1.5f, rowh=0.28f;
    float gh=0.38f+rowh*(NUM_ATOMS+1);
    glColor4f(0.02f,0.10f,0.05f,0.92f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.2f,0.75f,0.35f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.25f,1.0f,0.5f);
    drawText(x0+0.05f,y0+gh-0.13f,"Mulliken Populations [M]",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.35f,0.55f,0.4f);
    drawText(x0+0.05f,y0+gh-0.26f,"Atom  Gross Pop   Net Charge",GLUT_BITMAP_HELVETICA_10);
    char buf[64];
    float bar_max=0.45f;
    for (int i=0;i<NUM_ATOMS;i++) {
        float yp=y0+gh-0.38f-rowh*(i+1);
        double gross=(double)g_Z[i]-mulliken_q_r[i];
        double q_net=mulliken_q_r[i];
        glColor3f(0.9f,0.9f,0.5f);
        sprintf(buf,"%s%d",g_atom_symbol[i],i+1);
        drawText(x0+0.05f,yp,buf,GLUT_BITMAP_HELVETICA_10);
        float blen=(float)(gross/(double)g_Z[i])*bar_max;
        if (blen<0) blen=0; if (blen>bar_max) blen=bar_max;
        glColor4f(0.15f,0.75f,0.3f,0.75f);
        glBegin(GL_QUADS);
        glVertex2f(x0+0.38f,yp-0.01f); glVertex2f(x0+0.38f+blen,yp-0.01f);
        glVertex2f(x0+0.38f+blen,yp+0.14f); glVertex2f(x0+0.38f,yp+0.14f);
        glEnd();
        glColor3f(0.7f,0.9f,0.75f);
        sprintf(buf,"%.3f",gross); drawText(x0+0.42f,yp,buf,GLUT_BITMAP_HELVETICA_10);
        if (q_net>0.01f) glColor3f(1.0f,0.4f,0.4f);
        else if (q_net<-0.01f) glColor3f(0.4f,0.6f,1.0f);
        else glColor3f(0.7f,0.85f,0.7f);
        sprintf(buf,"%+.3f e",q_net); drawText(x0+0.90f,yp,buf,GLUT_BITMAP_HELVETICA_10);
    }
}

// ====================== PANEL E: ENERGY DECOMPOSITION ======================
void drawEnergyDecomp() {
    float x0=-2.95f,y0=-0.95f,gw=1.82f,gh=1.35f;
    glColor4f(0.08f,0.03f,0.15f,0.92f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.6f,0.3f,0.85f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.75f,0.45f,1.0f);
    drawText(x0+0.05f,y0+gh-0.13f,"Energy Decomposition [E]",GLUT_BITMAP_HELVETICA_10);
    double E_total=E_kin_r+E_vne_r+E_vee_r+E_nuc_r;
    const char *labels[5]={"T   (e. kinetic)","Vne (nuc. attr.)","Vee (e-e repuls.)","Vnn (nuc. repuls.)","E_total"};
    double vals[5]={E_kin_r,E_vne_r,E_vee_r,E_nuc_r,E_total};
    float colors[5][3]={{0.3f,0.8f,1.0f},{1.0f,0.5f,0.2f},{1.0f,0.8f,0.2f},{0.6f,0.9f,0.6f},{0.85f,0.85f,0.95f}};
    double abs_max=0.01;
    for (int i=0;i<5;i++) if (fabs(vals[i])>abs_max) abs_max=fabs(vals[i]);
    float bar_zone=0.55f, rowh=0.21f;
    char buf[80];
    for (int i=0;i<5;i++) {
        float yp=y0+gh-0.28f-rowh*(i+1);
        if (i==4){glColor3f(0.4f,0.4f,0.5f); glBegin(GL_LINES); glVertex2f(x0+0.03f,yp+rowh-0.02f); glVertex2f(x0+gw-0.03f,yp+rowh-0.02f); glEnd();}
        glColor3f(colors[i][0],colors[i][1],colors[i][2]);
        drawText(x0+0.05f,yp,labels[i],GLUT_BITMAP_HELVETICA_10);
        float midx=x0+1.12f, blen=(float)(vals[i]/abs_max)*bar_zone;
        float bx_lo=(blen>=0)?midx:midx+blen, bx_hi=(blen>=0)?midx+blen:midx;
        if (fabs(blen)>0.003f){
            glColor4f(colors[i][0],colors[i][1],colors[i][2],0.55f);
            glBegin(GL_QUADS);
            glVertex2f(bx_lo,yp+0.01f); glVertex2f(bx_hi,yp+0.01f);
            glVertex2f(bx_hi,yp+0.15f); glVertex2f(bx_lo,yp+0.15f);
            glEnd();
        }
        glColor3f(colors[i][0]*0.85f+0.15f,colors[i][1]*0.85f+0.15f,colors[i][2]*0.85f+0.15f);
        sprintf(buf,"%+.5f Ha",vals[i]);
        drawText(x0+gw-0.72f,yp,buf,GLUT_BITMAP_HELVETICA_10);
    }
}

// ====================== NEW: PANEL D — ELECTRIC DIPOLE ======================
// Shows the dipole moment vector in Debye + magnitude + a small 2D arrow diagram.
void drawDipolePanel() {
    float x0=-2.95f, y0=1.68f, gw=1.82f, gh=0.95f;

    // Background
    glColor4f(0.10f,0.09f,0.02f,0.93f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    // Border (yellow)
    glColor3f(0.9f,0.82f,0.1f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();

    // Title
    glColor3f(1.0f,0.92f,0.2f);
    drawText(x0+0.05f,y0+gh-0.13f,"Electric Dipole [Y]",GLUT_BITMAP_HELVETICA_10);

    double dx_D = dipole_r[0]*2.5418;
    double dy_D = dipole_r[1]*2.5418;
    double dz_D = dipole_r[2]*2.5418;
    double mag   = sqrt(dx_D*dx_D + dy_D*dy_D + dz_D*dz_D);

    char buf[80];
    glColor3f(0.95f,0.88f,0.5f);
    sprintf(buf,"mu_x : %+.4f D",dx_D);  drawText(x0+0.05f,y0+gh-0.28f,buf,GLUT_BITMAP_HELVETICA_10);
    sprintf(buf,"mu_y : %+.4f D",dy_D);  drawText(x0+0.05f,y0+gh-0.43f,buf,GLUT_BITMAP_HELVETICA_10);
    sprintf(buf,"mu_z : %+.4f D",dz_D);  drawText(x0+0.05f,y0+gh-0.58f,buf,GLUT_BITMAP_HELVETICA_10);

    glColor3f(1.0f,0.95f,0.3f);
    sprintf(buf,"|mu| : %.4f D",mag);  drawText(x0+0.05f,y0+gh-0.74f,buf,GLUT_BITMAP_HELVETICA_12);

    // Mini 2D bar representing |μ| (max 4 D = full bar)
    float bar_max=0.80f;
    float blen=(float)(mag/4.0)*bar_max; if (blen>bar_max) blen=bar_max;
    glColor4f(1.0f,0.85f,0.1f,0.65f);
    glBegin(GL_QUADS);
    glVertex2f(x0+0.92f,y0+gh-0.80f); glVertex2f(x0+0.92f+blen,y0+gh-0.80f);
    glVertex2f(x0+0.92f+blen,y0+gh-0.66f); glVertex2f(x0+0.92f,y0+gh-0.66f);
    glEnd();
    glColor3f(0.5f,0.4f,0.1f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0+0.92f,y0+gh-0.80f); glVertex2f(x0+0.92f+bar_max,y0+gh-0.80f);
    glVertex2f(x0+0.92f+bar_max,y0+gh-0.66f); glVertex2f(x0+0.92f,y0+gh-0.66f);
    glEnd();
    glColor3f(0.5f,0.45f,0.2f);
    drawText(x0+0.92f,y0+gh-0.90f,"0          4D",GLUT_BITMAP_HELVETICA_10);
}

// ====================== NEW: PANEL I — STATISTICS ======================
// Shows HOMO-LUMO gap, electronic virial ratio, and MD kinetic vs potential breakdown.

// ====================== STATUS PANEL (WITH STATISTICS) ======================
void drawStatsPanel(float x0, float y0) {

    const float gw = 1.04f;
    const float gh = 2.72f;   // aumentado para incluir statistics
    const float dy = 0.21f;

    const float pad = 0.10f;
    const float px = x0 + pad;
    const float py = y0 + gh - 0.28f;

    char buf[128];

    // ================= BACKGROUND =================
    if (white_background)
        glColor4f(0.02f, 0.02f, 0.02f, 0.95f);
    else
        glColor4f(0.03f, 0.03f, 0.09f, 0.94f);

    glBegin(GL_QUADS);
        glVertex2f(x0,       y0);
        glVertex2f(x0 + gw,  y0);
        glVertex2f(x0 + gw,  y0 + gh);
        glVertex2f(x0,       y0 + gh);
    glEnd();

    // ================= BORDER =================
    glLineWidth(1.4f);
    glColor3f(0.35f, 0.65f, 1.0f);
    glBegin(GL_LINE_STRIP);
        glVertex2f(x0,       y0);
        glVertex2f(x0 + gw,  y0);
        glVertex2f(x0 + gw,  y0 + gh);
        glVertex2f(x0,       y0 + gh);
        glVertex2f(x0,       y0);
    glEnd();
    glLineWidth(1.0f);

    // ================= TITLE =================
    glColor3f(0.45f, 0.85f, 1.0f);
    drawText(px, y0 + gh - 0.12f, "Simulation Status", GLUT_BITMAP_HELVETICA_10);

    // ================= STEP =================
    glColor3f(1.0f, 0.85f, 0.0f);
    sprintf(buf, "%-9s %d", "Step", step_r);
    drawText(px, py, buf, GLUT_BITMAP_HELVETICA_12);

    // ================= TIME =================
    glColor3f(0.8f, 0.8f, 0.9f);
    sprintf(buf, "%-9s %.4f fs", "Time", step_r * dt * 0.02418884f);
    drawText(px, py - dy, buf, GLUT_BITMAP_HELVETICA_12);

    // ================= ENERGY =================
    glColor3f(0.2f, 1.0f, 0.5f);
    sprintf(buf, "%-9s %+.5f Ha", "Energy", E_r);
    drawText(px, py - 2*dy, buf, GLUT_BITMAP_HELVETICA_12);

    sprintf(buf, "   (%+.4f eV)", E_r * 27.2114);
    drawText(px, py - 3*dy, buf, GLUT_BITMAP_HELVETICA_12);

    // ================= TEMPERATURE =================
    glColor3f(1.0f, 0.4f, 0.3f);
    sprintf(buf, "%-9s %.1f K", "Temp", T_r);
    drawText(px, py - 4*dy, buf, GLUT_BITMAP_HELVETICA_12);

    glColor3f(0.7f, 0.35f, 0.25f);
    sprintf(buf, "%-9s %.0f K", "Target", T_target);
    drawText(px, py - 5*dy, buf, GLUT_BITMAP_HELVETICA_12);

    // ================= PRESSURE =================
    glColor3f(0.4f, 0.7f, 1.0f);
    sprintf(buf, "%-9s %+.5f a.u.", "Pressure", P_r);
    drawText(px, py - 6*dy, buf, GLUT_BITMAP_HELVETICA_12);

    // ================= DISTANCE =================
    glColor3f(1.0f, 0.8f, 0.2f);
    sprintf(buf, "%-9s %.4f bohr", "Distance", bond_r);
    drawText(px, py - 7*dy, buf, GLUT_BITMAP_HELVETICA_12);

    sprintf(buf, "   (%.4f Ang)", bond_r * 0.529177f);
    drawText(px, py - 8*dy, buf, GLUT_BITMAP_HELVETICA_12);

    // ================= REFERENCE =================
    const char *ref;
    if      (molecule_type == 1) ref = "HeH+  ~1.46 bohr";
    else if (molecule_type == 2) ref = "He-He (no bond)";
    else                         ref = "H2    ~0.74 Ang";

    glColor3f(0.55f, 0.8f, 1.0f);
    sprintf(buf, "%-9s %s", "Ref", ref);
    drawText(px, py - 9*dy, buf, GLUT_BITMAP_HELVETICA_10);

    // ================= SEPARATOR =================
    float ysep = py - 10*dy;
  
    // ================= GAP =================
    glColor3f(0.3f, 0.85f, 1.0f);
    sprintf(buf, "%-9s %.4f Ha (%.3f eV)", "Gap",
            homo_lumo_gap_r, homo_lumo_gap_r * 27.2114);
    drawText(px, ysep, buf, GLUT_BITMAP_HELVETICA_10);

    // ================= VIRIAL =================
    float yv = ysep - dy;
    double vr = virial_ratio_r;

    if      (fabs(vr - 1.0) < 0.005) glColor3f(0.2f, 1.0f, 0.3f);
    else if (fabs(vr - 1.0) < 0.02)  glColor3f(1.0f, 0.85f, 0.1f);
    else                             glColor3f(1.0f, 0.3f, 0.2f);

    sprintf(buf, "%-9s %.4f (ideal = 1.000)", "Virial", vr);
    drawText(px, yv, buf, GLUT_BITMAP_HELVETICA_10);
    
}


// ====================== ENERGY GRAPH ======================
void drawEnergyGraph(float x0, float y0) {
    int n=(hcount_r<HISTORY_LEN)?hcount_r:HISTORY_LEN;
    if (n<2) return;
    int start=(hcount_r>HISTORY_LEN)?(hcount_r%HISTORY_LEN):0;
    double emin=ehist_r[start], emax=emin;
    for (int i=0;i<n;i++){int idx=(start+i)%HISTORY_LEN; if(ehist_r[idx]<emin)emin=ehist_r[idx]; if(ehist_r[idx]>emax)emax=ehist_r[idx];}
    double er=emax-emin; if(er<1e-12) er=1e-12;
    float gw=1.25f, gh=0.58f;
    glColor4f(0,0.05f,0.15f,0.85f);
    glBegin(GL_QUADS); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.25f,0.45f,0.7f);
    glBegin(GL_LINE_LOOP); glVertex2f(x0,y0); glVertex2f(x0+gw,y0); glVertex2f(x0+gw,y0+gh); glVertex2f(x0,y0+gh); glEnd();
    glColor3f(0.65f,0.85f,1.0f);
    drawText(x0+0.04f,y0+gh-0.1f,"E(HF) Ha vs step",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.1f,1.0f,0.55f); glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (int i=0;i<n;i++){
        int idx=(start+i)%HISTORY_LEN;
        float xp=x0+0.05f+(float)i/(n-1)*(gw-0.1f);
        float yp=y0+0.06f+(float)((ehist_r[idx]-emin)/er)*(gh-0.18f);
        glVertex2f(xp,yp);
    }
    glEnd(); glLineWidth(1.0f);
}



// ====================== HUD ======================
void drawHUD() {
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(-3,3,-3,3);
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity();
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST);
    char buf[256];

    // Main title
    glColor3f(0.3f,0.85f,1.0f);
    drawText(-2.9f,2.72f,"Ab initio MD  -  Hartree-Fock / STO-3G",GLUT_BITMAP_HELVETICA_18);

    char sys_name[64];
    if (molecule_type == 1) {
        if (N_heh_pairs == 1) strcpy(sys_name, "HeH+  (1 pair, charge +1)");
        else sprintf(sys_name, "%d x HeH+  (%d atoms, charge +%d)", N_heh_pairs, NUM_ATOMS, N_heh_pairs);
    } else if (molecule_type == 2) {
        sprintf(sys_name, "%d He atom(s)  (neutral)", NUM_ATOMS);
    } else {
        sprintf(sys_name, "%d H atom(s)", NUM_ATOMS);
    }

    glColor3f(0.5f,0.5f,0.62f);
    sprintf(buf,"System: %s | %d electrons ", sys_name, NUM_ELECTRONS);
    drawText(-2.9f,2.47f,buf,GLUT_BITMAP_HELVETICA_12);

    // NEW: PAUSED indicator
    if (md_paused) {
        glColor3f(1.0f,0.85f,0.0f);
        drawText(-2.9f,2.22f,"[ PAUSED — press P or SPACE to resume, ENTER = single step ]",GLUT_BITMAP_HELVETICA_12);
    } else if (computing) {
        glColor3f(1.0f,0.5f,0.0f);
        drawText(-2.9f,2.22f,"[ COMPUTING SCF... ]",GLUT_BITMAP_HELVETICA_12);
    }

    // Right-side status readouts


    // Bottom-left: quantum method legend
    glColor3f(0.45f,0.45f,0.75f); drawText(-2.9f,-1.72f,"[ Quantum Method ]",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.5f,0.5f,0.62f);
    drawText(-2.9f,-1.89f,"Theory    : Hartree-Fock (RHF, closed-shell)",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.04f,"Basis     : STO-3G (3 Gaussians per AO)",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.19f,"Gradient  : Central finite difference (d=0.001 a.u.)",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.34f,"Integrator: Velocity Verlet",GLUT_BITMAP_HELVETICA_10);
    drawText(-2.9f,-2.49f,"Thermostat: Langevin  |  Barostat: Berendsen",GLUT_BITMAP_HELVETICA_10);
    glColor3f(0.38f,0.38f,0.48f);
    drawText(-2.9f,-2.65f,"W/S:zoom  A/D:pan  Z/X:up/dn  +/-:temp  P:pause  ENTER:step",GLUT_BITMAP_HELVETICA_10);

    // Key indicator row (11 buttons, split across 2 rows)
    // Key indicator row - Nova lógica (V=1, H=2, L=3)
    float kx=-2.9f, ky=-2.80f, ksp=0.515f;
    const char *key_labels[12] = {
        "[O]Orbs", "[M]Mull", "[E]Enrg", "[F]Forc",
        "[V]Total", "[H]HOMO", "[L]LUMO", "[Y]Dip", 
        "[I]Stat", "[P]Paus", "[Q]Rst", "[B]Bg"
    };

    float key_colors[12][3] = {
        {0.3f,0.85f,1.0f}, {0.25f,1.0f,0.5f}, {0.75f,0.4f,1.0f},
        {1.0f,0.25f,0.25f}, {0.8f,0.7f,1.0f}, {1.0f,0.5f,0.9f},
        {1.0f,0.92f,0.2f}, {0.2f,0.9f,0.9f}, {0.5f,1.0f,0.4f}, 
        {0.7f,0.7f,0.7f}, {0.8f,0.8f,0.8f}, {1.0f,0.9f,0.3f}
    };

    for (int i = 0; i < 12; i++) {
        int active = 0;

        if (i == 0)      active = show_orbital;
        else if (i == 1) active = show_mulliken;
        else if (i == 2) active = show_energy_dec;
        else if (i == 3) active = show_forces;
        else if (i == 4) active = (show_homo_lumo == 1);   // V - Total Density
        else if (i == 5) active = (show_homo_lumo == 2);   // H - HOMO
        else if (i == 6) active = (show_homo_lumo == 3);   // L - LUMO
        else if (i == 7) active = show_dipole;
        else if (i == 8) active = show_stats;
        else if (i == 9) active = md_paused;
        else if (i == 11) active = white_background; //  NOVO


        if (active)
            glColor3f(key_colors[i][0], key_colors[i][1], key_colors[i][2]);
        else
            glColor3f(0.3f, 0.3f, 0.38f);

        drawText(kx + ksp * i, ky, key_labels[i], GLUT_BITMAP_HELVETICA_10);
    }

    drawEnergyGraph(-2.95f,-1.55f);

    if (show_orbital)    drawOrbitalDiagram();
    if (show_mulliken)   drawMulliken();
    if (show_energy_dec) drawEnergyDecomp();
    if (show_dipole)     drawDipolePanel();
    if (show_stats)      drawStatsPanel(1.92f, 0.2f);;

    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glMatrixMode(GL_MODELVIEW);  glPopMatrix();
}

// ====================== DISPLAY ======================
void display(void) {


    // --- NEW: background toggle ---
    if (white_background)
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // branco
    else
        glClearColor(0.04f, 0.04f, 0.09f, 1.0f); // seu fundo original

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(obs_x,obs_y,obs_z,0,0,0,0,1,0);

    struct Atom snap[MAX_ATOM]; float snap_box;
    pthread_mutex_lock(&render_mutex);
    memcpy(snap,atoms_r,NUM_ATOMS*sizeof(struct Atom)); snap_box=box_size;
    pthread_mutex_unlock(&render_mutex);

    // Simulation box wireframe
    glDisable(GL_LIGHTING);
    {
        float h=snap_box/2.0f;
        glColor4f(0.35f,0.55f,0.8f,0.35f); glLineWidth(1.2f);
        glBegin(GL_LINE_LOOP); glVertex3f(-h,-h,-h); glVertex3f(h,-h,-h); glVertex3f(h,h,-h); glVertex3f(-h,h,-h); glEnd();
        glBegin(GL_LINE_LOOP); glVertex3f(-h,-h,h);  glVertex3f(h,-h,h);  glVertex3f(h,h,h);  glVertex3f(-h,h,h);  glEnd();
        glBegin(GL_LINES);
        glVertex3f(-h,-h,-h); glVertex3f(-h,-h,h); glVertex3f(h,-h,-h); glVertex3f(h,-h,h);
        glVertex3f(h,h,-h);   glVertex3f(h,h,h);   glVertex3f(-h,h,-h); glVertex3f(-h,h,h);
        glEnd();
    }

    // Bonds
    for (int i=0;i<NUM_ATOMS;i++) for (int j=i+1;j<NUM_ATOMS;j++){
        float dd=atomDistance(snap[i],snap[j]);
        if (dd<BOND_CUTOFF && !show_dipole) drawBond(snap[i].x,snap[i].y,snap[i].z,snap[j].x,snap[j].y,snap[j].z);
    }

    // Force arrows
    if (show_forces) for (int i=0;i<NUM_ATOMS;i++){
        float fx=snap[i].ax*snap[i].mass, fy=snap[i].ay*snap[i].mass, fz=snap[i].az*snap[i].mass;
        drawForceArrow(snap[i].x,snap[i].y,snap[i].z,fx,fy,fz);
    }

    // NEW: Dipole arrow in 3D scene
    if (show_dipole) drawDipoleArrow3D();

    // Atoms
    glEnable(GL_LIGHTING);
    for (int i=0;i<NUM_ATOMS;i++){
        if (g_Z[i]==2) glColor3f(0.95f,0.90f,0.35f);
        else           glColor3f(0.82f,0.82f,0.95f);
        glPushMatrix();
        glTranslatef(snap[i].x,snap[i].y,snap[i].z);
        glutSolidSphere(snap[i].radius,slices,stacks);
        glPopMatrix();
    }

    // Orbital volume cloud (V H or L mode)
    if (show_homo_lumo != 0) drawOrbitalVolume();

    drawHUD();
    glutSwapBuffers();
}

static void idle(void){glutPostRedisplay();}

// ====================== INITIALIZE POSITIONS ======================
void initializePositions() {
    const float kB = 3.1668114e-6f;
    if (NUM_ATOMS > 4) box_size = 5.0f + (NUM_ATOMS - 4) * 1.5f;

    if (molecule_type == 1) {
        float spacing = 5.0f;
        for (int pair = 0; pair < N_heh_pairs; pair++) {
            int he_idx = pair * 2;
            int h_idx  = pair * 2 + 1;
            float cx = (pair - (N_heh_pairs - 1) / 2.0f) * spacing;
            float cy = 0.0f, cz = 0.0f;
            if (N_heh_pairs > 3) {
                int cols = (int)ceilf(sqrtf((float)N_heh_pairs));
                int row  = pair / cols, col = pair % cols;
                cx = (col - (cols - 1) / 2.0f) * spacing;
                cy = (row - (int)(N_heh_pairs / cols) / 2.0f) * spacing;
            }
            atoms[he_idx].x = cx - 0.73f; atoms[he_idx].y = cy; atoms[he_idx].z = cz;
            atoms[h_idx].x  = cx + 0.73f; atoms[h_idx].y  = cy; atoms[h_idx].z  = cz;
            atoms[he_idx].mass   = 4.0f * proton_mass;
            atoms[he_idx].radius = 0.35f;
            atoms[h_idx].mass    = proton_mass;
            atoms[h_idx].radius  = atom_radius;
        }
    } else if (molecule_type == 2) {
        int placed = 0;
        for (int i = 0; i < NUM_ATOMS; i++) {
            struct Atom *e = &atoms[i];
            e->mass = 4.0f * proton_mass; e->radius = 0.35f;
            do {
                e->x = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                e->y = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                e->z = ((float)(rand()%2000)/2000.0f*box_size) - box_size/2.0f;
                int col = 0;
                for (int j = 0; j < placed; j++)
                    if (atomDistance(*e, atoms[j]) < 2.5f) { col = 1; break; }
                if (!col) break;
            } while (1);
            placed++;
        }
    } else {
        int placed = 0;
        for (int i = 0; i < NUM_ATOMS; i++) {
            struct Atom *e = &atoms[i];
            e->mass = proton_mass; e->radius = atom_radius;
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

    // Maxwell-Boltzmann velocities
    for (int i = 0; i < NUM_ATOMS; i++) {
        struct Atom *e = &atoms[i];
        float sigma = sqrtf(kB * T_target / e->mass);
        e->vx_old = sigma * (float)gauss_rand();
        e->vy_old = sigma * (float)gauss_rand();
        e->vz_old = sigma * (float)gauss_rand();
    }

    // Zero center-of-mass momentum
    float v_cm[3] = {0,0,0};
    for (int i = 0; i < NUM_ATOMS; i++) {
        v_cm[0] += atoms[i].vx_old; v_cm[1] += atoms[i].vy_old; v_cm[2] += atoms[i].vz_old;
    }
    for (int i = 0; i < NUM_ATOMS; i++) {
        atoms[i].vx_old -= v_cm[0]/NUM_ATOMS;
        atoms[i].vy_old -= v_cm[1]/NUM_ATOMS;
        atoms[i].vz_old -= v_cm[2]/NUM_ATOMS;
    }

    compute_hf_accelerations(atoms, NUM_ATOMS, &E_hf_cached);

    min_bond = 1e9f;
    for (int i = 0; i < NUM_ATOMS; i++)
        for (int j = i+1; j < NUM_ATOMS; j++) {
            float d = atomDistance(atoms[i], atoms[j]);
            if (d < min_bond) min_bond = d;
        }

    energy_history[0] = E_hf_cached; history_count = 1;
    memcpy(atoms_r, atoms, sizeof(atoms)); E_r = E_hf_cached; bond_r = min_bond;
    memcpy(orbital_eps_r, g_orbital_eps, sizeof(g_orbital_eps));
    memcpy(mo_coeff_r,    g_mo_coeff,    sizeof(g_mo_coeff));
    memcpy(mulliken_q_r,  g_mulliken_q,  sizeof(g_mulliken_q));
    E_kin_r = g_E_kin; E_vne_r = g_E_vne; E_vee_r = g_E_vee; E_nuc_r = g_E_nuc;
    nb_r = g_nb; occ_r = g_occ;
    for (int k=0;k<3;k++) dipole_r[k] = g_dipole[k];
    homo_lumo_gap_r = g_homo_lumo_gap;
    virial_ratio_r  = g_virial_ratio;
}

// ====================== OPENGL INIT ======================
void init_gl(void){
    glClearColor(0.04f,0.04f,0.09f,0.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

void reshape(int w,int h){
    glViewport(0,0,(GLsizei)w,(GLsizei)h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluPerspective(60,(GLfloat)w/(GLfloat)h,0.1,50.0);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
}

// ====================== KEYBOARD ======================
// Handles user input for camera, simulation control, and visualization toggles.
void key(unsigned char k,int x,int y){
    switch(k){
        // ================= CAMERA CONTROL =================
        case 'w': case 'W': obs_z-=0.2; break;
        case 's': case 'S': obs_z+=0.2; break;
        case 'a': case 'A': obs_x-=0.2; break;
        case 'd': case 'D': obs_x+=0.2; break;
        case 'x': case 'X': obs_y-=0.2; break;
        case 'z': case 'Z': obs_y+=0.2; break;

        // ================= TEMPERATURE CONTROL =================
        case '+':
            T_target+=50.0f;
            printf("T_target: %.0f K\n",T_target);
            break;

        case '-':
            T_target=(T_target>50)?T_target-50:50;
            printf("T_target: %.0f K\n",T_target);
            break;

        // ================= BASIC PANELS =================
        case 'o': case 'O':
            show_orbital=!show_orbital;
            printf("[O] Orbitals: %s\n",show_orbital?"ON":"OFF");
            break;

        case 'm': case 'M':
            show_mulliken=!show_mulliken;
            printf("[M] Mulliken: %s\n",show_mulliken?"ON":"OFF");
            break;

        case 'e': case 'E':
            show_energy_dec=!show_energy_dec;
            printf("[E] Energy: %s\n",show_energy_dec?"ON":"OFF");
            break;

        case 'f': case 'F':
            show_forces=!show_forces;
            printf("[F] Forces: %s\n",show_forces?"ON":"OFF");
            break;

        // ================= ORBITAL VOLUME MODES (0=off, 1=V, 2=H, 3=L) =================
        case 'v': case 'V':
            if (show_homo_lumo == 1) {
                show_homo_lumo = 0;
                printf("[V] Total Density: OFF\n");
            } else {
                show_homo_lumo = 1;
                printf("[V] Total Density: ON\n");
            }
            break;

        case 'h': case 'H':
            if (show_homo_lumo == 2) {
                show_homo_lumo = 0;
                printf("[H] HOMO only: OFF\n");
            } else {
                show_homo_lumo = 2;
                printf("[H] HOMO only: ON\n");
            }
            break;

        case 'l': case 'L':
            if (show_homo_lumo == 3) {
                show_homo_lumo = 0;
                printf("[L] LUMO only: OFF\n");
            } else {
                show_homo_lumo = 3;
                printf("[L] LUMO only: ON\n");
            }
            break;

        // ================= DIPOLE =================
        case 'y': case 'Y':
            show_dipole=!show_dipole;
            printf("[Y] Dipole: %s\n",show_dipole?"ON":"OFF");
            break;

        // ================= SIMULATION CONTROL =================
        case 'p': case 'P': case ' ':
            pthread_mutex_lock(&pause_mutex);
            md_paused = !md_paused;
            if (!md_paused)
                pthread_cond_signal(&pause_cond);
            pthread_mutex_unlock(&pause_mutex);
            printf("[P/SPACE] Simulation: %s\n",md_paused?"PAUSED":"RUNNING");
            break;

        case '\r': case '\n':
            pthread_mutex_lock(&pause_mutex);
            if (md_paused) {
                md_step_once = 1;
                pthread_cond_signal(&pause_cond);
                printf("[ENTER] Single step\n");
            }
            pthread_mutex_unlock(&pause_mutex);
            break;

        // ================= STATISTICS PANEL =================
        case 'i': case 'I':
            show_stats=!show_stats;
            printf("[I] Statistics: %s\n",show_stats?"ON":"OFF");
            break;
            
        case 'b': case 'B':
            white_background = !white_background;
            printf("[B] Background: %s\n", white_background?"LIGHT":"DARK");
            break;

        // ================= RESET SIMULATION =================
        case 'q': case 'Q': {
            pthread_mutex_lock(&pause_mutex);
            md_paused=1;
            pthread_mutex_unlock(&pause_mutex);

            srand(initial_seed);
            md_step=0;
            history_count=0;
            box_size=5.0f;
            memset(energy_history,0,sizeof(energy_history));

            printf("[Q] Reset with seed=%u\n",initial_seed);

            initializePositions();

            pthread_mutex_lock(&pause_mutex);
            md_paused=0;
            pthread_cond_signal(&pause_cond);
            pthread_mutex_unlock(&pause_mutex);

            printf("Reset complete. E_initial = %+.8f Ha\n", E_hf_cached);
            break;
        }

        // ================= EXIT =================
        case 27: exit(0); // ESC
    }

    glutPostRedisplay();
}


// OpenGL lighting
const GLfloat light_ambient[]  = {0.1f,0.1f,0.1f,1};
const GLfloat light_diffuse[]  = {1,1,1,1};
const GLfloat light_specular[] = {1,1,1,1};
const GLfloat light_position[] = {2,5,5,0};
const GLfloat mat_ambient[]    = {0.7f,0.7f,0.7f,1};
const GLfloat mat_diffuse[]    = {0.8f,0.8f,0.8f,1};
const GLfloat mat_specular[]   = {1,1,1,1};
const GLfloat high_shininess[] = {100};

// ====================== MAIN ======================
int main(int argc, char **argv) {

    if (argc > 1) {
        if (strcmp(argv[1],"heh+")==0 || strcmp(argv[1],"HeH+")==0 ||
            strcmp(argv[1],"heh")==0  || strcmp(argv[1],"HEH+")==0) {
            molecule_type = 1; N_heh_pairs = 1;
            if (argc > 2) {
                int np = atoi(argv[2]);
                if (np < 1) { fprintf(stderr,"Error: number of HeH+ pairs must be >= 1.\n"); exit(EXIT_FAILURE); }
                if (np*2 > MAX_ATOM) { fprintf(stderr,"Error: %d pairs → %d atoms exceeds MAX_ATOM=%d.\n", np, np*2, MAX_ATOM); exit(EXIT_FAILURE); }
                N_heh_pairs = np;
            }
            NUM_ATOMS     = 2 * N_heh_pairs;
            NUM_ELECTRONS = 2 * N_heh_pairs;
            printf("Molecule  : %d x HeH+  (helium hydride cation)\n", N_heh_pairs);
            printf("Atoms     : %d  (%d He + %d H)\n", NUM_ATOMS, N_heh_pairs, N_heh_pairs);
            printf("Electrons : %d  (closed-shell RHF, occ=%d)\n", NUM_ELECTRONS, NUM_ELECTRONS/2);
            printf("Charge    : +%d\n", N_heh_pairs);
        } else if (strcmp(argv[1],"he")==0 || strcmp(argv[1],"He")==0 || strcmp(argv[1],"HE")==0) {
            molecule_type = 2;
            int n = 1;
            if (argc > 2) {
                n = atoi(argv[2]);
                if (n < 1) { fprintf(stderr,"Error: number of He atoms must be >= 1.\n"); exit(EXIT_FAILURE); }
                if (n > MAX_ATOM) { fprintf(stderr,"Error: %d atoms exceeds MAX_ATOM=%d.\n", n, MAX_ATOM); exit(EXIT_FAILURE); }
            }
            NUM_ATOMS = n; NUM_ELECTRONS = 2 * n;
            printf("Molecule  : %d He atom(s)  (neutral)\n", NUM_ATOMS);
            printf("Electrons : %d  (closed-shell RHF, occ=%d)\n", NUM_ELECTRONS, NUM_ELECTRONS/2);
        } else {
            int n;
            if (strcmp(argv[1],"h")==0 || strcmp(argv[1],"H")==0) {
                n = 2;
                if (argc > 2) n = atoi(argv[2]);
            } else {
                n = atoi(argv[1]);
            }
            if (n <= 0) {
                fprintf(stderr,"Error: unrecognized argument '%s'.\n", argv[1]);
                fprintf(stderr,"Usage:\n  ./aimd_hf [N]  |  ./aimd_hf h [N]  |  ./aimd_hf heh+ [N]  |  ./aimd_hf he [N]\n");
                exit(EXIT_FAILURE);
            }
            if (n % 2 != 0) { fprintf(stderr,"Error: RHF requires an even number of electrons. Got %d H atoms.\n", n); exit(EXIT_FAILURE); }
            if (n > MAX_ATOM) { fprintf(stderr,"Error: %d atoms exceeds MAX_ATOM=%d.\n", n, MAX_ATOM); exit(EXIT_FAILURE); }
            NUM_ATOMS = n; NUM_ELECTRONS = n;
            printf("Molecule  : %d hydrogen atom(s)\n", NUM_ATOMS);
            printf("Electrons : %d  (closed-shell RHF, occ=%d)\n", NUM_ELECTRONS, NUM_ELECTRONS/2);
        }
    } else {
        NUM_ATOMS = 2; NUM_ELECTRONS = 2;
        printf("Molecule  : H2  (default)\n");
    }

    // Assign per-atom STO-3G basis parameters
    if (molecule_type == 1) {
        for (int pair = 0; pair < N_heh_pairs; pair++) {
            int hi = pair * 2, li = pair * 2 + 1;
            g_Z[hi] = 2; for (int p = 0; p < NP; p++) g_a_exp[hi][p] = STO3G_He_a[p]; strcpy(g_atom_symbol[hi], "He");
            g_Z[li] = 1; for (int p = 0; p < NP; p++) g_a_exp[li][p] = STO3G_H_a[p];  strcpy(g_atom_symbol[li], "H");
        }
    } else if (molecule_type == 2) {
        for (int i = 0; i < NUM_ATOMS; i++) {
            g_Z[i] = 2; for (int p = 0; p < NP; p++) g_a_exp[i][p] = STO3G_He_a[p]; strcpy(g_atom_symbol[i], "He");
        }
    } else {
        for (int i = 0; i < NUM_ATOMS; i++) {
            g_Z[i] = 1; for (int p = 0; p < NP; p++) g_a_exp[i][p] = STO3G_H_a[p]; strcpy(g_atom_symbol[i], "H");
        }
    }

    // NEW: save seed for Q-reset, then seed RNG
    initial_seed = (unsigned int)time(NULL);
    srand(initial_seed);
    printf("Random seed: %u  (press Q to reset with same seed)\n", initial_seed);

    printf("================================================\n");
    printf("  AIMD — Hartree-Fock / STO-3G  (v2.0)\n");
    if      (molecule_type == 1) printf("  %d x HeH+  |  %d atoms  |  %d electrons  |  charge +%d\n", N_heh_pairs, NUM_ATOMS, NUM_ELECTRONS, N_heh_pairs);
    else if (molecule_type == 2) printf("  %d He atom(s)  |  %d electrons  |  neutral\n", NUM_ATOMS, NUM_ELECTRONS);
    else                         printf("  %d H atom(s)  |  %d electrons\n", NUM_ATOMS, NUM_ELECTRONS);
    printf("  Panels: O M E F V H L Y P I L Q B\n");
    printf("  dt = %.4f a.u. = %.6f fs\n", dt, dt*0.02418884f);
    printf("================================================\n");

    printf("Computing initial HF forces...\n");
    initializePositions();
    printf("Done. E_initial = %+.8f Ha  (%+.4f eV)\n\n", E_hf_cached, E_hf_cached*27.2114);
    printf("  Dipole |mu| = %.4f Debye\n", sqrt(g_dipole[0]*g_dipole[0]+g_dipole[1]*g_dipole[1]+g_dipole[2]*g_dipole[2])*2.5418);
    printf("  HOMO-LUMO gap = %.4f eV\n", g_homo_lumo_gap*27.2114);
    printf("  Virial 2<T>/|<V>| = %.5f\n\n", g_virial_ratio);

    if (molecule_type == 1 && N_heh_pairs == 1) printf("HeH+ STO-3G reference energy ≈ -2.8418 Ha\n\n");
    else if (molecule_type == 2 && NUM_ATOMS == 1) printf("He STO-3G reference energy ≈ -2.8077 Ha\n\n");

    pthread_t physics_thread;
    pthread_create(&physics_thread, NULL, physics_loop, NULL);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
    glutInitWindowSize(1150, 820);
    glutInitWindowPosition(50, 50);

    char title[256];
    if (molecule_type == 1)      sprintf(title,"Ab initio MD  -  HF/STO-3G  |  %dx HeH+  |  O M E F V H L Y I P Q B", N_heh_pairs);
    else if (molecule_type == 2) sprintf(title,"Ab initio MD  -  HF/STO-3G  |  %d He  |  O M E F V H L Y I P Q B", NUM_ATOMS);
    else                         sprintf(title,"Ab initio MD  -  HF/STO-3G  |  %dH  |  O M E F V H L Y I P Q B", NUM_ATOMS);
    glutCreateWindow(title);

    init_gl();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(key);
    glutIdleFunc(idle);

    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);
    glEnable(GL_LIGHT0);     glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL); glEnable(GL_LIGHTING);

    glLightfv(GL_LIGHT0, GL_AMBIENT,   light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR,  light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION,  light_position);

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    glutMainLoop();
    return 0;
}

