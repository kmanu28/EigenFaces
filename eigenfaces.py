# -*- coding: utf-8 -*-
"""
================================================================================
  FACE RECOGNITION USING EIGENFACES AND PCA-BASED SUBSPACE PROJECTION
  UE24MA241B — Linear Algebra and Its Applications
  PES University, Dept. of CSE  |  Orange-2 Mini Project
================================================================================

  Dataset : AT&T Olivetti Face Database
  Access  : sklearn.datasets.fetch_olivetti_faces()
  Size    : 400 grayscale images · 40 subjects · 10 images each · 64×64 px

  Linear Algebra Pipeline:
    01  Matrix Representation         — faces as column vectors in A (400×4096)
    02  Mean-Centering & RREF         — center data, find effective rank
    03  Face Space Analysis           — rank, nullity, column space
    04  Independent Facial Patterns   — basis extraction via pivots
    05  Gram–Schmidt Orthogonalization
    06  Orthogonal Projection         — project faces into eigenface space
    07  Least-Squares Recognition     — identify unknown faces via x̂=(AᵀA)⁻¹Aᵀb
    08  Eigenfaces (Covariance)       — eigendecompose C = AᵀA
    09  Diagonalization & PCA         — C = PDP⁻¹, keep top-k, scree plot

================================================================================
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# ── Matplotlib style ──────────────────────────────────────────────────────────
# Fix Unicode output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

matplotlib.rcParams.update({
    'figure.facecolor': '#1a1612',
    'axes.facecolor':   '#222018',
    'axes.edgecolor':   '#4a4038',
    'text.color':       '#f5f0e8',
    'axes.labelcolor':  '#f5f0e8',
    'xtick.color':      '#7a7060',
    'ytick.color':      '#7a7060',
    'axes.titlecolor':  '#f5f0e8',
    'grid.color':       '#2e2a24',
    'grid.linewidth':   0.7,
    'font.family':      'DejaVu Sans',
})

ACCENT  = '#b5451b'
BLUE    = '#4a90d9'
GREEN   = '#5aab6e'
MUTED   = '#7a7060'
CREAM   = '#f5f0e8'
IMG_DIM = 64      # each face is 64×64 pixels

# ─────────────────────────────────────────────────────────────────────────────
# STEP 01 — Matrix Representation
# ─────────────────────────────────────────────────────────────────────────────
def step01_load_and_build_matrix():
    """
    Load the AT&T Olivetti Face Database and build the data matrix A.
      - Each 64×64 image is flattened into a 4096-dimensional vector.
      - 400 such vectors are stacked as rows  →  A  (400 × 4096).
      - The matrix represents the entire face database.
    LA concept: Matrices, linear transformations, systems of equations.
    """
    print("\n" + "="*70)
    print("STEP 01 — Matrix Representation: Faces as Vectors")
    print("="*70)

    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X    = data.images          # (400, 64, 64) — pixel value range [0, 1]
    y    = data.target          # (400,)         — labels 0–39

    # Flatten each 64×64 image into a 4096-dim row vector
    A = X.reshape(400, -1).astype(np.float64)   # (400, 4096)

    print(f"  Loaded  : {X.shape[0]} face images from AT&T Olivetti dataset")
    print(f"  Subjects: {len(np.unique(y))}  |  Images per subject: 10")
    print(f"  Image   : {IMG_DIM}×{IMG_DIM} = {IMG_DIM**2} pixels each")
    print(f"  Matrix A: shape = {A.shape}  (n_samples × n_features)")
    print(f"  A[i, :] is the {A.shape[1]}-dimensional vector for face i")

    return A, y, X

# ─────────────────────────────────────────────────────────────────────────────
# STEP 02 — Mean-Centering & RREF
# ─────────────────────────────────────────────────────────────────────────────
def step02_mean_center_and_rref(A, sample_size=40):
    """
    Subtract the mean face from every row vector to center the data.
    Apply (partial) RREF on a sample to assess rank / independence.
    LA concept: Gaussian elimination, RREF, rank.
    """
    print("\n" + "="*70)
    print("STEP 02 — Mean-Centering & RREF")
    print("="*70)

    mean_face = A.mean(axis=0)          # μ — the average face (4096-dim)
    A_centered = A - mean_face          # subtract mean from every row

    print(f"  Mean face computed  : shape = {mean_face.shape}")
    print(f"  A_centered          : shape = {A_centered.shape}")
    print(f"  Max deviation from 0: {np.abs(A_centered.mean(axis=0)).max():.2e} (≈ 0 ✓)")

    # RREF on a small sample block to illustrate rank
    sample = A_centered[:sample_size, :sample_size].copy()
    rref_sample, pivot_cols = _rref(sample)
    sample_rank = len(pivot_cols)

    print(f"\n  RREF on {sample_size}×{sample_size} sub-block of A_centered:")
    print(f"    Pivot columns found : {pivot_cols}")
    print(f"    Rank of sub-block   : {sample_rank}  "
          f"(max possible = {sample_size})")
    print(f"    → Even within a small block, some rows are linearly dependent")

    return A_centered, mean_face

def _rref(M):
    """Return (rref_M, list_of_pivot_col_indices) for matrix M."""
    A = M.astype(float).copy()
    rows, cols = A.shape
    pivot_cols = []
    row = 0
    for col in range(cols):
        # Find pivot
        max_row = np.argmax(np.abs(A[row:, col])) + row
        if abs(A[max_row, col]) < 1e-9:
            continue
        A[[row, max_row]] = A[[max_row, row]]
        A[row] = A[row] / A[row, col]
        for r in range(rows):
            if r != row:
                A[r] -= A[r, col] * A[row]
        pivot_cols.append(col)
        row += 1
        if row == rows:
            break
    return A, pivot_cols

# ─────────────────────────────────────────────────────────────────────────────
# STEP 03 — Face Space Analysis (Rank & Nullity)
# ─────────────────────────────────────────────────────────────────────────────
def step03_face_space_analysis(A, A_centered):
    """
    Compute rank and nullity of A.
    LA concept: Vector spaces, subspaces, rank-nullity theorem.
    The column space is 'face space' — all valid face images live here.
    """
    print("\n" + "="*70)
    print("STEP 03 — Face Space Analysis (Rank & Nullity)")
    print("="*70)

    n_samples, n_features = A.shape             # 400, 4096

    # Numerical rank via SVD (stable & fast)
    _, s, _ = np.linalg.svd(A_centered, full_matrices=False)
    tol  = s[0] * max(A_centered.shape) * np.finfo(float).eps * 10
    rank = int(np.sum(s > tol))
    nullity = n_samples - rank    # rank-nullity on the row space

    print(f"  Data matrix A       : {n_samples} rows × {n_features} cols")
    print(f"  Rank of A_centered  : {rank}")
    print(f"  Nullity (rows)      : {nullity}  (= {n_samples} − {rank})")
    print(f"  Face space dim      : {rank} out of {n_features} possible dims")
    print(f"  Pixel space dim     : {n_features}")
    print(f"\n  → Faces occupy only {rank/n_features*100:.1f}% of the full pixel space")
    print(f"  → {nullity} face vectors ARE linearly dependent combinations of others")
    print(f"  Rank-Nullity theorem: {rank} + {nullity} = {rank + nullity} = n_samples ✓")

    return rank

# ─────────────────────────────────────────────────────────────────────────────
# STEP 04 — Independent Facial Patterns (Basis Extraction)
# ─────────────────────────────────────────────────────────────────────────────
def step04_independent_basis(A_centered, max_basis=50):
    """
    Extract a linearly independent basis from the face vectors using
    pivotal column selection on AᵀA.  Returns the top-max_basis
    independent face directions (column indices).
    LA concept: Linear independence, basis selection.
    """
    print("\n" + "="*70)
    print("STEP 04 — Independent Facial Patterns (Basis Extraction)")
    print("="*70)

    # Work with the compact covariance for efficiency: (n_samples × n_samples)
    # Gram matrix G = A_centered @ A_centered.T  — inner products of face pairs
    G = A_centered @ A_centered.T    # (400, 400)

    # Partial RREF on G to find independent row-indices (up to max_basis)
    _, pivots = _rref(G[:max_basis, :max_basis])
    independent_indices = list(pivots)

    print(f"  Gramian G = A·Aᵀ    : shape = {G.shape}")
    print(f"  Requested basis size: {max_basis}")
    print(f"  Independent faces   : {len(independent_indices)} selected")
    print(f"  Pivot indices (first 10): {independent_indices[:10]}")
    print(f"  → {400 - len(independent_indices)} face vectors removed as redundant")

    # The independent basis — rows of A_centered at these indices
    independent_basis = A_centered[independent_indices, :]   # (k, 4096)
    print(f"  Basis matrix shape  : {independent_basis.shape}  (k × 4096)")

    return independent_basis, independent_indices

# ─────────────────────────────────────────────────────────────────────────────
# STEP 05 — Gram–Schmidt Orthogonalization
# ─────────────────────────────────────────────────────────────────────────────
def step05_gram_schmidt(basis, max_vectors=30):
    """
    Apply the Gram–Schmidt process to orthogonalize the chosen basis.
    LA concept: Orthogonal vectors, Gram–Schmidt, orthogonal bases.
    Note: We use a numerically stable modified Gram–Schmidt.
    """
    print("\n" + "="*70)
    print("STEP 05 — Gram–Schmidt Orthogonalization")
    print("="*70)

    # Work on a manageable subset to show the concept clearly
    V = basis[:max_vectors, :].copy().astype(np.float64)

    Q = []
    for i, v in enumerate(V):
        u = v.copy()
        for q in Q:
            u -= np.dot(u, q) * q          # subtract projection onto each prior q
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            Q.append(u / norm)             # normalize to unit length

    Q_matrix = np.array(Q)                 # (k, 4096) — orthonormal set

    # Verify orthogonality: Q @ Q.T should be ≈ I
    gram = Q_matrix @ Q_matrix.T
    off_diag = gram - np.eye(len(Q))
    ortho_error = np.abs(off_diag).max()

    print(f"  Input basis vectors : {len(V)}")
    print(f"  Output Q vectors    : {len(Q)}")
    print(f"  Q shape             : {Q_matrix.shape}  (each row is a unit vector)")
    print(f"  Orthogonality check : max |⟨qᵢ,qⱼ⟩| for i≠j = {ortho_error:.2e}  (≈0 ✓)")
    print(f"  Normality check     : max |‖qᵢ‖−1| = "
          f"{np.abs(np.linalg.norm(Q_matrix, axis=1) - 1).max():.2e}  (≈0 ✓)")

    return Q_matrix

# ─────────────────────────────────────────────────────────────────────────────
# STEP 06 — Orthogonal Projection onto Eigenface Space
# ─────────────────────────────────────────────────────────────────────────────
def step06_project_faces(A_centered, eigenfaces, mean_face):
    """
    Project every face (training and test) onto the eigenface subspace.
    Each face → short weight vector  (face fingerprint in eigenface space).
    LA concept: Orthogonal projections, projection onto subspaces.
    """
    print("\n" + "="*70)
    print("STEP 06 — Project Faces onto Eigenface Space")
    print("="*70)

    # eigenfaces: (k, 4096)  each row is a unit eigenface
    k = eigenfaces.shape[0]
    weights = A_centered @ eigenfaces.T   # (400, k) — projection coefficients

    # Verify: reconstructed face ≈ original face (within the subspace)
    A_reconstructed = weights @ eigenfaces + mean_face
    recon_error = np.mean((A_reconstructed - (A_centered + mean_face))**2)

    print(f"  Eigenface basis size: k = {k}")
    print(f"  Weight matrix W     : shape = {weights.shape}  (400 faces × {k} weights)")
    print(f"  Compression ratio   : {4096/k:.0f}× (4096 → {k} dimensions)")
    print(f"  Mean recon. error   : {recon_error:.6f}  "
          f"(zero if k = full rank)")
    print(f"  Each face is now a {k}-dim 'fingerprint' vector in eigenspace")

    return weights

# ─────────────────────────────────────────────────────────────────────────────
# STEP 07 — Least-Squares Face Recognition
# ─────────────────────────────────────────────────────────────────────────────
def step07_least_squares_recognition(weights, labels, eigenfaces,
                                     A_centered, mean_face, n_components):
    """
    Identify unknown faces using the least-squares formula x̂ = (AᵀA)⁻¹Aᵀb.
    LA concept: Least squares solution, matrix transpose & inverse.
    """
    print("\n" + "="*70)
    print("STEP 07 — Identification via Least Squares")
    print("="*70)

    # Keep k=n_components weights for recognition
    W = weights[:, :n_components]         # (400, k)

    # Train/test split: 8 train + 2 test per subject
    X_train, X_test, y_train, y_test = train_test_split(
        W, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ── Least-squares recognition ──────────────────────────────────────────
    # For a test face vector b (k-dim weights), solve:
    #   min ‖M · x − b‖²  →  x̂ = (MᵀM)⁻¹Mᵀb
    # where M is the matrix with training faces as COLUMNS.
    
    M = X_train.T                                   # (k, n_train)
    MTM = M.T @ M                                   # (n_train, n_train)
    MTM_inv = np.linalg.pinv(MTM)                   # (n_train, n_train)
    MTM_inv_MT = MTM_inv @ M.T                      # (n_train, k)

    predictions = []
    for b in X_test:
        x_hat = MTM_inv_MT @ b                      # LS solution, shape: (n_train,)
        pred_idx = np.argmax(x_hat)                 # which training face?
        predictions.append(y_train[pred_idx])

    predictions = np.array(predictions)
    accuracy    = np.mean(predictions == y_test) * 100

    print(f"  Training faces   : {len(y_train)}")
    print(f"  Test faces       : {len(y_test)}")
    print(f"  Eigenspace dim k : {n_components}")
    print(f"\n  x̂ = (MᵀM)⁻¹Mᵀb  applied to each test face")
    print(f"  MᵀM shape        : {MTM.shape}  |  x̂ shape: ({len(y_train)},)")
    print(f"\n  Recognition Accuracy : {accuracy:.1f}%  ({np.sum(predictions==y_test)}"
          f"/{len(y_test)} correct)")

    return X_train, X_test, y_train, y_test, predictions, accuracy

# ─────────────────────────────────────────────────────────────────────────────
# STEP 08 — Eigenfaces via Covariance Matrix
# ─────────────────────────────────────────────────────────────────────────────
def step08_eigenfaces_covariance(A_centered):
    """
    Compute covariance C = AᵀA (using the compact dual trick for efficiency).
    Eigendecompose → eigenvalues λ and eigenvectors (Eigenfaces).
    LA concept: Eigenvalues, eigenvectors, covariance matrix.
    """
    print("\n" + "="*70)
    print("STEP 08 — Eigenfaces via Covariance Matrix")
    print("="*70)

    n, d = A_centered.shape     # 400, 4096

    # Dual trick: compute eigenvectors of A·Aᵀ (400×400) instead of AᵀA (4096×4096)
    L = A_centered @ A_centered.T     # (400, 400) — Gram/kernel matrix
    print(f"  Full covariance AᵀA : {d}×{d}  (too large to eigendecompose directly)")
    print(f"  Compact dual L=AAᵀ  : {n}×{n}  (eigendecomp here, then lift)")

    eigenvalues, eigvecs_L = np.linalg.eigh(L)   # eigh for symmetric matrix

    # Sort descending
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigvecs_L    = eigvecs_L[:, idx]

    # Lift eigenvectors of L to eigenvectors of AᵀA (the actual eigenfaces)
    # If L·v = λ·v  →  (AᵀA)·(Aᵀv) = λ·(Aᵀv)
    eigenfaces_raw  = A_centered.T @ eigvecs_L        # (4096, 400)
    norms           = np.linalg.norm(eigenfaces_raw, axis=0, keepdims=True)
    norms[norms < 1e-10] = 1
    eigenfaces_norm = (eigenfaces_raw / norms).T      # (400, 4096) — unit rows

    # Only keep positive eigenvalues
    pos_mask        = eigenvalues > 1e-6
    eigenvalues     = eigenvalues[pos_mask]
    eigenfaces_norm = eigenfaces_norm[pos_mask]

    print(f"  Positive eigenvalues: {len(eigenvalues)}")
    print(f"  Largest eigenvalue  : {eigenvalues[0]:.2f}")
    print(f"  Smallest kept       : {eigenvalues[-1]:.4f}")
    print(f"  Eigenfaces shape    : {eigenfaces_norm.shape}  "
          f"(each row = one eigenface, unit norm)")

    # Verification: AᵀA · v ≈ λ · v  (for top eigenface)
    ef0   = eigenfaces_norm[0]
    AtA_v = (A_centered.T @ (A_centered @ ef0))     # equivalent to AᵀA · v
    lam_v = eigenvalues[0] * ef0
    ev_err = np.linalg.norm(AtA_v - lam_v) / np.linalg.norm(lam_v)
    print(f"  Eigenvector check   : ‖AᵀAv − λv‖/‖λv‖ = {ev_err:.2e}  (≈0 ✓)")

    return eigenfaces_norm, eigenvalues

# ─────────────────────────────────────────────────────────────────────────────
# STEP 09 — Diagonalization & Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────
def step09_diagonalization(eigenvalues):
    """
    Express the covariance as C = PDP⁻¹.
    Choose k so that cumulative variance ≥ 95%.
    LA concept: Diagonalization, symmetric matrix decomposition.
    """
    print("\n" + "="*70)
    print("STEP 09 — Diagonalization & Dimensionality Reduction")
    print("="*70)

    # D is the diagonal matrix of eigenvalues
    # P is the matrix of eigenvectors (columns = eigenfaces)
    # For a symmetric matrix: P⁻¹ = Pᵀ  →  C = PDP ᵀ

    cumvar   = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100
    k50      = int(np.searchsorted(cumvar, 95.0)) + 1
    k_reco   = min(k50, len(eigenvalues))

    print(f"  Total eigenvalues   : {len(eigenvalues)}")
    print(f"  C = P D Pᵀ  (C is symmetric ⟹ P⁻¹ = Pᵀ)")
    print(f"  D is diagonal with entries: λ₁ ≥ λ₂ ≥ … ≥ λ_n")
    print(f"\n  Variance captured:")
    for k in [1, 5, 10, 20, 50, 100, k_reco]:
        k = min(k, len(eigenvalues))
        print(f"    top {k:4d} components → {cumvar[k-1]:.1f}% variance")
    print(f"\n  → k = {k_reco} components capture ≥ 95% of variance")
    print(f"  → Dimensionality: {len(eigenvalues)} → {k_reco}  "
          f"(compression: {len(eigenvalues)/k_reco:.0f}×)")

    return k_reco, cumvar

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_mean_face(mean_face):
    """Display the mean (average) face."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(mean_face.reshape(IMG_DIM, IMG_DIM), cmap='bone')
    ax.set_title("Mean Face  (μ)", fontsize=12, color=CREAM, pad=10)
    ax.axis('off')
    fig.suptitle("Step 02 — Mean Face", fontsize=10, color=MUTED, y=0.02)
    plt.tight_layout()


def plot_eigenface_gallery(eigenfaces, n_show=20):
    """Fig 1: Top-k eigenfaces as ghostly face images."""
    n_cols = 5
    n_rows = n_show // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 6))
    fig.suptitle(
        f"Fig 1 — Eigenface Gallery  (top {n_show} eigenfaces)",
        fontsize=14, color=CREAM, fontweight='bold', y=1.01
    )
    for i, ax in enumerate(axes.flat):
        ef = eigenfaces[i].reshape(IMG_DIM, IMG_DIM)
        # Normalize for display
        ef_disp = (ef - ef.min()) / (ef.max() - ef.min() + 1e-9)
        ax.imshow(ef_disp, cmap='bone')
        ax.set_title(f"EF {i+1}", fontsize=8, color=MUTED, pad=3)
        ax.axis('off')
    plt.tight_layout()


def plot_scree(eigenvalues, cumvar, k_reco):
    """Fig 2: Scree plot — eigenvalue magnitudes + cumulative variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Fig 2 — Scree Plot  (Step 09: Diagonalization)",
        fontsize=14, color=CREAM, fontweight='bold'
    )

    n_show = min(100, len(eigenvalues))
    x = np.arange(1, n_show + 1)

    # --- Eigenvalue magnitudes ---
    ax1.bar(x, eigenvalues[:n_show], color=ACCENT, alpha=0.8, width=0.9)
    ax1.set_xlabel("Principal Component #", fontsize=10)
    ax1.set_ylabel("Eigenvalue (λ)", fontsize=10)
    ax1.set_title("Eigenvalue Spectrum", fontsize=11, color=CREAM)
    ax1.axvline(k_reco, color=GREEN, lw=1.5, ls='--', label=f'k={k_reco}')
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y')

    # --- Cumulative variance explained ---
    ax2.plot(x, cumvar[:n_show], color=BLUE, lw=2.5)
    ax2.fill_between(x, cumvar[:n_show], alpha=0.15, color=BLUE)
    ax2.axhline(95, color=GREEN, lw=1.2, ls='--', label='95% variance')
    ax2.axvline(k_reco, color=GREEN, lw=1.5, ls='--', label=f'k={k_reco}')
    ax2.scatter([k_reco], [cumvar[k_reco-1]], color=GREEN, s=80, zorder=5)
    ax2.set_xlabel("Number of Components (k)", fontsize=10)
    ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=10)
    ax2.set_title("Cumulative Variance Captured", fontsize=11, color=CREAM)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True)

    plt.tight_layout()


def plot_reconstruction(A, mean_face, eigenfaces, eigenvalues, subject_idx=0):
    """Fig 3: Face reconstruction at different values of k."""
    k_values = [5, 20, 50, 100, len(eigenfaces)]
    k_values = sorted(set(min(k, len(eigenfaces)) for k in k_values))

    face = A[subject_idx].copy()
    face_centered = face - mean_face

    n = len(k_values) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    fig.suptitle(
        "Fig 3 — Reconstruction at k Eigenfaces  (Step 09: PCA compression)",
        fontsize=12, color=CREAM, fontweight='bold'
    )

    # Original
    axes[0].imshow(face.reshape(IMG_DIM, IMG_DIM), cmap='bone')
    axes[0].set_title("Original", fontsize=9, color=CREAM, pad=5)
    axes[0].axis('off')

    for ax, k in zip(axes[1:], k_values):
        ef_k    = eigenfaces[:k]
        weights = face_centered @ ef_k.T           # project
        recon   = weights @ ef_k + mean_face       # lift back + add mean
        recon   = np.clip(recon, 0, 1)
        err     = np.mean((recon - face)**2)
        ax.imshow(recon.reshape(IMG_DIM, IMG_DIM), cmap='bone')
        ax.set_title(f"k = {k}\nMSE={err:.4f}", fontsize=8, color=MUTED, pad=5)
        ax.axis('off')

    plt.tight_layout()


def plot_recognition_result(A, y, X_test_w, y_test, y_pred,
                             A_centered_full, mean_face, eigenfaces_k,
                             n_components, accuracy):
    """Fig 4: Test face vs best match, with overall accuracy bar."""

    # Pick a correct prediction to display
    correct_mask = y_pred == y_test
    idx_to_show  = np.where(correct_mask)[0][0] if correct_mask.any() else 0

    # Reconstruct test face from weights (for display)
    w_test  = X_test_w[idx_to_show, :n_components]
    reconstructed_test = w_test @ eigenfaces_k[:n_components] + mean_face
    reconstructed_test = np.clip(reconstructed_test, 0, 1)

    # Find closest training face in original image space (for visual match)
    true_label  = y_test[idx_to_show]
    pred_label  = y_pred[idx_to_show]
    match_idxs  = np.where(y == pred_label)[0]
    match_face  = A[match_idxs[0]]

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(
        f"Fig 4 — Recognition Result  |  Overall Accuracy: {accuracy:.1f}%  "
        f"(k = {n_components})",
        fontsize=13, color=CREAM, fontweight='bold'
    )

    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.imshow(reconstructed_test.reshape(IMG_DIM, IMG_DIM), cmap='bone')
    ax1.set_title(f"Test Face\n(Subject {true_label})", fontsize=10, color=CREAM)
    ax1.axis('off')

    ax2.imshow(match_face.reshape(IMG_DIM, IMG_DIM), cmap='bone')
    color = GREEN if pred_label == true_label else ACCENT
    ax2.set_title(f"Best Match\n(Predicted: Subject {pred_label})",
                  fontsize=10, color=color)
    ax2.axis('off')

    # Accuracy bar breakdown — per subject
    per_subject = []
    for s in range(40):
        mask = y_test == s
        if mask.sum() > 0:
            per_subject.append(np.mean(y_pred[mask] == y_test[mask]) * 100)
        else:
            per_subject.append(0)
    subj_ids    = list(range(len(per_subject)))
    bar_colors  = [GREEN if v == 100 else (ACCENT if v == 0 else BLUE)
                   for v in per_subject]
    ax3.bar(subj_ids, per_subject, color=bar_colors, width=0.8)
    ax3.set_xlabel("Subject ID", fontsize=9)
    ax3.set_ylabel("Accuracy (%)", fontsize=9)
    ax3.set_title("Per-Subject Recognition Accuracy", fontsize=10, color=CREAM)
    ax3.set_ylim(0, 110)
    ax3.axhline(accuracy, color=CREAM, lw=1, ls='--', alpha=0.5,
                label=f'Overall {accuracy:.0f}%')
    ax3.legend(fontsize=8)
    ax3.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 74)
    print("  FACE RECOGNITION USING EIGENFACES  --  PES University UE24MA241B")
    print("  Linear Algebra and Its Applications  |  Orange-2 Mini Project")
    print("=" * 74)
    print()

    # -- Step 01: Load & build matrix --
    A, y, X_images = step01_load_and_build_matrix()

    # -- Step 02: Mean-center & RREF --
    A_centered, mean_face = step02_mean_center_and_rref(A)

    # -- Step 03: Rank & Nullity --
    rank = step03_face_space_analysis(A, A_centered)

    # -- Step 04: Independent basis --
    ind_basis, ind_indices = step04_independent_basis(A_centered, max_basis=50)

    # -- Step 05: Gram–Schmidt --
    Q_matrix = step05_gram_schmidt(ind_basis, max_vectors=30)

    # -- Step 08: Eigenfaces from covariance (do before step 06 so we have
    #             the proper eigenfaces from PCA to use in projection) --
    eigenfaces, eigenvalues = step08_eigenfaces_covariance(A_centered)

    # -- Step 09: Diagonalization & k selection --
    k_reco, cumvar = step09_diagonalization(eigenvalues)

    # -- Step 06: Project all faces using PCA-derived eigenfaces --
    weights = step06_project_faces(A_centered, eigenfaces[:k_reco], mean_face)

    # -- Step 07: Least-squares recognition --
    X_train, X_test, y_train, y_test, predictions, accuracy = \
        step07_least_squares_recognition(
            weights, y, eigenfaces[:k_reco], A_centered, mean_face, k_reco
        )

    # ── Visualizations ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS …")
    print("="*70)

    plot_mean_face(mean_face)
    plot_eigenface_gallery(eigenfaces, n_show=20)
    plot_scree(eigenvalues, cumvar, k_reco)
    plot_reconstruction(A, mean_face, eigenfaces, eigenvalues, subject_idx=0)
    plot_recognition_result(
        A, y, weights, y_test, predictions,
        A_centered, mean_face, eigenfaces, k_reco, accuracy
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  RESULTS SUMMARY                                                         ║""")
    print(f"║  Dataset   : 400 faces · 40 subjects · 64×64 px                        ║")
    print(f"║  Matrix A  : 400 × 4096                                                ║")
    print(f"║  Rank      : {rank:<5}  (out of 4096 possible dims)                    ║")
    print(f"║  Eigenfaces: {len(eigenvalues):<5}  positive eigenvalues                          ║")
    print(f"║  k chosen  : {k_reco:<5}  (captures ≥ 95% variance)                   ║")
    print(f"║  Accuracy  : {accuracy:<6.1f}% recognition on held-out test faces            ║")
    print("""╚══════════════════════════════════════════════════════════════════════════╝

  Figures:
    Mean Face           — Step 02 output (mean-centering anchor)
    Fig 1: Eigenface Gallery   — Top-20 eigenfaces (ghostly face images)
    Fig 2: Scree Plot          — Step 09 PCA variance explained
    Fig 3: Reconstruction      — k=5 / 20 / 50 / 100 / full
    Fig 4: Recognition Result  — Test face → predicted identity + per-subject accuracy
    """)

    plt.show()


if __name__ == "__main__":
    main()
