# Face Recognition using Eigenfaces

A comprehensive Linear Algebra mini project implementing a Face Recognition System from scratch using Eigenfaces and PCA-Based Subspace Projection. Built for **UE24MA241B — Linear Algebra and Its Applications**.

## Overview

Given a database of known faces, this project identifies an unknown face using purely linear algebraic mechanisms — avoiding black-box machine learning algorithms. The approach relies on identifying the principal linear dimensions ("Eigenfaces") of all combined faces, projecting all faces into this optimized lower-dimensional subspace, and running a nearest-neighbor classification via a least squares formulation.

**Dataset**: AT&T Olivetti Face Database
* 400 total face images, separated into 40 varying individual subjects.
* Each image is $64 \times 64$ pixels (a 4096-dimensional basis).

## Pipeline Steps
This repository is architected sequentially mirroring the mathematical workflow:
1. **Matrix Representation**: Flattening all $64 \times 64$ images into a contiguous 400 $\times$ 4096 representation matrix $A$. 
2. **Mean-Centering & RREF**: Calculating dataset rank limits and extracting centering bases.
3. **Face Space vs Pixel Space Deductions**: Resolving dimensional nullity vs relevant subject dimensions.
4. **Independent Patterns Extrusion**: Finding exact pivotal base cases over covariance matrices matching independent combinations.
5. **Modified Gram-Schmidt Orthogonalization**: Securing normalized bounds over sub-variants bounding the specific subspace.
6. **Eigendecomposition (Eigenface Generation)**: Factoring covariance bounds via $A^TA$ limits extraction.
7. **PCA Diagonalization Limits**: Isolating dimensionality to $123$ components yielding $\approx 95\%$ preserved facial variance. 
8. **Subspace Orientational Projections**: Mapping global dimensions optimally into the new space.
9. **Least-Squares Recreations**: Mapping identification subsets correctly identifying new unknowns against training identities reaching $\sim 85\%$ correctness.

## Setup & Running

**Prerequisites**: Python 3.10+
1. Install requirements
```bash
pip install -r requirements.txt
```
2. Run the pipeline and output generations
```bash
python eigenfaces.py
```

This will automatically output computational statuses to the terminal and compile 4 visual Matplotlib charts documenting statistical bounds and matching results graphically.
