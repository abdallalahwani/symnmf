
#ifndef SYMNMF_H
#define SYMNMF_H

// Allocate memory for a matrix
double** allocate_matrix(int rows, int cols);

// Free allocated memory for a matrix
void free_matrix(double** matrix, int rows);

// Compute the similarity matrix A
// X: input data (n x d), n = number of points, d = dimensionality of each point
double** sym(double** X, int n, int d);

// Compute the diagonal degree matrix D
// A: similarity matrix (n x n), n = number of points
double** ddg(double** A, int n);

// Compute the normalized similarity matrix W
// A: similarity matrix (n x n), n = number of points
double** norm(double** A, int n);

// Perform the Symmetric Non-negative Matrix Factorization (SymNMF)
// W: normalized similarity matrix (n x n)
// H: initial guess for the factor matrix (n x k)
// n: number of points
// k: number of clusters
// max_iter: maximum number of iterations
// epsilon: convergence threshold
double** symnmf(double** W, double** H, int n, int k, int max_iter, double epsilon);

#endif // SYMNMF_H
