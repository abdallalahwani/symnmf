// File: symnmf.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

// Utility function to free a matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Utility function to allocate a matrix
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) {
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)calloc(cols, sizeof(double));
        if (!matrix[i]) {
            return NULL;
        }
    }
    return matrix;
}

// Calculate the similarity matrix
double** sym(double** X, int n, int d) {
    double** A = allocate_matrix(n, n);
    if (!A) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                double dist2 = 0.0;
                for (int k = 0; k < d; k++) {
                    double diff = X[i][k] - X[j][k];
                    dist2 += diff * diff;
                }
                A[i][j] = exp(-dist2);
            }
        }
    }

    return A;
}

// Calculate the diagonal degree matrix
double** ddg(double** A, int n) {
    double** D = allocate_matrix(n, n);
    if (!D) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j];
        }
        D[i][i] = sum;
    }

    return D;
}

// Calculate the normalized similarity matrix
double** norm(double** A, int n) {
    double** D = ddg(A, n);
    if (!D) {
        return NULL;
    }

    double** W = allocate_matrix(n, n);
    if (!W) {
        free_matrix(D, n);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (D[i][i] > 0 && D[j][j] > 0) {
                W[i][j] = A[i][j] / (sqrt(D[i][i]) * sqrt(D[j][j]));
            }
        }
    }

    free_matrix(D, n);
    return W;
}

// Perform SymNMF algorithm
double** symnmf(double** W, double** H, int n, int k, int max_iter, double epsilon) {
    for (int iter = 0; iter < max_iter; iter++) {
        double max_change = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double WH_ij = 0.0;
                double HH_ij = 0.0;

                for (int p = 0; p < n; p++) {
                    WH_ij += W[i][p] * H[p][j];
                }
                for (int p = 0; p < k; p++) {
                    HH_ij += H[i][p] * H[p][j];
                }

                double H_new = H[i][j] * (0.5 + 0.5 * (WH_ij / HH_ij));
                max_change = fmax(max_change, fabs(H_new - H[i][j]));
                H[i][j] = H_new;
            }
        }

        if (max_change < epsilon) {
            break;
        }
    }

    return H;
}
