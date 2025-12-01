/*
 * Jacobi Iterative Method - Sequential Version
 * Solves a system of linear equations Ax = b
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Function to initialize a diagonally dominant matrix (ensures convergence)
void initializeSystem(vector<vector<double>>& A, vector<double>& b, int n) {
    // Create a diagonally dominant matrix for convergence
    for (int i = 0; i < n; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = (double)(rand() % 10) / 10.0; // Small off-diagonal values
                rowSum += fabs(A[i][j]);
            }
        }
        // Make diagonal element dominant
        A[i][i] = rowSum + (double)(rand() % 10 + 1);
        b[i] = (double)(rand() % 100) / 10.0;
    }
}

// Sequential Jacobi Iterative Method
int jacobiSequential(const vector<vector<double>>& A, const vector<double>& b,
                     vector<double>& x, int n, double tolerance, int maxIterations) {
    vector<double> x_new(n, 0.0);
    int iterations = 0;
    
    for (int iter = 0; iter < maxIterations; iter++) {
        double maxDiff = 0.0;
        
        // Update each element
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;
            
            // Sum of A[i][j] * x[j] for j != i
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sigma += A[i][j] * x[j];
                }
            }
            
            // Jacobi formula: x_new[i] = (b[i] - sigma) / A[i][i]
            x_new[i] = (b[i] - sigma) / A[i][i];
            
            // Track maximum difference for convergence check
            double diff = fabs(x_new[i] - x[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        
        // Copy x_new to x
        for (int i = 0; i < n; i++) {
            x[i] = x_new[i];
        }
        
        iterations++;
        
        // Check for convergence
        if (maxDiff < tolerance) {
            break;
        }
    }
    
    return iterations;
}

// Function to verify solution by computing residual ||Ax - b||
double computeResidual(const vector<vector<double>>& A, const vector<double>& b,
                       const vector<double>& x, int n) {
    double residual = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        residual += (sum - b[i]) * (sum - b[i]);
    }
    return sqrt(residual);
}

int main() {
    // Problem sizes to test
    vector<int> sizes = {100, 500, 1000, 2000};
    double tolerance = 1e-6;
    int maxIterations = 10000;
    
    cout << "=============================================" << endl;
    cout << "  Jacobi Iterative Method - Sequential" << endl;
    cout << "=============================================" << endl;
    cout << fixed << setprecision(6);
    
    srand(42); // Fixed seed for reproducibility
    
    for (int n : sizes) {
        // Initialize system
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);
        vector<double> x(n, 0.0); // Initial guess
        
        initializeSystem(A, b, n);
        
        // Solve using Jacobi method
        auto start = high_resolution_clock::now();
        int iterations = jacobiSequential(A, b, x, n, tolerance, maxIterations);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(end - start);
        double timeMs = duration.count() / 1000.0;
        
        // Compute residual to verify solution
        double residual = computeResidual(A, b, x, n);
        
        cout << "\nMatrix size: " << n << " x " << n << endl;
        cout << "  Iterations: " << iterations << endl;
        cout << "  Time: " << timeMs << " ms" << endl;
        cout << "  Residual: " << scientific << residual << fixed << endl;
    }
    
    cout << "\n=============================================" << endl;
    
    return 0;
}
