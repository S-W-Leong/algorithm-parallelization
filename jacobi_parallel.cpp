/*
 * Jacobi Iterative Method - OpenMP Parallel Version
 * Solves a system of linear equations Ax = b
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>

using namespace std;

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

// Sequential Jacobi Iterative Method (for comparison)
int jacobiSequential(const vector<vector<double>>& A, const vector<double>& b,
                     vector<double>& x, int n, double tolerance, int maxIterations) {
    vector<double> x_new(n, 0.0);
    int iterations = 0;
    
    for (int iter = 0; iter < maxIterations; iter++) {
        double maxDiff = 0.0;
        
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sigma += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / A[i][i];
            double diff = fabs(x_new[i] - x[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        
        for (int i = 0; i < n; i++) {
            x[i] = x_new[i];
        }
        
        iterations++;
        if (maxDiff < tolerance) {
            break;
        }
    }
    
    return iterations;
}

// Parallel Jacobi Iterative Method using OpenMP
int jacobiParallel(const vector<vector<double>>& A, const vector<double>& b,
                   vector<double>& x, int n, double tolerance, int maxIterations,
                   int numThreads) {
    vector<double> x_new(n, 0.0);
    int iterations = 0;
    
    omp_set_num_threads(numThreads);
    
    for (int iter = 0; iter < maxIterations; iter++) {
        double maxDiff = 0.0;
        
        // Manual reduction: each thread tracks its own max
        vector<double> threadMaxDiff(numThreads, 0.0);
        
        // Parallel region for computing new values
        #pragma omp parallel for schedule(static)
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
            
            // Track maximum difference for convergence check (manual reduction)
            double diff = fabs(x_new[i] - x[i]);
            int threadId = omp_get_thread_num();
            if (diff > threadMaxDiff[threadId]) {
                threadMaxDiff[threadId] = diff;
            }
        }
        
        // Combine partial results from all threads
        for (int t = 0; t < numThreads; t++) {
            if (threadMaxDiff[t] > maxDiff) {
                maxDiff = threadMaxDiff[t];
            }
        }
        
        // Copy x_new to x (can also be parallelized)
        #pragma omp parallel for schedule(static)
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
    vector<int> threadCounts = {1, 2, 4, 8};
    double tolerance = 1e-6;
    int maxIterations = 10000;
    
    int maxThreads = omp_get_max_threads();
    
    cout << "=============================================" << endl;
    cout << "  Jacobi Iterative Method - OpenMP Parallel" << endl;
    cout << "=============================================" << endl;
    cout << "Maximum available threads: " << maxThreads << endl;
    cout << fixed << setprecision(6);
    
    // Store results for analysis
    vector<vector<double>> seqTimes(sizes.size());
    vector<vector<vector<double>>> parTimes(sizes.size());
    
    for (size_t s = 0; s < sizes.size(); s++) {
        int n = sizes[s];
        parTimes[s].resize(threadCounts.size());
        
        cout << "\n=====================================================" << endl;
        cout << "Matrix size: " << n << " x " << n << endl;
        cout << "=====================================================" << endl;
        
        // Initialize system (same for all tests)
        srand(42); // Fixed seed for reproducibility
        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);
        initializeSystem(A, b, n);
        
        // Sequential execution
        {
            vector<double> x(n, 0.0);

            double start = omp_get_wtime();
            int iterations = jacobiSequential(A, b, x, n, tolerance, maxIterations);
            double end = omp_get_wtime();

            double timeMs = (end - start) * 1000.0;
            seqTimes[s].push_back(timeMs);
            
            double residual = computeResidual(A, b, x, n);
            
            cout << "\nSequential:" << endl;
            cout << "  Iterations: " << iterations << endl;
            cout << "  Time: " << timeMs << " ms" << endl;
            cout << "  Residual: " << scientific << residual << fixed << endl;
        }
        
        // Parallel execution with different thread counts
        cout << "\nParallel (OpenMP):" << endl;
        cout << "-----------------------------------------------------" << endl;
        cout << setw(10) << "Threads" << setw(15) << "Time (ms)" 
             << setw(12) << "Speedup" << setw(15) << "Efficiency" << endl;
        cout << "-----------------------------------------------------" << endl;
        
        for (size_t t = 0; t < threadCounts.size(); t++) {
            int numThreads = threadCounts[t];
            
            // Skip if more threads than available
            if (numThreads > maxThreads) {
                continue;
            }
            
            vector<double> x(n, 0.0);

            double start = omp_get_wtime();
            int iterations = jacobiParallel(A, b, x, n, tolerance, maxIterations, numThreads);
            double end = omp_get_wtime();

            double timeMs = (end - start) * 1000.0;
            parTimes[s][t].push_back(timeMs);
            
            double residual = computeResidual(A, b, x, n);
            
            // Calculate speedup and efficiency
            double speedup = seqTimes[s][0] / timeMs;
            double efficiency = (speedup / numThreads) * 100.0;
            
            cout << setw(10) << numThreads 
                 << setw(15) << timeMs
                 << setw(12) << setprecision(2) << speedup
                 << setw(14) << efficiency << "%" << setprecision(6) << endl;
        }
    }
    
    return 0;
}
