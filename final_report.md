# Jacobi Iterative Method - Performance Analysis Report

## 1. Introduction

This report presents a comprehensive analysis of the Jacobi iterative method for solving systems of linear equations ($$Ax = b$$) using both sequential and parallel (OpenMP) implementations. The objective is to evaluate the performance gains achieved through parallelization and to understand the factors affecting scalability.

The Jacobi method is an iterative algorithm particularly well-suited for parallelization due to its inherent data independence—each element of the solution vector can be computed independently within a single iteration.

---

## 2. Background

### 2.1 Problem Statement

Given a system of linear equations represented as $$Ax = b$$, where:
- $$A$$ is an $$n \times n$$ coefficient matrix
- $$x$$ is the unknown solution vector
- $$b$$ is the right-hand side vector

The goal is to find the solution vector $x$ that satisfies the equation.

### 2.2 Jacobi Method Theory

The Jacobi iterative method computes the solution by iteratively updating each element using the formula:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)$$

Where:
- $$x_i^{(k+1)}$$ is the new value of the i-th element at iteration $$k+1$$
- $$a_{ij}$$ are the matrix coefficients
- $$b_i$$ is the i-th element of the right-hand side vector
- The iteration continues until convergence (when the maximum change is below a tolerance threshold)

---

## 3. Methodology

### 3.1 Algorithm Design

The logic of the Jacobi method proceeds as follows:

1. Initialize the solution vector $$x$$ (usually with zeros)
2. Create a separate storage for the new solution, $$x_{new}$$
3. For each iteration $$k$$:
     - For each row $$i$$ from 0 to $$n-1$$:
         - Calculate the sum $$\sigma = \sum_{j \neq i} a_{ij} x_j$$
         - Compute $$x_{new}[i] = (b_i - \sigma) / a_{ii}$$
     - Check for convergence (if $$\max|x_{new}[i] - x[i]| < tolerance$$)
     - Update $$x = x_{new}$$

**Pseudocode:**
```
INPUT: Matrix A, Vector b, Tolerance TOL, Max_Iter
OUTPUT: Solution Vector x

Initialize x = [0, 0, ... 0]
Initialize x_new = [0, 0, ... 0]

For k from 1 to Max_Iter:
    maxDiff = 0
    For i from 0 to N-1:
        sigma = 0
        For j from 0 to N-1:
            If j != i:
                sigma = sigma + A[i][j] * x[j]
        
        x_new[i] = (b[i] - sigma) / A[i][i]
        diff = abs(x_new[i] - x[i])
        If diff > maxDiff:
            maxDiff = diff
    
    x = x_new  // Update solution vector
    
    If maxDiff < TOL:
        Break Loop
```

### 3.2 System Design

The system consists of three main modules:

1. **Data Initialization (`initializeSystem`)**: Generates a random strictly diagonally dominant matrix $$A$$ and vector $$b$$ to ensure convergence.

2. **Solver Engine**: Contains both `jacobiSequential()` and `jacobiParallel()` functions.

3. **Performance Metrics**: Uses `chrono::high_resolution_clock` and `omp_get_wtime()` to measure execution time and validates correctness via residual computation.

### 3.3 Parallelization Strategy

We utilized **Domain Decomposition (Data Parallelism)** to parallelize the outer loop.

**OpenMP Directive:**
```cpp
#pragma omp parallel for reduction(max:maxDiff) schedule(static)
for (int i = 0; i < n; i++) {
    // Compute x_new[i] independently
}
```

**Data Scoping:**
- `shared(A, b, x, x_new)`: The matrix and solution vectors must be read by all threads
- `private(sigma)`: The temporary sum accumulator must be private to each thread to prevent race conditions
- `reduction(max:maxDiff)`: Combines the maximum difference from all threads for convergence checking

**Synchronization:**
- An implicit barrier exists at the end of the parallel for loop
- This ensures all $$x_{new}$$ values are computed before swapping them into $$x$$ for the next iteration

---

## 4. Implementation Details

### 4.1 Hardware & Software Specifications

The performance analysis was conducted on the following system:

| Component | Specification |
|-----------|---------------|
| Processor | Apple Silicon (ARM64) |
| Operating System | macOS |
| Compiler | Clang with OpenMP support |
| OpenMP Library | libomp (Homebrew) |
| C++ Standard | C++17 |

### 4.2 Key Functions

| Function | Description |
|----------|-------------|
| `initializeSystem(int n)` | Creates an $$n \times n$$ matrix. To guarantee convergence, we ensure diagonal dominance by setting $$a_{ii} > \sum_{j \neq i} |a_{ij}|$$ |
| `jacobiSequential(...)` | Implements the standard algorithm using a single thread |
| `jacobiParallel(...)` | Implements the OpenMP version with configurable thread count |
| `computeResidual(...)` | Multiplies $$Ax$$ and compares it with $$b$$ to verify solution correctness |

### 4.3 Test Configuration

| Parameter | Value |
|-----------|-------|
| Tolerance | $$1 \times 10^{-6}$$ |
| Maximum Iterations | 10,000 |
| Matrix Type | Diagonally dominant (ensures convergence) |
| Random Seed | 42 (for reproducibility) |
| Matrix Sizes | 100×100, 500×500, 1000×1000, 2000×2000 |
| Thread Counts | 1, 2, 4, 8 |

---

## 5. Results and Analysis

### 5.1 Performance Data

The following table summarizes the execution times for each configuration:

| Matrix Size | Sequential (ms) | 1 Thread (ms) | 2 Threads (ms) | 4 Threads (ms) | 8 Threads (ms) |
|-------------|-----------------|---------------|----------------|----------------|----------------|
| 100×100     | 1.19            | 1.41          | 2.21           | 2.96           | 8.10           |
| 500×500     | 97.49           | 90.55         | 50.54          | 35.51          | 45.49          |
| 1000×1000   | 627.48          | 718.14        | 366.30         | 187.25         | 196.59         |
| 2000×2000   | 4679.55         | 4680.20       | 2491.05        | 1306.83        | 1017.59        |

### 5.2 Speedup Analysis

Speedup is calculated as: $$S = \frac{T_{sequential}}{T_{parallel}}$$

| Matrix Size | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
|-------------|----------|-----------|-----------|-----------|
| 100×100     | 0.84×    | 0.54×     | 0.40×     | 0.15×     |
| 500×500     | 1.08×    | 1.93×     | 2.75×     | 2.14×     |
| 1000×1000   | 0.87×    | 1.71×     | 3.35×     | 3.19×     |
| 2000×2000   | 1.00×    | 1.88×     | 3.58×     | 4.60×     |

### 5.3 Efficiency Analysis

Efficiency is calculated as: $$E = \frac{S}{P} \times 100\%$$

Where $$S$$ is the speedup and $$P$$ is the number of threads.

| Matrix Size | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
|-------------|----------|-----------|-----------|-----------|
| 100×100     | 84%      | 27%       | 10%       | 2%        |
| 500×500     | 108%     | 96%       | 69%       | 27%       |
| 1000×1000   | 87%      | 86%       | 84%       | 40%       |
| 2000×2000   | 100%     | 94%       | 90%       | 57%       |

### 5.4 Critical Analysis

The results reveal distinct behaviors based on matrix size:

#### Small Matrices (N = 100)

The parallel version was actually **slower** than the serial version across all thread counts. This is a classic example of **parallel overhead**. The cost of creating threads, managing their private stacks, and the implicit synchronization barriers at the end of every iteration outweighs the computational benefit when the workload is small.

#### Medium Matrices (N = 500, 1000)

At these sizes, we begin to see meaningful speedup (up to 3.35× with 4 threads). The computational work grows as $$O(n^2)$$, which starts to dominate the parallelization overhead. However, 8 threads shows diminishing returns, suggesting memory bandwidth saturation.

#### Large Matrices (N = 2000)

We achieved the best speedup of **4.60×** with 8 threads. This confirms that the Jacobi method scales well for large datasets where the overhead-to-computation ratio is minimal.

### 5.5 Diminishing Returns at Higher Thread Counts

The observation that 8 threads sometimes performs worse than 4 threads can be explained by:

1. **Memory Bandwidth Saturation**: The Jacobi method is memory-bound; multiple threads compete for memory bandwidth
2. **Cache Contention**: More threads lead to more cache misses and false sharing
3. **NUMA Effects**: Memory access latency varies based on core-to-memory distance
4. **Synchronization Overhead**: More threads require more synchronization at barriers

### 5.6 Amdahl's Law Implications

The maximum theoretical speedup is limited by the serial portion of the code:

$$S_{max} = \frac{1}{s + \frac{1-s}{P}}$$

Where $$s$$ is the serial fraction and $$P$$ is the number of processors.

**Serial Components in the Implementation:**
- Convergence check (finding maximum difference via reduction)
- Loop control and iteration management
- Memory allocation and initialization

Even with a small serial fraction (e.g., 5%), maximum speedup with 8 threads is limited to approximately 5.9×.

---

## 6. Optimal Configuration Recommendations

Based on our analysis, the following configurations are recommended:

| Matrix Size | Recommended Threads | Reasoning |
|-------------|---------------------|-----------|
| < 200       | 1 (Sequential)      | Overhead exceeds benefit |
| 200 - 800   | 4                   | Best balance of speedup/efficiency |
| 800 - 1500  | 4                   | Good speedup with high efficiency |
| > 1500      | 8                   | Large enough to benefit from more threads |

**Suggested Adaptive Thread Selection:**
```cpp
int selectOptimalThreads(int matrixSize, int maxThreads) {
    if (matrixSize < 200) return 1;
    if (matrixSize < 800) return min(4, maxThreads);
    return maxThreads;
}
```

---

## 7. Conclusion

### 7.1 Key Findings

1. **Problem Size Matters**: Parallelization is only beneficial when the problem size is large enough to overcome threading overhead. For matrices smaller than 200×200, sequential execution is preferred.

2. **Sweet Spot Exists**: For most matrix sizes, 4 threads provides the best balance of speedup and efficiency.

3. **Memory Bound Nature**: The Jacobi method's performance is limited by memory bandwidth, not just CPU compute capacity.

4. **Scalability Achieved**: For the largest test case (2000×2000), we achieved a speedup of 4.60× with 8 threads, demonstrating effective parallelization.

### 7.2 Recommendations for Further Optimization

1. **Adaptive Threading**: Implement runtime checks to select thread count based on matrix size
2. **Chunk Size Tuning**: Experiment with different OpenMP schedule policies (static, dynamic, guided)
3. **Memory Optimization**: Consider cache-blocking techniques for very large matrices
4. **Hybrid Approaches**: For extremely large problems, consider combining OpenMP with MPI for distributed computing
5. **GPU Acceleration**: Porting the algorithm to CUDA could potentially offer speedups in the range of 50×-100× due to the massive number of available cores

---

## 8. References

1. Jacobi, C. G. J. (1845). "Ueber eine neue Auflösungsart der bei der Methode der kleinsten Quadrate vorkommenden lineären Gleichungen"
2. OpenMP Architecture Review Board. "OpenMP Application Programming Interface Version 5.0". https://www.openmp.org/specifications/
3. Amdahl, G. M. (1967). "Validity of the single processor approach to achieving large scale computing capabilities"
4. Saad, Y. (2003). Iterative methods for sparse linear systems (2nd ed.). Society for Industrial and Applied Mathematics (SIAM).
5. Karniadakis, G. E., & Kirby, R. M. (2003). Parallel scientific computing in C++ and MPI. Cambridge University Press.

---

*Report generated on December 13, 2025*  
*Test system: macOS with Apple Silicon (ARM64)*
