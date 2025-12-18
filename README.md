# Jacobi Iterative Method - Parallel Computing Project

This project implements and compares sequential and parallel versions of the Jacobi iterative method for solving systems of linear equations (Ax = b). The parallel implementation uses OpenMP for multi-threaded execution. 

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Building the Programs](#building-the-programs)
- [Running the Programs](#running-the-programs)
- [Visualizing Results](#visualizing-results)
- [File Descriptions](#file-descriptions)

## ⚠️ Platform Note

**This project is configured for macOS.** The VS Code task configuration (in `.vscode/tasks.json`) and the Python visualization script (`visualize_performance.py`) are optimized for macOS with Clang/LLVM and Homebrew-installed OpenMP. If you're using a different operating system (Linux or Windows), you may need to adjust:
- Compiler flags in the VS Code build tasks
- Include/library paths for OpenMP
- Compilation commands in the visualization script
- Binary execution paths (e.g., `.exe` extensions on Windows)

Refer to the platform-specific compilation instructions in the [Building the Programs](#building-the-programs) and [Running the Programs](#running-the-programs) sections for your system.

## 🎯 Overview

The Jacobi method is an iterative algorithm for solving systems of linear equations. This project demonstrates:
- Sequential implementation as a baseline
- Parallel implementation using OpenMP
- Performance analysis with varying matrix sizes and thread counts
- Data visualization for performance comparison

## 📁 Project Structure

```
jacobi/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── jacobi_sequential.cpp        # Sequential implementation
├── jacobi_parallel.cpp          # Parallel OpenMP implementation
├── visualize_performance.py     # Python script for visualization
├── final_report.md             # Detailed performance analysis report
├── jacobi_sequential           # Compiled sequential binary
├── jacobi_parallel             # Compiled parallel binary
└── jacobi_*.dSYM/              # Debug symbols (macOS)
```

## 🔧 Prerequisites

### Required Software
- **C++ Compiler**: 
  - **macOS**: Clang with C++17 support
  - **Linux**: GCC with C++17 support
  - **Windows**: Visual Studio 2019+ or MinGW-w64 with GCC
- **OpenMP**: Required for parallel execution
  - **macOS**: `brew install libomp`
  - **Linux**: Usually included with GCC
  - **Windows**: Included with Visual Studio or MinGW-w64
- **Python 3.x**: For visualization script
- **Python Libraries**:
  - matplotlib
  - numpy

### Setting Up Python Environment

#### Option 1: Using Virtual Environment (Recommended)

A virtual environment isolates project dependencies and prevents conflicts with other Python projects.

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# When done working, deactivate
deactivate
```

**Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# When done working, deactivate
deactivate
```

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies from requirements.txt
pip install -r requirements.txt

# When done working, deactivate
deactivate
```

#### Option 2: Global Installation

Install packages system-wide (simpler but not recommended for multiple projects):

**macOS/Linux:**
```bash
pip3 install matplotlib numpy
# Or use requirements.txt:
pip3 install -r requirements.txt
```

**Windows (Command Prompt or PowerShell):**
```cmd
pip install matplotlib numpy
# Or use requirements.txt:
pip install -r requirements.txt
```

**Note:** Once the virtual environment is activated, you'll see `(venv)` in your terminal prompt. All subsequent Python and pip commands will use the virtual environment.

### Installing OpenMP

**macOS:**
```bash
brew install libomp
```

**Linux:**
```bash
# Usually pre-installed with GCC
# If needed:
sudo apt-get install libomp-dev
```

**Windows:**
- **Visual Studio**: OpenMP is included (enable in project properties)
- **MinGW-w64**: OpenMP is included with GCC

## 🛠️ Building the Programs

### Option 1: Using VS Code Tasks
If you're using VS Code with the provided tasks configuration:

1. Open either `jacobi_sequential.cpp` or `jacobi_parallel.cpp`
2. Press `Cmd+Shift+B` (macOS) or `Ctrl+Shift+B` (Windows/Linux) (Build tasks have already been defined in .vscode/tasks.json)
3. Select the appropriate build task

### Option 2: Manual Compilation

#### Sequential Version

**macOS/Linux:**
```bash
clang++ -std=c++17 -O2 jacobi_sequential.cpp -o jacobi_sequential
# Or with GCC:
g++ -std=c++17 -O2 jacobi_sequential.cpp -o jacobi_sequential
```

**Windows (MinGW-w64):**
```cmd
g++ -std=c++17 -O2 jacobi_sequential.cpp -o jacobi_sequential.exe
```

**Windows (Visual Studio Developer Command Prompt):**
```cmd
cl /EHsc /std:c++17 /O2 jacobi_sequential.cpp
```

#### Parallel Version

**macOS:**
```bash
clang++ -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib \
  -lomp -std=c++17 -O2 \
  jacobi_parallel.cpp -o jacobi_parallel
```

**Linux:**
```bash
g++ -fopenmp -std=c++17 -O2 jacobi_parallel.cpp -o jacobi_parallel
```

**Windows (MinGW-w64):**
```cmd
g++ -fopenmp -std=c++17 -O2 jacobi_parallel.cpp -o jacobi_parallel.exe
```

**Windows (Visual Studio Developer Command Prompt):**
```cmd
cl /EHsc /std:c++17 /O2 /openmp jacobi_parallel.cpp
```

## 🚀 Running the Programs

### Sequential Implementation

**macOS/Linux:**
```bash
./jacobi_sequential
```

**Windows:**
```cmd
jacobi_sequential.exe
```

**What it does:**
- Tests multiple matrix sizes (500, 1000, 1500, 2000, 2500)
- Runs the sequential Jacobi algorithm
- Reports execution time and number of iterations
- Verifies solution accuracy

**Sample Output:**
```
===================================
   Sequential Jacobi Method
===================================

Testing with Matrix size: 500 x 500
Sequential: 0.2345 seconds, 42 iterations
Maximum error: 9.87654e-07
✓ Solution converged successfully
```

### Parallel Implementation

**macOS/Linux:**
```bash
./jacobi_parallel
```

**Windows:**
```cmd
jacobi_parallel.exe
```

**What it does:**
- Tests the same matrix sizes as sequential version
- Runs with multiple thread counts (1, 2, 4, 8)
- Compares sequential vs parallel performance
- Calculates speedup and efficiency metrics
- Verifies solution accuracy for all configurations

**Sample Output:**
```
===================================
   Parallel Jacobi Method (OpenMP)
===================================

Testing with Matrix size: 1000 x 1000

Sequential baseline: 0.8532 seconds, 45 iterations

Parallel with 4 threads: 0.2341 seconds, 45 iterations
Speedup: 3.64x | Efficiency: 91.0%
Maximum error: 9.12345e-07
✓ Solution converged successfully
```

## 📊 Visualizing Results

### Automated Visualization
The Python script compiles, runs the parallel program, and generates performance charts:

**macOS/Linux:**
```bash
python3 visualize_performance.py
```

**Windows:**
```cmd
python visualize_performance.py
```

**Note for Windows users:** You may need to modify `visualize_performance.py` to use the correct compilation commands for your compiler (MinGW or Visual Studio). See the troubleshooting section below.

**What it does:**
1. Compiles `jacobi_parallel.cpp` (if needed)
2. Executes the program and captures output
3. Parses performance metrics
4. Generates multiple charts:
   - Execution time comparison
   - Speedup analysis
   - Efficiency trends
   - Strong scaling analysis

**Generated Charts:**
- `jacobi_execution_times.png` - Time vs Matrix Size
- `jacobi_speedup.png` - Speedup vs Thread Count
- `jacobi_efficiency.png` - Parallel Efficiency
- `jacobi_strong_scaling.png` - Strong Scaling Analysis

### View Generated Charts
Charts are saved as PNG files in the project directory. Open them with any image viewer or include them in reports.

## 📄 File Descriptions

### Source Code Files

#### `jacobi_sequential.cpp`
**Purpose:** Sequential baseline implementation of the Jacobi iterative method.

**Key Features:**
- Implements the classic Jacobi algorithm without parallelization
- Creates diagonally dominant matrices (ensures convergence)
- Measures execution time for performance comparison
- Tests multiple matrix sizes (500 to 2500)
- Validates solution accuracy

**Algorithm:**
```
For each iteration:
  For each equation i:
    x_new[i] = (b[i] - sum(A[i][j] * x[j] for j ≠ i)) / A[i][i]
  Check convergence
```

#### `jacobi_parallel.cpp`
**Purpose:** Parallel implementation using OpenMP for multi-threaded execution.

**Key Features:**
- Implements both sequential and parallel Jacobi methods
- Uses OpenMP `#pragma omp parallel for` for parallelization
- Tests with 1, 2, 4, and 8 threads
- Calculates speedup and efficiency metrics
- Compares against sequential baseline
- Includes detailed performance reporting

**Parallelization Strategy:**
- Outer loop iterations are parallelized using `#pragma omp parallel for`
- Each thread processes independent rows of the matrix
- Reduction operations for convergence checking
- Private variables to avoid race conditions

#### `visualize_performance.py`
**Purpose:** Automated compilation, execution, and visualization of performance data.

**Key Features:**
- Compiles the parallel program automatically
- Executes and captures program output
- Parses performance metrics using regex
- Generates professional performance charts
- Saves charts as PNG files
- Handles multiple matrix sizes and thread counts

**Functions:**
- `compile_and_run_parallel()`: Builds and executes the program
- `parse_output()`: Extracts performance data from output
- `plot_*()`: Various plotting functions for different metrics
- `main()`: Orchestrates the entire workflow

### Documentation Files

#### `requirements.txt`
**Purpose:** Lists Python package dependencies for the project.

**Contents:**
**Typically includes:**
- Compiled binaries (`jacobi_sequential`, `jacobi_parallel`, `*.exe`)
- Debug symbols (`*.dSYM/`, `*.pdb`, `*.ilk`)
- Python virtual environment (`venv/`, `env/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Generated charts (`*.png`)
- IDE configuration files (`.vs/`, `.vscode/`)
- Object files (`*.o`, `*.obj`)

This file makes it easy to install all required Python packages in one command and ensures version compatibility.

#### `.gitignore`
**Purpose:** Specifies files that Git should ignore.
**Purpose:** Specifies files that Git should ignore.

**Typically includes:**
- Compiled binaries (`jacobi_sequential`, `jacobi_parallel`, `*.exe`)
- Debug symbols (`*.dSYM/`, `*.pdb`, `*.ilk`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Generated charts (`*.png`)
- IDE configuration files (`.vs/`, `.vscode/`)
- Object files (`*.o`, `*.obj`)

### Binary Files

#### `jacobi_sequential` and `jacobi_parallel` (macOS/Linux) or `*.exe` (Windows)
**Purpose:** Compiled executable programs (generated after building).

These are the compiled binaries that you run to execute the Jacobi algorithms. They are platform-specific and should be recompiled on different systems:
- **macOS/Linux**: No file extension
- **Windows**: `.exe` extension

#### Debug Symbol Files
**Purpose:** Debug information files (generated during compilation with debug flags).

- **macOS**: `*.dSYM/` directories
- **Windows (Visual Studio)**: `*.pdb` files
- **Linux**: Debug symbols embedded or in separate files

These contain debugging information that allows debuggers to map machine code back to source code. They can be safely deleted if not debugging.

## 🧪 Understanding the Output

### Performance Metrics

- **Execution Time**: Wall-clock time for algorithm completion
- **Iterations**: Number of iterations until convergence
- **Speedup**: Sequential time / Parallel time (higher is better)
- **Efficiency**: Speedup / Number of threads × 100% (ideal is 100%)
- **Maximum Error**: Largest difference between computed and expected solution

### What Good Performance Looks Like

- **Speedup**: Should approach the number of threads (e.g., 4x speedup with 4 threads)
- **Efficiency**: Should be > 70% for good parallel performance
- **Convergence**: Algorithm should converge for all configurations
- **Accuracy**: Maximum error should be < 1e-6

## 🔍 Troubleshooting

### Compilation Issues

**Problem:** OpenMP not found

**macOS:**
```bash
brew install libomp
```

**Linux:**
```bash
sudo apt-get install libomp-dev
```

**Windows:**
- **MinGW-w64**: Ensure you have the full MinGW-w64 distribution (not just MinGW)
  - Download from: https://www.mingw-w64.org/ or use MSYS2
  - With MSYS2: `pacman -S mingw-w64-x86_64-gcc`
- **Visual Studio**: Enable OpenMP in project properties:
  - Project → Properties → C/C++ → Language → Open MP Support → Yes (/openmp)

**Problem:** C++17 not supported

**macOS:**
```bash
brew install llvm
```

**Linux:**
```bash
sudo apt-get install g++-9
```

**Windows:**
- Update Visual Studio to 2019 or later
- Or install the latest MinGW-w64

### Windows-Specific Issues

**Problem:** `visualize_performance.py` fails to compile on Windows

**Solution:** Modify the `compile_and_run_parallel()` function in `visualize_performance.py`:

**For MinGW-w64:**
```python
def compile_and_run_parallel():
    # Compile
    compile_cmd = [
        "g++", "-fopenmp",
        "-O2", "-std=c++17",
        "jacobi_parallel.cpp", "-o", "jacobi_parallel.exe"
    ]
    
    print("Compiling jacobi_parallel.cpp...")
    subprocess.run(compile_cmd, check=True, shell=True)
    
    print("Running jacobi_parallel...")
    result = subprocess.run(["jacobi_parallel.exe"], capture_output=True, text=True, shell=True)
    return result.stdout
```

**For Visual Studio:**
```python
def compile_and_run_parallel():
    # Compile
    compile_cmd = [
        "cl", "/EHsc", "/std:c++17", "/O2", "/openmp",
        "jacobi_parallel.cpp"
    ]
    
    print("Compiling jacobi_parallel.cpp...")
    subprocess.run(compile_cmd, check=True, shell=True)
    
    print("Running jacobi_parallel...")
    result = subprocess.run(["jacobi_parallel.exe"], capture_output=True, text=True, shell=True)
    return result.stdout
```

**Problem:** Command not found errors on Windows

**Solution:**
- Ensure your compiler is in the system PATH
- For Visual Studio, use the "Developer Command Prompt for VS" instead of regular Command Prompt
- For MinGW, add the bin directory to PATH: `C:\msys64\mingw64\bin` (or your installation path)

**Problem:** Permission denied when running executables

**Solution:**
- Windows may block downloaded executables. Right-click → Properties → Unblock
- Run Command Prompt as Administrator if needed
- Check antivirus settings (some may flag compiled programs)

### Runtime Issues

**Problem:** Program doesn't converge
- The matrix might not be diagonally dominant
- Increase `maxIterations` in the source code
- Check the tolerance value

**Problem:** Visualization script fails
```bash
# Install missing Python packages
pip3 install matplotlib numpy

# Or use conda
conda install matplotlib numpy
```

## 📚 Additional Resources

- [OpenMP Documentation](https://www.openmp.org/specifications/)
- [Jacobi Method Theory](https://en.wikipedia.org/wiki/Jacobi_method)
- See `final_report.md` for detailed performance analysis

## 👨‍💻 Author

Created by Leong Shi Wei, Heng Chee Xun, Teo Jing Zhu as part of a Parallel and Distributed Computing course project.

## 📝 License

This project is created for educational purposes.
