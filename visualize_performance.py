#!/usr/bin/env python3
"""
Jacobi Method Performance Visualization
Parses output from jacobi_parallel and creates performance charts
"""

import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def compile_and_run_parallel():
    """Compile and run the parallel Jacobi program"""
    # Compile
    compile_cmd = [
        "clang++", "-Xpreprocessor", "-fopenmp",
        "-I/opt/homebrew/opt/libomp/include",
        "-L/opt/homebrew/opt/libomp/lib", "-lomp",
        "-O2", "-std=c++17",
        "jacobi_parallel.cpp", "-o", "jacobi_parallel"
    ]
    
    print("Compiling jacobi_parallel.cpp...")
    subprocess.run(compile_cmd, check=True)
    
    print("Running jacobi_parallel...")
    result = subprocess.run(["./jacobi_parallel"], capture_output=True, text=True)
    return result.stdout

def parse_output(output):
    """Parse the program output to extract performance data"""
    data = {
        'sizes': [],
        'sequential_times': [],
        'parallel_results': {}  # {threads: {size: time}}
    }
    
    current_size = None
    seq_time = None
    
    lines = output.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Match matrix size
        size_match = re.search(r'Matrix size:\s*(\d+)\s*x\s*\d+', line)
        if size_match:
            current_size = int(size_match.group(1))
            if current_size not in data['sizes']:
                data['sizes'].append(current_size)
        
        # Match sequential time
        if 'Sequential:' in line:
            # Look for time in next few lines
            for j in range(i, min(i+5, len(lines))):
                time_match = re.search(r'Time:\s*([\d.]+)\s*ms', lines[j])
                if time_match:
                    seq_time = float(time_match.group(1))
                    data['sequential_times'].append(seq_time)
                    break
        
        # Match parallel results table
        thread_match = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%', line)
        if thread_match and current_size is not None:
            threads = int(thread_match.group(1))
            time_ms = float(thread_match.group(2))
            
            if threads not in data['parallel_results']:
                data['parallel_results'][threads] = {}
            data['parallel_results'][threads][current_size] = time_ms
        
        i += 1
    
    return data

def create_visualizations(data):
    """Create performance visualization charts"""
    sizes = data['sizes']
    seq_times = data['sequential_times']
    parallel_results = data['parallel_results']
    
    thread_counts = sorted(parallel_results.keys())
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Jacobi Iterative Method - OpenMP Performance Analysis', fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(thread_counts)))
    
    # 1. Execution Time Comparison (Bar Chart)
    ax1 = axes[0, 0]
    x = np.arange(len(sizes))
    width = 0.15
    
    # Sequential bars
    bars = ax1.bar(x - width * (len(thread_counts)) / 2, seq_times, width, 
                   label='Sequential', color='gray', edgecolor='black')
    
    # Parallel bars for each thread count
    for idx, threads in enumerate(thread_counts):
        times = [parallel_results[threads].get(size, 0) for size in sizes]
        offset = width * (idx - len(thread_counts)/2 + 1)
        ax1.bar(x + offset, times, width, label=f'{threads} Thread(s)', 
                color=colors[idx], edgecolor='black')
    
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup vs Threads (Line Chart)
    ax2 = axes[0, 1]
    
    for idx, size in enumerate(sizes):
        speedups = []
        for threads in thread_counts:
            par_time = parallel_results[threads].get(size, 1)
            speedup = seq_times[idx] / par_time if par_time > 0 else 0
            speedups.append(speedup)
        
        ax2.plot(thread_counts, speedups, 'o-', label=f'{size}x{size}', 
                 linewidth=2, markersize=8)
    
    # Ideal speedup line
    ax2.plot(thread_counts, thread_counts, 'k--', label='Ideal', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Number of Threads')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(thread_counts)
    
    # 3. Efficiency vs Threads (Line Chart)
    ax3 = axes[1, 0]
    
    for idx, size in enumerate(sizes):
        efficiencies = []
        for threads in thread_counts:
            par_time = parallel_results[threads].get(size, 1)
            speedup = seq_times[idx] / par_time if par_time > 0 else 0
            efficiency = (speedup / threads) * 100
            efficiencies.append(efficiency)
        
        ax3.plot(thread_counts, efficiencies, 'o-', label=f'{size}x{size}', 
                 linewidth=2, markersize=8)
    
    # 100% efficiency line
    ax3.axhline(y=100, color='k', linestyle='--', label='Ideal (100%)', linewidth=1.5, alpha=0.7)
    
    ax3.set_xlabel('Number of Threads')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Parallel Efficiency vs Number of Threads')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(thread_counts)
    ax3.set_ylim(0, 120)
    
    # 4. Speedup Heatmap
    ax4 = axes[1, 1]
    
    speedup_matrix = []
    for size_idx, size in enumerate(sizes):
        row = []
        for threads in thread_counts:
            par_time = parallel_results[threads].get(size, 1)
            speedup = seq_times[size_idx] / par_time if par_time > 0 else 0
            row.append(speedup)
        speedup_matrix.append(row)
    
    speedup_array = np.array(speedup_matrix)
    im = ax4.imshow(speedup_array, cmap='YlGn', aspect='auto')
    
    ax4.set_xticks(np.arange(len(thread_counts)))
    ax4.set_yticks(np.arange(len(sizes)))
    ax4.set_xticklabels(thread_counts)
    ax4.set_yticklabels([f'{s}x{s}' for s in sizes])
    ax4.set_xlabel('Number of Threads')
    ax4.set_ylabel('Matrix Size')
    ax4.set_title('Speedup Heatmap')
    
    # Add text annotations
    for i in range(len(sizes)):
        for j in range(len(thread_counts)):
            text = ax4.text(j, i, f'{speedup_array[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Speedup')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'jacobi_performance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    
    plt.show()

def print_summary_table(data):
    """Print a summary table of results"""
    sizes = data['sizes']
    seq_times = data['sequential_times']
    parallel_results = data['parallel_results']
    thread_counts = sorted(parallel_results.keys())
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*80)
    
    # Header
    header = f"{'Size':<12}{'Sequential':<15}"
    for t in thread_counts:
        header += f"{t} Thread(s)".center(15)
    print(header)
    print("-"*80)
    
    for idx, size in enumerate(sizes):
        row = f"{size}x{size:<6}{seq_times[idx]:<15.2f}"
        for threads in thread_counts:
            par_time = parallel_results[threads].get(size, 0)
            speedup = seq_times[idx] / par_time if par_time > 0 else 0
            row += f"{par_time:.2f} ({speedup:.2f}x)".center(15)
        print(row)
    
    print("="*80)

def main():
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("="*60)
    print("  Jacobi Method Performance Visualization Tool")
    print("="*60)
    
    try:
        # Compile and run the parallel program
        output = compile_and_run_parallel()
        
        # Parse the output
        data = parse_output(output)
        
        if not data['sizes']:
            print("Error: Could not parse program output")
            print("Raw output:")
            print(output)
            return
        
        # Print summary table
        print_summary_table(data)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(data)
        
    except subprocess.CalledProcessError as e:
        print(f"Error compiling/running program: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure jacobi_parallel.cpp exists in the current directory")

if __name__ == "__main__":
    main()
