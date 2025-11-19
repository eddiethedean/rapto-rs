#!/usr/bin/env python3
"""Compare GitHub Actions ARM64 results with Docker results."""

import json
import glob
import sys
from pathlib import Path

def load_results(filepath):
    """Load JSON benchmark results."""
    with open(filepath) as f:
        return json.load(f)

def find_latest_results(directory, pattern):
    """Find the latest result file matching pattern."""
    files = glob.glob(str(Path(directory) / pattern))
    files = [f for f in files if not f.endswith('.metadata.json')]
    if not files:
        return None
    return sorted(files)[-1]

def compare_results(docker_file, github_file):
    """Compare Docker and GitHub Actions results."""
    docker_data = load_results(docker_file)
    github_data = load_results(github_file)
    
    print("=" * 80)
    print("Performance Comparison: Docker vs GitHub Actions ARM64")
    print("=" * 80)
    print()
    
    # Compare each test case
    for docker_case in docker_data.get('cases', []):
        shape = docker_case['shape']
        dtype = docker_case['dtype']
        
        # Find matching GitHub case
        github_case = None
        for gc in github_data.get('cases', []):
            if gc['shape'] == shape and gc['dtype'] == dtype:
                github_case = gc
                break
        
        if not github_case:
            print(f"⚠️  {shape[0]}² {dtype}: No matching GitHub Actions result")
            continue
        
        print(f"📊 {shape[0]}² {dtype}:")
        print()
        
        # Compare each operation
        for docker_op in docker_case.get('operations', []):
            op_name = docker_op['name']
            
            # Find matching GitHub operation
            github_op = None
            for go in github_case.get('operations', []):
                if go['name'] == op_name:
                    github_op = go
                    break
            
            if not github_op:
                print(f"  ⚠️  {op_name}: No matching GitHub Actions result")
                continue
            
            docker_speedup = docker_op.get('speedup', 0)
            docker_raptors = docker_op.get('raptors_mean_s', 0) * 1000
            docker_numpy = docker_op.get('numpy_mean_s', 0) * 1000
            
            github_speedup = github_op.get('speedup', 0)
            github_raptors = github_op.get('raptors_mean_s', 0) * 1000
            github_numpy = github_op.get('numpy_mean_s', 0) * 1000
            
            improvement = (docker_raptors - github_raptors) / docker_raptors * 100 if docker_raptors > 0 else 0
            speedup_ratio = github_speedup / docker_speedup if docker_speedup > 0 else 0
            
            print(f"  {op_name}:")
            print(f"    Docker:      {docker_speedup:.3f}x (Raptors: {docker_raptors:.3f}ms, NumPy: {docker_numpy:.3f}ms)")
            print(f"    GitHub ARM64: {github_speedup:.3f}x (Raptors: {github_raptors:.3f}ms, NumPy: {github_numpy:.3f}ms)")
            
            if improvement > 0:
                print(f"    🚀 Improvement: {improvement:.1f}% faster ({speedup_ratio:.2f}x speedup ratio)")
            else:
                print(f"    ⚠️  Regression: {abs(improvement):.1f}% slower")
            
            # Check if GitHub is faster than NumPy
            if github_speedup >= 1.0:
                print(f"    ✅ GitHub Actions is {github_speedup:.2f}x faster than NumPy!")
            elif github_speedup >= 0.8:
                print(f"    ⚠️  GitHub Actions is close to NumPy ({github_speedup:.2f}x)")
            else:
                print(f"    ❌ GitHub Actions is still slower than NumPy ({github_speedup:.2f}x)")
            
            print()
        
        print()

def main():
    """Main function."""
    if len(sys.argv) > 1:
        github_file = sys.argv[1]
    else:
        # Try to find latest GitHub Actions results
        github_file = find_latest_results(
            'benchmarks/results/github-actions-arm64',
            '*.json'
        )
        if not github_file:
            print("❌ No GitHub Actions results found.")
            print("   Download artifacts from GitHub Actions and extract to benchmarks/results/github-actions-arm64/")
            sys.exit(1)
    
    # Find latest Docker results
    docker_file = find_latest_results(
        'benchmarks/linux_investigation',
        'baseline_auto_*.json'
    )
    
    if not docker_file:
        print("❌ No Docker results found.")
        print("   Run: ./scripts/linux_baseline_measurement.sh in Docker")
        sys.exit(1)
    
    print(f"📁 Docker results: {docker_file}")
    print(f"📁 GitHub Actions results: {github_file}")
    print()
    
    compare_results(docker_file, github_file)

if __name__ == '__main__':
    main()

