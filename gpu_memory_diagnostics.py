#!/usr/bin/env python3
"""
GPU Memory Diagnostics for Colab
Safely check what's using GPU memory before taking action
"""

import subprocess
import torch
import psutil
import os
from typing import List, Dict, Any


def get_gpu_processes() -> List[Dict[str, Any]]:
    """Get detailed information about processes using GPU memory."""
    try:
        # Run nvidia-smi to get process information
        result = subprocess.run([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid = int(parts[0])
                    name = parts[1]
                    memory_mb = int(parts[2])
                    
                    # Get additional process info
                    try:
                        proc = psutil.Process(pid)
                        processes.append({
                            'pid': pid,
                            'name': name,
                            'memory_mb': memory_mb,
                            'memory_gb': memory_mb / 1024,
                            'cmd': ' '.join(proc.cmdline()[:3]) if proc.cmdline() else 'N/A',
                            'status': proc.status(),
                            'create_time': proc.create_time(),
                            'is_current_process': pid == os.getpid()
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        processes.append({
                            'pid': pid,
                            'name': name,
                            'memory_mb': memory_mb,
                            'memory_gb': memory_mb / 1024,
                            'cmd': 'Access Denied',
                            'status': 'Unknown',
                            'create_time': None,
                            'is_current_process': False
                        })
        
        return processes
        
    except subprocess.CalledProcessError:
        print("❌ Could not run nvidia-smi")
        return []


def get_pytorch_memory_info() -> Dict[str, float]:
    """Get PyTorch memory usage information."""
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    try:
        memory_info = {
            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)
        }
        return memory_info
    except Exception as e:
        return {'error': str(e)}


def analyze_memory_usage():
    """Comprehensive analysis of GPU memory usage."""
    print("🔍 GPU Memory Diagnostic Report")
    print("=" * 50)
    
    # Get PyTorch memory info
    print("\n📊 PyTorch Memory Status:")
    pytorch_info = get_pytorch_memory_info()
    if 'error' in pytorch_info:
        print(f"❌ Error: {pytorch_info['error']}")
    else:
        print(f"  Total GPU Memory: {pytorch_info['total_gb']:.2f} GB")
        print(f"  PyTorch Allocated: {pytorch_info['allocated_gb']:.2f} GB")
        print(f"  PyTorch Reserved: {pytorch_info['reserved_gb']:.2f} GB")
        print(f"  Available: {pytorch_info['free_gb']:.2f} GB")
        print(f"  Max Ever Allocated: {pytorch_info['max_allocated_gb']:.2f} GB")
    
    # Get process information
    print(f"\n🏃 Processes Using GPU Memory:")
    processes = get_gpu_processes()
    
    if not processes:
        print("  No GPU processes found (or nvidia-smi unavailable)")
        return
    
    # Sort by memory usage (highest first)
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    total_used = sum(p['memory_mb'] for p in processes)
    
    print(f"  Total GPU Memory Used by Processes: {total_used / 1024:.2f} GB")
    print()
    
    for i, proc in enumerate(processes, 1):
        status_emoji = "🟢" if proc['is_current_process'] else "🔴"
        print(f"  {status_emoji} Process #{i}")
        print(f"    PID: {proc['pid']}")
        print(f"    Name: {proc['name']}")
        print(f"    Memory: {proc['memory_gb']:.2f} GB ({proc['memory_mb']} MB)")
        print(f"    Command: {proc['cmd']}")
        print(f"    Status: {proc['status']}")
        if proc['create_time']:
            import datetime
            create_time = datetime.datetime.fromtimestamp(proc['create_time'])
            print(f"    Started: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Provide recommendations
    print("💡 Recommendations:")
    
    # Find the biggest memory user that's not the current process
    other_processes = [p for p in processes if not p['is_current_process']]
    
    if other_processes:
        biggest = other_processes[0]
        print(f"  • Largest external process: PID {biggest['pid']} using {biggest['memory_gb']:.2f} GB")
        
        if biggest['memory_gb'] > 30:  # More than 30GB
            print(f"  • This process is using excessive memory!")
            print(f"  • Consider investigating: {biggest['cmd']}")
            
            # Check if it looks like another training process
            if any(keyword in biggest['cmd'].lower() for keyword in ['python', 'jupyter', 'colab']):
                print(f"  • This appears to be another Python/training process")
                print(f"  • You may want to stop it: kill {biggest['pid']}")
            else:
                print(f"  • This is a non-Python process, investigate before killing")
    
    current_proc = next((p for p in processes if p['is_current_process']), None)
    if current_proc:
        print(f"  • Current process using: {current_proc['memory_gb']:.2f} GB")
    
    # Memory recommendations
    available_memory = pytorch_info.get('free_gb', 0)
    if available_memory < 5:
        print(f"  • Available memory ({available_memory:.2f} GB) is very low")
        print(f"  • Consider reducing batch size or clearing memory")
    elif available_memory < 10:
        print(f"  • Available memory ({available_memory:.2f} GB) is limited")
        print(f"  • Use conservative batch sizes")
    else:
        print(f"  • Available memory ({available_memory:.2f} GB) looks good")


def clear_pytorch_cache():
    """Clear PyTorch cache safely."""
    if torch.cuda.is_available():
        print("🧹 Clearing PyTorch CUDA cache...")
        torch.cuda.empty_cache()
        print("✅ Cache cleared")
    else:
        print("❌ CUDA not available")


if __name__ == "__main__":
    analyze_memory_usage()
