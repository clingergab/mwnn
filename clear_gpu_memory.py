#!/usr/bin/env python3
"""
GPU Memory Management Script for Colab
Run this FIRST to clear memory before training.
"""

import os
import subprocess
import torch
import gc

def clear_all_gpu_memory():
    """Aggressively clear all GPU memory"""
    print("🧹 Clearing all GPU memory...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    print("✅ GPU memory cleared")

def kill_all_gpu_processes():
    """Kill all GPU processes to free memory"""
    print("🔫 Killing all GPU processes...")
    
    try:
        # Get all GPU processes
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            killed_count = 0
            
            for line in lines:
                if line.strip():
                    try:
                        pid, memory = line.split(',')
                        pid = pid.strip()
                        memory_gb = float(memory.strip()) / 1024
                        
                        print(f"🔫 Killing process {pid} using {memory_gb:.1f} GB")
                        subprocess.run(['kill', '-9', pid], check=True)
                        killed_count += 1
                        
                    except Exception as e:
                        print(f"Failed to kill process {pid}: {e}")
            
            if killed_count > 0:
                print(f"✅ Killed {killed_count} GPU processes")
                # Wait a moment for processes to actually die
                import time
                time.sleep(2)
            else:
                print("ℹ️  No GPU processes to kill")
        else:
            print("ℹ️  No GPU processes found")
            
    except Exception as e:
        print(f"Could not check GPU processes: {e}")

def setup_pytorch_memory_optimization():
    """Configure PyTorch for optimal memory usage"""
    print("🚀 Setting up PyTorch memory optimization...")
    
    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution for better performance
    
    # Set PyTorch optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Enable memory-efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
    
    print("✅ PyTorch optimizations configured")

def show_gpu_status():
    """Show current GPU memory status"""
    print("\n📊 GPU Memory Status:")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            # Get total memory
            try:
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024**3
                free = total - allocated
                
                print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free ({total:.2f} GB total)")
            except Exception:
                print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("No CUDA GPUs available")

def main():
    """Main memory management function"""
    print("🚀 GPU Memory Management for MWNN Training")
    print("=" * 50)
    
    # Step 1: Kill processes
    kill_all_gpu_processes()
    
    # Step 2: Clear memory
    clear_all_gpu_memory()
    
    # Step 3: Setup optimizations
    setup_pytorch_memory_optimization()
    
    # Step 4: Show status
    show_gpu_status()
    
    print("\n✅ GPU memory management complete!")
    print("🎯 You can now run your training script")

if __name__ == "__main__":
    main()
