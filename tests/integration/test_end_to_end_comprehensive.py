#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite
Run complete MWNN pipeline tests on both MNIST and ImageNet datasets.
"""

import sys
from pathlib import Path
import subprocess
import time
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))


def run_test_file(test_file, test_name):
    """Run a test file and capture results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"📁 Test file: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=test_file.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        
        print(f"\n📊 {test_name} Results:")
        print(f"   ⏱️  Elapsed time: {elapsed_time:.2f} seconds")
        print(f"   🎯 Exit code: {result.returncode}")
        print(f"   ✅ Status: {'PASSED' if success else 'FAILED'}")
        
        return success, elapsed_time
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"❌ {test_name} timed out after {elapsed_time:.2f} seconds")
        return False, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {test_name} failed with exception: {e}")
        return False, elapsed_time


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"   ✅ TorchVision {torchvision.__version__}")
        
        import pandas
        print(f"   ✅ Pandas {pandas.__version__}")
        
        import numpy
        print(f"   ✅ NumPy {numpy.__version__}")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"   🚀 CUDA available: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   🚀 MPS (Metal) available")
        else:
            print(f"   💻 Using CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def check_data_availability():
    """Check if required datasets are available."""
    print("\n📂 Checking dataset availability...")
    
    base_path = Path(__file__).parent.parent.parent / 'data'
    
    # Check MNIST
    mnist_path = base_path / 'MNIST'
    mnist_available = (
        mnist_path.exists() and
        (mnist_path / 'mnist_train.csv').exists() and
        (mnist_path / 'mnist_test.csv').exists()
    )
    
    if mnist_available:
        print(f"   ✅ MNIST dataset available at {mnist_path}")
    else:
        print(f"   ❌ MNIST dataset not found at {mnist_path}")
    
    # Check ImageNet
    imagenet_path = base_path / 'ImageNet-1K'
    imagenet_available = imagenet_path.exists()
    
    if imagenet_available:
        print(f"   ✅ ImageNet dataset available at {imagenet_path}")
    else:
        print(f"   ⚠️  ImageNet dataset not found at {imagenet_path}")
        print(f"       ImageNet tests will be skipped")
    
    return mnist_available, imagenet_available


def main():
    """Run comprehensive end-to-end tests."""
    parser = argparse.ArgumentParser(description="Run MWNN end-to-end tests")
    parser.add_argument("--mnist-only", action="store_true", help="Run only MNIST tests")
    parser.add_argument("--imagenet-only", action="store_true", help="Run only ImageNet tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()
    
    print("🎯 MWNN Comprehensive End-to-End Test Suite")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages.")
        sys.exit(1)
    
    # Check data availability
    mnist_available, imagenet_available = check_data_availability()
    
    if not mnist_available and not imagenet_available:
        print("❌ No datasets available. Please prepare MNIST or ImageNet data.")
        sys.exit(1)
    
    # Determine which tests to run
    test_files = []
    test_path = Path(__file__).parent
    
    if not args.imagenet_only and mnist_available:
        test_files.append((
            test_path / 'test_end_to_end_mnist.py',
            'MNIST End-to-End Tests'
        ))
    
    if not args.mnist_only and imagenet_available:
        test_files.append((
            test_path / 'test_end_to_end_imagenet.py',
            'ImageNet End-to-End Tests'
        ))
    
    if not test_files:
        print("❌ No tests to run based on available data and arguments.")
        sys.exit(1)
    
    # Run tests
    results = []
    total_time = 0
    
    print(f"\n🚀 Starting {len(test_files)} test suites...")
    
    for test_file, test_name in test_files:
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            results.append((test_name, False, 0))
            continue
        
        success, elapsed_time = run_test_file(test_file, test_name)
        results.append((test_name, success, elapsed_time))
        total_time += elapsed_time
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_name, success, elapsed_time in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_name:35} ({elapsed_time:.2f}s)")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Summary:")
    print(f"   Total tests: {len(results)}")
    print(f"   Passed:     {passed}")
    print(f"   Failed:     {failed}")
    print(f"   Total time: {total_time:.2f} seconds")
    
    # Final verdict
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"MWNN end-to-end pipeline is working correctly.")
        exit_code = 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        print(f"Please check the output above for details.")
        exit_code = 1
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if mnist_available:
        print(f"   • MNIST tests provide quick validation of the MWNN API")
    if imagenet_available:
        print(f"   • ImageNet tests validate real-world performance")
    else:
        print(f"   • Consider adding ImageNet data for comprehensive testing")
    
    print(f"   • Run individual test files for more detailed debugging")
    print(f"   • Use --quick flag for faster validation during development")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
