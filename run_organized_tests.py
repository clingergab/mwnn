#!/usr/bin/env python3
"""
Test runner for the Multi-Weight Neural Networks project.

This script provides an organized way to run different categories of tests.
"""

import subprocess
import sys
import os
import argparse

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=os.getcwd())
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Weight Neural Networks tests")
    parser.add_argument('--category', '-c', 
                       choices=['all', 'unit', 'integration', 'verification', 'components', 'models'],
                       default='all',
                       help='Category of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Run tests with verbose output')
    
    args = parser.parse_args()
    
    verbose_flag = "-v" if args.verbose else ""
    passed = []
    failed = []
    
    print("🧪 MULTI-WEIGHT NEURAL NETWORKS TEST RUNNER")
    print("=" * 60)
    
    if args.category in ['all', 'unit', 'components']:
        # Component tests
        success = run_command(
            f"python3 -m pytest tests/components/ {verbose_flag}",
            "Component Tests (neurons, layers, blocks)"
        )
        (passed if success else failed).append("Components")
    
    if args.category in ['all', 'unit', 'models']:
        # Model tests
        success = run_command(
            f"python3 -m pytest tests/models/ {verbose_flag}",
            "Model Tests (all architectures)"
        )
        (passed if success else failed).append("Models")
    
    if args.category in ['all', 'unit']:
        # Preprocessing tests
        success = run_command(
            f"python3 -m pytest tests/preprocessing/ {verbose_flag}",
            "Preprocessing Tests (data loaders, feature extractors)"
        )
        (passed if success else failed).append("Preprocessing")
        
        # Training tests
        success = run_command(
            f"python3 -m pytest tests/training/ {verbose_flag}",
            "Training Tests (losses, metrics, trainer)"
        )
        (passed if success else failed).append("Training")
    
    if args.category in ['all', 'integration']:
        # Integration tests
        success = run_command(
            f"python3 tests/integration/test_multi_channel_comprehensive.py",
            "Multi-Channel Model Comprehensive Test"
        )
        (passed if success else failed).append("Multi-Channel Comprehensive")
        
        success = run_command(
            f"python3 tests/integration/test_neuron_integration.py",
            "MultiChannelNeuron Integration Test"
        )
        (passed if success else failed).append("Neuron Integration")
        
        success = run_command(
            f"python3 tests/integration/test_all_neurons.py",
            "All Neurons DESIGN.md Compliance Test"
        )
        (passed if success else failed).append("All Neurons Compliance")
    
    if args.category in ['all', 'verification']:
        # Verification scripts
        success = run_command(
            f"python3 tests/verification/verify_option_1a.py",
            "Option 1A DESIGN.md Verification"
        )
        (passed if success else failed).append("Option 1A Verification")
        
        success = run_command(
            f"python3 tests/verification/verify_final.py",
            "Final MultiChannelNeuron Verification"
        )
        (passed if success else failed).append("Final Verification")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    if passed:
        print(f"✅ PASSED ({len(passed)}):")
        for test in passed:
            print(f"   • {test}")
    
    if failed:
        print(f"\n❌ FAILED ({len(failed)}):")
        for test in failed:
            print(f"   • {test}")
    
    total_tests = len(passed) + len(failed)
    success_rate = len(passed) / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n📊 Success Rate: {success_rate:.1f}% ({len(passed)}/{total_tests})")
    
    if failed:
        print("\n🔧 To debug failures, run individual test categories with -v flag")
        sys.exit(1)
    else:
        print("\n🎉 ALL TESTS PASSED! Project is ready for research and production.")
        sys.exit(0)

if __name__ == "__main__":
    main()
