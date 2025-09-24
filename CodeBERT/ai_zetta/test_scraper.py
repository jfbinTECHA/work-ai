#!/usr/bin/env python3
"""
Test script for the GitHub scraper to verify functionality.
"""

import os
import sys
import subprocess
from github_scraper import extract_functions_from_content, get_unique_snippets

def test_extract_functions():
    """Test the function extraction logic."""
    print("Testing function extraction...")
    
    sample_code = '''
import os

def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "success"

def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    result = a + b
    return result

class MyClass:
    def method1(self):
        pass

def another_function():
    """Another function for testing."""
    x = 1
    y = 2
    return x + y
'''
    
    functions = extract_functions_from_content(sample_code)
    print(f"Extracted {len(functions)} functions:")
    for i, func in enumerate(functions, 1):
        print(f"\nFunction {i}:")
        print("-" * 40)
        print(func[:200] + "..." if len(func) > 200 else func)
    
    return len(functions) >= 3  # Should extract at least 3 functions

def test_unique_snippets():
    """Test the unique snippets filtering."""
    print("\nTesting unique snippets filtering...")
    
    snippets = [
        "def func1(): pass",
        "def func2(): return 1",
        "def func1(): pass",  # duplicate
        "def func3(): print('hello')",
        "def func2(): return 1",  # duplicate
    ]
    
    unique = get_unique_snippets(snippets)
    print(f"Original: {len(snippets)} snippets")
    print(f"Unique: {len(unique)} snippets")
    
    return len(unique) == 3  # Should have 3 unique snippets

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    try:
        import PyGithub
        print("✓ PyGithub is installed")
        return True
    except ImportError:
        print("✗ PyGithub is not installed")
        print("Install with: pip install PyGithub")
        return False

def main():
    """Run all tests."""
    print("GitHub Scraper Test Suite")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\nPlease install missing dependencies before running the scraper.")
        return False
    
    # Test function extraction
    func_test = test_extract_functions()
    
    # Test unique snippets
    unique_test = test_unique_snippets()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Function extraction: {'PASS' if func_test else 'FAIL'}")
    print(f"Unique snippets: {'PASS' if unique_test else 'FAIL'}")
    print(f"Dependencies: {'PASS' if deps_ok else 'FAIL'}")
    
    all_passed = func_test and unique_test and deps_ok
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nThe scraper is ready to use!")
        print("Don't forget to set your GITHUB_TOKEN environment variable.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
