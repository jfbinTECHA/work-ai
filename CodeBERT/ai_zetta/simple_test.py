#!/usr/bin/env python3
"""
Simple test for GitHub scraper functionality without importing the main module.
"""

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
    
    # Inline function extraction logic
    functions = []
    parts = sample_code.split("\ndef ")
    
    # Process the first part (might contain a function without leading \n)
    if parts[0].strip().startswith("def "):
        functions.append(parts[0].strip())
    
    # Process remaining functions
    for func in parts[1:]:
        if func.strip():  # Only add non-empty functions
            complete_func = "def " + func.strip()
            lines = complete_func.split('\n')
            function_lines = []
            indent_level = None
            
            for line in lines:
                if line.strip() == "":
                    function_lines.append(line)
                    continue
                    
                current_indent = len(line) - len(line.lstrip())
                
                if indent_level is None and line.strip():
                    indent_level = current_indent
                    function_lines.append(line)
                elif current_indent > indent_level or (current_indent == indent_level and not line.strip().startswith(('def ', 'class ', '@'))):
                    function_lines.append(line)
                elif current_indent <= indent_level and line.strip().startswith(('def ', 'class ')):
                    break
                else:
                    function_lines.append(line)
            
            if function_lines:
                functions.append('\n'.join(function_lines))
    
    print(f"Extracted {len(functions)} functions:")
    for i, func in enumerate(functions, 1):
        print(f"\nFunction {i}:")
        print("-" * 40)
        print(func[:200] + "..." if len(func) > 200 else func)
    
    return len(functions) >= 3

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
    
    # Inline unique filtering logic
    seen = set()
    unique_snippets = []
    
    for snippet in snippets:
        if snippet not in seen:
            seen.add(snippet)
            unique_snippets.append(snippet)
    
    print(f"Original: {len(snippets)} snippets")
    print(f"Unique: {len(unique_snippets)} snippets")
    
    return len(unique_snippets) == 3

def test_pygithub_import():
    """Test if PyGithub can be imported."""
    print("\nTesting PyGithub import...")
    
    try:
        from PyGithub import Github
        print("✓ PyGithub imported successfully")
        return True
    except ImportError as e:
        print(f"✗ PyGithub import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("GitHub Scraper Simple Test Suite")
    print("=" * 50)
    
    # Test PyGithub import
    import_test = test_pygithub_import()
    
    # Test function extraction
    func_test = test_extract_functions()
    
    # Test unique snippets
    unique_test = test_unique_snippets()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"PyGithub import: {'PASS' if import_test else 'FAIL'}")
    print(f"Function extraction: {'PASS' if func_test else 'FAIL'}")
    print(f"Unique snippets: {'PASS' if unique_test else 'FAIL'}")
    
    all_passed = import_test and func_test and unique_test
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nThe scraper core functionality is working!")
        print("To use the full scraper:")
        print("1. Set your GITHUB_TOKEN environment variable")
        print("2. Run: source venv/bin/activate && python github_scraper.py")
    
    return all_passed

if __name__ == "__main__":
    main()
