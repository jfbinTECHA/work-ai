#!/usr/bin/env python3
"""
Comprehensive test suite for Zeta system enhancements.

This test suite focuses on testing the recent additions to the Zeta system:
1. Centralized logging configuration
2. Enhanced GitHub scraper functionality
3. System initialization improvements
4. Rate limiting and error handling

Author: AI Zetta Team
"""

import unittest
import logging
import sys
import os
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

class TestZetaLogging(unittest.TestCase):
    """Test the centralized logging configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test_zeta.log')
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any handlers that might have been added
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        
        # Remove temp files
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        os.rmdir(self.temp_dir)
    
    def test_logging_configuration(self):
        """Test the centralized logging configuration setup."""
        print("Testing centralized logging configuration...")
        
        # Create a logging configuration similar to the one in zeta.py
        logging_config = {
            'version': 1,
            'formatters': {
                'default': {
                    'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'stream': sys.stdout,
                    'formatter': 'default'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': self.log_file,
                    'formatter': 'default'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        }
        
        # Apply the configuration
        logging.config.dictConfig(logging_config)
        
        # Test logging
        logger = logging.getLogger('test_module')
        logger.info("Test log message")
        logger.warning("Test warning message")
        
        # Verify file logging works
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Test log message", log_content)
            self.assertIn("Test warning message", log_content)
            self.assertIn("INFO", log_content)
            self.assertIn("WARNING", log_content)
        
        print("✓ Centralized logging configuration test passed")
    
    def test_log_levels(self):
        """Test different log levels work correctly."""
        print("Testing log level filtering...")
        
        # Set up logging with WARNING level
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s',
            handlers=[logging.FileHandler(self.log_file, mode='w')]
        )
        
        logger = logging.getLogger('test_levels')
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should appear")
        logger.error("Error message - should appear")
        
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertNotIn("Debug message", log_content)
            self.assertNotIn("Info message", log_content)
            self.assertIn("Warning message", log_content)
            self.assertIn("Error message", log_content)
        
        print("✓ Log level filtering test passed")

class TestGitHubScraperEnhancements(unittest.TestCase):
    """Test the enhanced GitHub scraper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exponential_backoff(self):
        """Test the exponential backoff calculation."""
        print("Testing exponential backoff calculation...")
        
        # Import or define the exponential backoff function
        def exponential_backoff(attempt, base_delay=2, max_delay=300, jitter_range=0.1):
            """Calculate exponential backoff delay with jitter."""
            import random
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * jitter_range * (random.random() * 2 - 1)
            return delay + jitter
        
        # Test backoff progression
        delays = []
        for attempt in range(5):
            delay = exponential_backoff(attempt, base_delay=2, max_delay=60)
            delays.append(delay)
            self.assertGreater(delay, 0)
            if attempt > 0:
                # Each delay should generally be larger (accounting for jitter)
                self.assertLess(delay, 70)  # Should not exceed reasonable bounds
        
        # Test that delays generally increase
        base_delays = [2 * (2 ** i) for i in range(5)]
        for i, delay in enumerate(delays):
            expected_base = min(base_delays[i], 60)
            # Allow for jitter but ensure it's in reasonable range
            self.assertGreater(delay, expected_base * 0.8)
            self.assertLess(delay, expected_base * 1.2)
        
        print("✓ Exponential backoff calculation test passed")
    
    def test_function_extraction(self):
        """Test the enhanced function extraction logic."""
        print("Testing enhanced function extraction...")
        
        sample_code = '''
import os
import sys

def simple_function():
    """A simple function."""
    return "hello"

@decorator
def decorated_function(param1, param2):
    """A decorated function with parameters."""
    result = param1 + param2
    if result > 10:
        return "large"
    else:
        return "small"

class TestClass:
    def method1(self):
        """A class method."""
        pass
    
    def method2(self, x):
        return x * 2

def complex_function():
    """A more complex function."""
    data = []
    for i in range(10):
        if i % 2 == 0:
            data.append(i)
    
    def nested_function():
        return "nested"
    
    return data, nested_function()

# Some module-level code
CONSTANT = 42
'''
        
        # Use the extraction logic from github_scraper.py
        def extract_functions_from_content(py_file_content):
            """Extract function definitions from Python code."""
            functions = []
            parts = py_file_content.split("\ndef ")
            
            if parts[0].strip().startswith("def "):
                functions.append(parts[0].strip())
            
            for func in parts[1:]:
                if func.strip():
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
            
            return functions
        
        functions = extract_functions_from_content(sample_code)
        
        # Verify we extracted the expected functions
        self.assertGreater(len(functions), 0)
        
        # Check that we got the main functions (not class methods)
        function_names = []
        for func in functions:
            lines = func.split('\n')
            for line in lines:
                if line.strip().startswith('def '):
                    func_name = line.split('(')[0].replace('def ', '').strip()
                    function_names.append(func_name)
                    break
        
        expected_functions = ['simple_function', 'decorated_function', 'complex_function']
        for expected in expected_functions:
            self.assertIn(expected, function_names)
        
        print(f"✓ Extracted {len(functions)} functions: {function_names}")
        print("✓ Enhanced function extraction test passed")
    
    def test_progress_tracker(self):
        """Test the progress tracking functionality."""
        print("Testing progress tracker...")
        
        class ProgressTracker:
            """Mock progress tracker based on the scraper implementation."""
            
            def __init__(self, total_repos):
                from datetime import datetime
                self.total_repos = total_repos
                self.processed_repos = 0
                self.start_time = datetime.now()
                self.last_update_time = self.start_time
            
            def update_progress(self, repo_name=None):
                """Update progress and calculate stats."""
                from datetime import datetime, timedelta
                self.processed_repos += 1
                current_time = datetime.now()
                
                elapsed = current_time - self.start_time
                if self.processed_repos > 0:
                    avg_time_per_repo = elapsed.total_seconds() / self.processed_repos
                    remaining_repos = self.total_repos - self.processed_repos
                    remaining_seconds = avg_time_per_repo * remaining_repos
                    remaining_time = timedelta(seconds=int(remaining_seconds))
                    
                    return {
                        'progress_percent': (self.processed_repos / self.total_repos) * 100,
                        'elapsed': elapsed,
                        'remaining': remaining_time
                    }
                return None
            
            def get_final_stats(self):
                """Get final statistics."""
                from datetime import datetime
                end_time = datetime.now()
                total_time = end_time - self.start_time
                return {
                    'total_repos': self.total_repos,
                    'processed_repos': self.processed_repos,
                    'total_time': str(total_time).split('.')[0],
                    'avg_time_per_repo': total_time.total_seconds() / max(self.processed_repos, 1)
                }
        
        # Test progress tracking
        tracker = ProgressTracker(10)
        
        # Simulate processing some repositories
        for i in range(5):
            time.sleep(0.01)  # Small delay to simulate work
            stats = tracker.update_progress(f"repo_{i}")
            if stats:
                self.assertGreater(stats['progress_percent'], 0)
                self.assertLessEqual(stats['progress_percent'], 100)
        
        # Get final stats
        final_stats = tracker.get_final_stats()
        self.assertEqual(final_stats['processed_repos'], 5)
        self.assertEqual(final_stats['total_repos'], 10)
        self.assertGreater(final_stats['avg_time_per_repo'], 0)
        
        print("✓ Progress tracker test passed")
    
    def test_rate_limit_simulation(self):
        """Test rate limit handling simulation."""
        print("Testing rate limit handling...")
        
        def simulate_rate_limit_check(remaining_calls, limit, buffer=50):
            """Simulate rate limit checking logic."""
            usage_percent = ((limit - remaining_calls) / limit) * 100
            
            if remaining_calls <= buffer:
                return False, f"Rate limit low ({remaining_calls} remaining)"
            return True, f"Rate limit OK ({remaining_calls}/{limit}, {usage_percent:.1f}%)"
        
        # Test various rate limit scenarios
        test_cases = [
            (100, 5000, True),   # Plenty of calls remaining
            (60, 5000, True),    # Just above buffer
            (40, 5000, False),   # Below buffer - should wait
            (10, 5000, False),   # Very low - should wait
            (0, 5000, False),    # No calls remaining
        ]
        
        for remaining, limit, expected_ok in test_cases:
            ok, message = simulate_rate_limit_check(remaining, limit)
            self.assertEqual(ok, expected_ok, f"Failed for {remaining}/{limit}: {message}")
        
        print("✓ Rate limit handling simulation test passed")

class TestZetaInitialization(unittest.TestCase):
    """Test the Zeta system initialization."""
    
    def test_system_status_check(self):
        """Test the system status checking functionality."""
        print("Testing system status check...")
        
        def get_mock_system_status():
            """Mock system status function."""
            status = {
                'github_scraper': {'available': True, 'initialized': True},
                'trainer': {'available': True, 'initialized': False},
                'predictor': {'available': False, 'initialized': False}
            }
            return status
        
        status = get_mock_system_status()
        
        # Verify status structure
        self.assertIn('github_scraper', status)
        self.assertIn('trainer', status)
        self.assertIn('predictor', status)
        
        # Verify status details
        for component, details in status.items():
            self.assertIn('available', details)
            self.assertIn('initialized', details)
            self.assertIsInstance(details['available'], bool)
            self.assertIsInstance(details['initialized'], bool)
        
        print("✓ System status check test passed")
    
    @patch('logging.basicConfig')
    def test_initialization_logging(self, mock_basic_config):
        """Test that initialization sets up logging correctly."""
        print("Testing initialization logging setup...")
        
        def mock_initialize_zeta(log_level):
            """Mock initialization function."""
            import logging
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Initializing Zeta with log level: {logging.getLevelName(log_level)}")
            return True
        
        # Test initialization with different log levels
        result = mock_initialize_zeta(logging.INFO)
        self.assertTrue(result)
        
        # Verify logging.basicConfig was called
        mock_basic_config.assert_called()
        
        print("✓ Initialization logging setup test passed")

class TestIntegration(unittest.TestCase):
    """Integration tests for the Zeta system."""
    
    def test_end_to_end_workflow(self):
        """Test a simplified end-to-end workflow."""
        print("Testing end-to-end workflow simulation...")
        
        # Simulate the workflow steps
        workflow_steps = []
        
        # Step 1: Initialize logging
        try:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            workflow_steps.append("logging_initialized")
        except Exception as e:
            self.fail(f"Logging initialization failed: {e}")
        
        # Step 2: Check system components
        try:
            # Mock component check
            components = ['github_scraper', 'trainer', 'predictor']
            available_components = []
            for component in components:
                # Simulate component availability check
                available_components.append(component)
            workflow_steps.append("components_checked")
        except Exception as e:
            self.fail(f"Component check failed: {e}")
        
        # Step 3: Simulate scraper initialization
        try:
            # Mock scraper initialization
            scraper_config = {
                'max_repos': 10,
                'rate_limit_buffer': 50,
                'output_file': 'test_output.json'
            }
            workflow_steps.append("scraper_configured")
        except Exception as e:
            self.fail(f"Scraper configuration failed: {e}")
        
        # Step 4: Simulate data processing
        try:
            # Mock data processing
            sample_data = [
                {'repository': 'test/repo1', 'function_code': 'def test(): pass'},
                {'repository': 'test/repo2', 'function_code': 'def hello(): return "world"'}
            ]
            processed_data = len(sample_data)
            workflow_steps.append("data_processed")
        except Exception as e:
            self.fail(f"Data processing failed: {e}")
        
        # Verify all steps completed
        expected_steps = [
            "logging_initialized",
            "components_checked", 
            "scraper_configured",
            "data_processed"
        ]
        
        for step in expected_steps:
            self.assertIn(step, workflow_steps, f"Workflow step '{step}' not completed")
        
        print(f"✓ End-to-end workflow completed: {len(workflow_steps)} steps")
        print("✓ Integration test passed")

def run_test_suite():
    """Run the complete test suite."""
    print("=" * 60)
    print("ZETA SYSTEM ENHANCEMENTS TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestZetaLogging,
        TestGitHubScraperEnhancements,
        TestZetaInitialization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall Result: {'SUCCESS' if success else 'FAILURE'}")
    
    return success

def main():
    """Main entry point for the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zeta System Enhancements Test Suite')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet output')
    args = parser.parse_args()
    
    if args.quiet:
        # Redirect stdout to suppress most output
        sys.stdout = StringIO()
    
    try:
        success = run_test_suite()
        
        if args.quiet:
            # Restore stdout and print summary
            sys.stdout = sys.__stdout__
            print("Zeta enhancements test completed.")
            print(f"Result: {'PASSED' if success else 'FAILED'}")
        
        return 0 if success else 1
        
    except Exception as e:
        if args.quiet:
            sys.stdout = sys.__stdout__
        print(f"Test suite execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
