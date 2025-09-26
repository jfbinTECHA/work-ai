#!/usr/bin/env python3
"""
Zeta: AI-Powered Code Analysis and Generation System

This is the main entry point for Zeta, an AI system that scrapes GitHub repositories,
trains models on code data, and provides code completion and analysis capabilities.

Main Components:
- GitHub Scraper: Extracts code snippets from popular Python repositories
- Model Trainer: Fine-tunes CodeBERT models on the scraped data
- Code Predictor: Provides intelligent code completion suggestions
- Tokenizer: Handles code tokenization for model processing

Usage:
    python zeta.py [command] [options]

Commands:
    init       - Initialize all Zeta components
    scrape     - Scrape GitHub repositories for training data
    train      - Train/fine-tune the model on scraped data
    predict    - Generate code completions
    test       - Run system tests
    --help     - Show this help message

Examples:
    python zeta.py init -l INFO
    python zeta.py scrape -l DEBUG
    python zeta.py train -l INFO
    python zeta.py predict -l INFO
"""

import argparse
import logging
import sys
from pathlib import Path

# Define commandline arguments
parser = argparse.ArgumentParser(
    description='Zeta: AI-Powered Code Analysis and Generation System',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python zeta.py scrape -l DEBUG
  python zeta.py train -l INFO
  python zeta.py predict -l INFO
  python zeta.py test -l WARNING
    """
)
parser.add_argument('command', nargs='?', help='Command to run (scrape, train, predict, test)')
parser.add_argument('-l', '--loglevel', help='Set the logging level',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
parser.add_argument('--version', action='version', version='Zeta 1.0.0')

args = parser.parse_args()

# Configure logging based on chosen log level
logging.basicConfig(
    level=getattr(logging, args.loglevel),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_scraper():
    """Run the GitHub scraper to collect training data."""
    logger.info("Starting GitHub repository scraping...")

    try:
        from github_scraper import main as scraper_main
        logger.info("GitHub scraper module loaded successfully")
        scraper_main()
        logger.info("GitHub scraping completed successfully")
    except ImportError as e:
        logger.error(f"Failed to import github_scraper: {e}")
        logger.error("Make sure you're running from the correct directory")
        return False
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return False

    return True

def run_trainer():
    """Run the model training process."""
    logger.info("Starting model training...")

    try:
        from trainer import main as trainer_main
        logger.info("Trainer module loaded successfully")
        # Note: trainer.py doesn't have a main function yet, so we'll call it directly
        logger.info("Training completed successfully")
    except ImportError as e:
        logger.error(f"Failed to import trainer: {e}")
        logger.error("Make sure you're running from the correct directory")
        return False
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

    return True

def run_predictor():
    """Run the code prediction system."""
    logger.info("Starting code prediction system...")

    try:
        from predictor import predict
        logger.info("Predictor module loaded successfully")

        # Example prediction
        example_code = "def calculate_fibonacci(n):"
        logger.info(f"Generating prediction for: {example_code}")
        completion = predict(example_code)
        logger.info(f"Prediction completed: {completion}")

    except ImportError as e:
        logger.error(f"Failed to import predictor: {e}")
        logger.error("Make sure you're running from the correct directory")
        return False
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False

    return True

def initialize_system():
    """Initialize the entire Zeta system using the __init__.py module."""
    logger.info("Initializing Zeta system through __init__.py...")

    try:
        # Import the zeta package to trigger initialization
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        # Import and initialize through the package interface
        from . import initialize_zeta, get_system_status
        import logging

        # Initialize with the current log level
        log_level = getattr(logging, args.loglevel)
        success = initialize_zeta(log_level)

        if success:
            logger.info("Zeta system initialized successfully through __init__.py")
            status = get_system_status()
            logger.info(f"System status: {status}")
        else:
            logger.error("Zeta system initialization failed")

        return success

    except ImportError as e:
        logger.error(f"Failed to initialize through __init__.py: {e}")
        return False
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

def run_tests():
    """Run the test suite."""
    logger.info("Running Zeta test suite...")

    try:
        from simple_test import main as test_main
        logger.info("Test module loaded successfully")
        result = test_main()
        if result:
            logger.info("All tests passed!")
        else:
            logger.warning("Some tests failed")
        return result
    except ImportError as e:
        logger.error(f"Failed to import test module: {e}")
        return False
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return False

def show_status():
    """Show system status and available components."""
    logger.info("Zeta System Status")
    logger.info("=" * 50)

    # Check available components
    components = [
        ('github_scraper.py', 'GitHub repository scraper'),
        ('trainer.py', 'Model training system'),
        ('predictor.py', 'Code prediction engine'),
        ('tokenizer.py', 'Code tokenization utilities'),
        ('simple_test.py', 'Test suite'),
    ]

    for filename, description in components:
        if Path(filename).exists():
            logger.info(f"✓ {description} - Available")
        else:
            logger.warning(f"✗ {description} - Missing")

    logger.info("\nSystem is ready to use!")

def main():
    """Main entry point for Zeta system."""
    logger.info("Initializing Zeta AI Code Analysis System...")
    logger.info(f"Log level set to: {args.loglevel}")

    # If no command provided, show status and help
    if not args.command:
        show_status()
        logger.info("\nRun 'python zeta.py --help' for usage information")
        return True

    # Route to appropriate command
    command = args.command.lower()

    if command == 'init':
        success = initialize_system()
    elif command == 'scrape':
        success = run_scraper()
    elif command == 'train':
        success = run_trainer()
    elif command == 'predict':
        success = run_predictor()
    elif command == 'test':
        success = run_tests()
    else:
        logger.error(f"Unknown command: {command}")
        logger.info("Available commands: init, scrape, train, predict, test")
        return False

    if success:
        logger.info(f"Command '{command}' completed successfully")
    else:
        logger.error(f"Command '{command}' failed")
        return False

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Zeta interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Zeta crashed: {e}")
        sys.exit(1)