from . import github_scraper
from . import trainer
from . import predictor
import logging
import argparse

def initialize_zeta(log_level):
    """
    Initialize the Zeta AI system with the specified logging level.

    This function sets up the logging configuration and initializes all
    Zeta components including the GitHub scraper, model trainer, and
    code predictor.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO)

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Set the logging level
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing Zeta with log level: {logging.getLevelName(log_level)}")

        # Initialize GitHub scraper
        logger.info("Initializing GitHub scraper...")
        github_scraper.init_scraper()
        logger.info("GitHub scraper initialized successfully")

        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer.init_trainer()
        logger.info("Model trainer initialized successfully")

        # Initialize predictor
        logger.info("Initializing code predictor...")
        predictor.init_predictor()
        logger.info("Code predictor initialized successfully")

        logger.info("Zeta initialization completed successfully")
        return True

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Zeta initialization failed: {e}")
        return False

def get_system_status():
    """
    Get the current status of all Zeta components.

    Returns:
        dict: Status information for each component
    """
    status = {
        'github_scraper': {'available': False, 'initialized': False},
        'trainer': {'available': False, 'initialized': False},
        'predictor': {'available': False, 'initialized': False}
    }

    try:
        import importlib
        components = ['github_scraper', 'trainer', 'predictor']

        for component in components:
            try:
                module = importlib.import_module(f'.{component}', package=__name__)
                status[component]['available'] = True
                if hasattr(module, 'init_' + component):
                    status[component]['initialized'] = True
            except ImportError:
                pass

    except Exception:
        pass

    return status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zeta AI System Initialization')
    parser.add_argument('-l', '--loglevel', help='Set the logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    args = parser.parse_args()

    log_level = getattr(logging, args.loglevel)
    success = initialize_zeta(log_level)

    if success:
        print("✓ Zeta system initialized successfully!")
        print(f"✓ Log level set to: {args.loglevel}")
    else:
        print("✗ Zeta system initialization failed!")
        exit(1)