# Import necessary libraries
import os
import json
import logging
import time
import hashlib
import argparse
import random
from datetime import datetime, timedelta
from itertools import chain
from PyGithub import Github
from typing import List, Set, Optional

# Define commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--loglevel', help='Set the logging level',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
args = parser.parse_args()

# Configure logging based on chosen log level
logging.basicConfig(level=getattr(logging, args.loglevel), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Enhanced GitHub Code Scraper Module

 This module provides advanced functionality to scrape Python code snippets from GitHub repositories.
 It uses the GitHub API to search for popular Python repositories, extract function-level
 code segments from Python files, and save them to a JSON file for further analysis or training.

 Enhanced Features:
 - Advanced rate limiting with exponential backoff and jitter
 - Progress tracking with time estimation
 - Enhanced error handling with retry logic
 - Better pagination handling
 - Comprehensive logging and monitoring
 - Graceful degradation when rate limits are hit
 - Detailed statistics and reporting

 Key Capabilities:
 - Searches GitHub repositories by language (Python) and sorts by star count
 - Extracts individual function definitions from Python source files
 - Handles GitHub API rate limiting with sophisticated retry logic
 - Removes duplicate code snippets based on MD5 checksums
 - Saves extracted data with comprehensive metadata
 - Provides real-time progress tracking and ETA calculations

 Usage:
     Set the GITHUB_TOKEN environment variable with a valid GitHub personal access token.
     Run the script directly: python github_scraper.py

 Dependencies:
 - PyGithub: For GitHub API interaction
 - Standard library: os, json, logging, time, hashlib, itertools, random, datetime

 Output:
 - scraped_code_snippets.json: JSON file containing unique function snippets with metadata

 Enhanced Rate Limiting:
 - Exponential backoff with jitter for retries
 - Separate handling of core and search API limits
 - Configurable buffers and retry limits
 - Detailed rate limit monitoring and logging
 - Graceful handling of network errors

 Progress Tracking:
 - Real-time progress updates every 10 repositories
 - Estimated completion time calculation
 - Detailed final statistics and performance metrics
 - Comprehensive error reporting and recovery

 Error Handling:
 - Robust retry logic with exponential backoff
 - Continues processing when individual items fail
 - Graceful handling of API errors and network issues
 - Detailed error logging with context
 - Keyboard interrupt handling for user control

 Author: AI Zetta Team
 """

# Define constants for GitHub token and repository query parameters

# Environment variable name for GitHub personal access token
# This token is required for API authentication and must be set before running the script
GITHUB_TOKEN_ENV_VAR = 'GITHUB_TOKEN'

# GitHub search query to filter repositories by programming language
# Uses GitHub's search syntax to find Python repositories
QUERY_LANGUAGE = 'language:python'

# Sort order for repository search results
# 'stars' sorts repositories by their star count in descending order (most popular first)
SORT_ORDER = 'stars'

# Maximum number of repositories to process
# Limits the scope of scraping to prevent excessive API usage and processing time
MAX_REPOS = 100

# Output filename for the scraped code snippets
# JSON file containing all extracted function snippets with metadata
OUTPUT_FILE = 'scraped_code_snippets.json'

# Enhanced rate limiting constants
RATE_LIMIT_BUFFER = 50  # Buffer for rate limits (keep more calls in reserve)
SEARCH_RATE_LIMIT_BUFFER = 5  # More conservative buffer for search API
MAX_RETRIES = 3  # Maximum number of retries for failed requests
BASE_DELAY = 2  # Base delay in seconds for exponential backoff
MAX_DELAY = 300  # Maximum delay in seconds (5 minutes)
JITTER_RANGE = 0.1  # Random jitter as percentage of delay

# Initialize PyGithub client with personal access token
# This creates the authenticated connection to GitHub's API
try:
    g = Github(os.environ[GITHUB_TOKEN_ENV_VAR])
    logger.info("GitHub API initialized successfully")
except KeyError:
    # Handle missing environment variable (most common setup error)
    logger.error(f"Environment variable {GITHUB_TOKEN_ENV_VAR} not found")
    raise
except Exception as e:
    # Handle other initialization errors (invalid token, network issues, etc.)
    logger.error(f"Failed to initialize GitHub API: {e}")
    raise

def exponential_backoff(attempt: int, base_delay: float = BASE_DELAY) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Args:
        attempt (int): The current attempt number (0-based)
        base_delay (float): Base delay in seconds

    Returns:
        float: Delay in seconds to wait
    """
    delay = min(base_delay * (2 ** attempt), MAX_DELAY)
    jitter = delay * JITTER_RANGE * (random.random() * 2 - 1)
    return delay + jitter

def check_rate_limit(g: Github, api_type: str = 'core', retry_count: int = 0) -> bool:
    """
    Enhanced rate limit checking with exponential backoff and better error handling.

    This function monitors the remaining API calls for the specified API type
    and automatically waits until the rate limit resets if the remaining calls
    drop below a safety threshold. Uses exponential backoff for retries.

    Args:
        g (Github): Authenticated PyGithub Github instance
        api_type (str): Type of API to check. Either 'core' or 'search'
        retry_count (int): Current retry attempt number

    Returns:
        bool: True if rate limit check passed, False if should retry

    Behavior:
        - Retrieves current rate limit status from GitHub API
        - Uses different thresholds for core vs search API
        - Implements exponential backoff with jitter for retries
        - Logs detailed rate limit status and wait times
        - Handles network errors gracefully
    """
    try:
        rate_limit = g.get_rate_limit()

        if api_type == 'core':
            remaining = rate_limit.core.remaining
            limit = rate_limit.core.limit
            reset_time = rate_limit.core.reset
            buffer = RATE_LIMIT_BUFFER
        elif api_type == 'search':
            remaining = rate_limit.search.remaining
            limit = rate_limit.search.limit
            reset_time = rate_limit.search.reset
            buffer = SEARCH_RATE_LIMIT_BUFFER
        else:
            logger.warning(f"Unknown API type: {api_type}")
            return True

        # Calculate usage percentage
        usage_percent = ((limit - remaining) / limit) * 100

        # Log current rate limit status
        logger.debug(f"Rate limit status for {api_type}: {remaining}/{limit} ({usage_percent:.1f}%)")

        # Check if we need to wait
        if remaining <= buffer:
            wait_time = max(0, reset_time.timestamp() - time.time())

            if wait_time > 0:
                # Add exponential backoff for retries
                if retry_count > 0:
                    backoff_delay = exponential_backoff(retry_count - 1)
                    wait_time = max(wait_time, backoff_delay)
                    logger.info(f"Rate limit low ({remaining} remaining for {api_type}) + retry backoff. Waiting {wait_time:.0f} seconds.")
                else:
                    logger.info(f"Rate limit low ({remaining} remaining for {api_type}). Waiting {wait_time:.0f} seconds until reset.")

                time.sleep(wait_time)
                return False  # Signal that we waited and should retry
            else:
                logger.warning("Rate limit reset time is in the past - proceeding with caution")

        return True  # Rate limit check passed

    except Exception as e:
        logger.warning(f"Failed to check rate limit ({api_type}): {e}")

        # If we can't check rate limits, use exponential backoff
        if retry_count < MAX_RETRIES:
            delay = exponential_backoff(retry_count)
            logger.info(f"Rate limit check failed, using backoff delay: {delay:.1f}s")
            time.sleep(delay)
            return False

        logger.error("Max retries exceeded for rate limit check")
        return True  # Proceed anyway to avoid infinite loop

class ProgressTracker:
    """
    Tracks scraping progress and estimates completion time.
    """

    def __init__(self, total_repos: int):
        self.total_repos = total_repos
        self.processed_repos = 0
        self.start_time = datetime.now()
        self.last_update_time = self.start_time

    def update_progress(self, repo_name: str = None) -> None:
        """Update progress and log status."""
        self.processed_repos += 1
        current_time = datetime.now()

        # Calculate elapsed time
        elapsed = current_time - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

        # Estimate remaining time
        if self.processed_repos > 0:
            avg_time_per_repo = elapsed.total_seconds() / self.processed_repos
            remaining_repos = self.total_repos - self.processed_repos
            remaining_seconds = avg_time_per_repo * remaining_repos
            remaining_time = timedelta(seconds=int(remaining_seconds))

            # Log progress every 10 repositories or when significant time has passed
            time_since_update = current_time - self.last_update_time
            if self.processed_repos % 10 == 0 or time_since_update.total_seconds() > 300:
                progress_percent = (self.processed_repos / self.total_repos) * 100
                logger.info(f"Progress: {self.processed_repos}/{self.total_repos} ({progress_percent:.1f}%) - "
                          f"Elapsed: {elapsed_str}, Remaining: ~{remaining_time}")
                if repo_name:
                    logger.info(f"Currently processing: {repo_name}")
                self.last_update_time = current_time

    def get_final_stats(self) -> dict:
        """Get final statistics."""
        end_time = datetime.now()
        total_time = end_time - self.start_time
        return {
            'total_repos': self.total_repos,
            'processed_repos': self.processed_repos,
            'total_time': str(total_time).split('.')[0],
            'avg_time_per_repo': total_time.total_seconds() / max(self.processed_repos, 1)
        }

def get_unique_snippets(code_snippets: List[str]) -> List[str]:
    """
    Remove duplicate code snippets from a list using exact string matching.

    This function iterates through a list of code snippets and filters out
    duplicates by maintaining a set of seen strings. Only the first occurrence
    of each unique snippet is kept, preserving the original order.

    Args:
        code_snippets (List[str]): A list of strings, where each string represents
                                  a code snippet (typically a function definition).

    Returns:
        List[str]: A new list containing only unique code snippets in their
                  original order. Duplicates are removed based on exact string
                  equality.

    Behavior:
        - Uses a set for O(1) lookup time to check for duplicates
        - Maintains insertion order of first occurrences
        - Logs the filtering statistics (original count vs unique count)
        - Does not modify the input list

    Performance:
        - Time complexity: O(n) where n is the number of snippets
        - Space complexity: O(n) for the set and output list
        - Efficient for large lists as set operations are fast

    Note:
        This function performs exact string matching. Minor differences like
        whitespace changes would be treated as different snippets. For more
        sophisticated deduplication, consider using checksums or AST comparison.
    """
    seen: Set[str] = set()
    unique_snippets = []
    
    for snippet in code_snippets:
        if snippet not in seen:
            seen.add(snippet)
            unique_snippets.append(snippet)
    
    logger.info(f"Filtered {len(code_snippets)} snippets down to {len(unique_snippets)} unique snippets")
    return unique_snippets

def extract_functions_from_content(py_file_content: str) -> List[str]:
    """
    Parse Python file content and extract individual function definitions.

    This function performs syntactic analysis on Python source code to identify
    and extract complete function definitions. It uses simple string splitting
    and indentation analysis to isolate functions from the rest of the code.

    Args:
        py_file_content (str): The complete content of a Python file as a string.
                              Should be valid Python code with proper indentation.

    Returns:
        List[str]: A list of strings, where each string contains the complete
                  code for one function definition, including the 'def' line
                  and all indented body lines.

    Extraction Algorithm:
        1. Split the file content by '\ndef ' to separate function definitions
        2. Handle the first part (may contain a function without leading newline)
        3. For each potential function:
           - Reconstruct the complete function with 'def ' prefix
           - Analyze line-by-line indentation to determine function boundaries
           - Stop at the next function/class definition at the same indent level
           - Include decorators and docstrings within the function

    Limitations:
        - May not handle complex cases like nested functions perfectly
        - Assumes standard Python indentation (4 spaces)
        - May include some non-function code if indentation analysis fails
        - Does not validate Python syntax (assumes input is valid Python)

    Edge Cases Handled:
        - Functions at the beginning of file (no leading newline)
        - Empty or whitespace-only functions
        - Functions with decorators
        - Nested code structures within functions

    Example:
        Input: "def foo():\n    return 1\n\ndef bar():\n    return 2"
        Output: ["def foo():\n    return 1", "def bar():\n    return 2"]
    """
    functions = []
    
    # Split Python file into potential functions using 'def ' as separator
    # This handles most cases but may split incorrectly for strings containing 'def '
    parts = py_file_content.split("\ndef ")

    # Process the first part (might contain a function without leading \n)
    if parts[0].strip().startswith("def "):
        functions.append(parts[0].strip())

    # Process remaining parts that should start with function definitions
    for func in parts[1:]:
        if func.strip():  # Skip empty parts
            # Reconstruct the complete function by adding back the 'def ' prefix
            complete_func = "def " + func.strip()

            # Parse the function line by line to extract only the function definition
            # We need to stop at the next function/class at the same indentation level
            lines = complete_func.split('\n')
            function_lines = []
            indent_level = None  # Will store the indentation level of the def line

            for line in lines:
                if line.strip() == "":
                    # Preserve empty lines within the function
                    function_lines.append(line)
                    continue

                # Calculate indentation level (number of leading spaces)
                current_indent = len(line) - len(line.lstrip())

                # Set the base indentation level from the first non-empty line (should be 'def ')
                if indent_level is None and line.strip():
                    indent_level = current_indent
                    function_lines.append(line)
                # Include lines that are more indented (function body) or same level but not new def/class
                elif current_indent > indent_level or (current_indent == indent_level and not line.strip().startswith(('def ', 'class ', '@'))):
                    function_lines.append(line)
                # Stop when we encounter another function or class at the same or higher level
                elif current_indent <= indent_level and line.strip().startswith(('def ', 'class ')):
                    break  # Exit the loop, we've reached the end of this function
                else:
                    # Include other lines (comments, etc.) that don't match the stop conditions
                    function_lines.append(line)

            # Add the extracted function if we got any lines
            if function_lines:
                functions.append('\n'.join(function_lines))
    
    return functions

def scrape_github() -> None:
    """
    Enhanced main scraping function with improved rate limiting and progress tracking.

    This function performs a comprehensive search of GitHub repositories, extracts Python
    function definitions from source files, and saves the results to a JSON file. It
    features enhanced rate limiting, exponential backoff, and detailed progress tracking.

    Process Overview:
        1. Search GitHub for Python repositories sorted by star count
        2. For each repository (up to MAX_REPOS limit):
           - Get repository contents recursively
           - Identify Python files (.py extension)
           - Extract function definitions from each Python file
           - Collect metadata (repository name, file path, star count)
        3. Remove duplicate functions based on MD5 checksums
        4. Save results to JSON file with comprehensive metadata

    Enhanced Features:
        - Exponential backoff for rate limit retries
        - Progress tracking with time estimation
        - Enhanced error handling with retry logic
        - Better pagination handling
        - Detailed logging of rate limit status

    Args:
        None: This function doesn't take parameters. It uses global constants
              for configuration (GITHUB_TOKEN, QUERY_LANGUAGE, etc.).

    Returns:
        None: Results are saved to OUTPUT_FILE. Success/failure is logged.
    """
    all_code_snippets = []
    processed_repos = 0
    retry_count = 0

    # Initialize progress tracker
    progress = ProgressTracker(MAX_REPOS)

    try:
        # Check rate limit before making the search API call
        logger.info("Starting GitHub repository search...")
        rate_check_passed = check_rate_limit(g, 'search', retry_count)

        if not rate_check_passed:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.error("Max retries exceeded for search API rate limit")
                return
            logger.info(f"Retrying search API call (attempt {retry_count})")
            time.sleep(exponential_backoff(retry_count - 1))

        # Perform GitHub repository search using the configured query and sort order
        logger.info(f"Searching for repositories with query: {QUERY_LANGUAGE}")
        query = g.search_repositories(query=QUERY_LANGUAGE, sort=SORT_ORDER)

        # Process each repository from the search results
        for repo in query:
            # Enforce the maximum repository limit
            if processed_repos >= MAX_REPOS:
                logger.info(f"Reached maximum repository limit: {MAX_REPOS}")
                break

            try:
                # Log which repository we're currently processing
                logger.info(f"Processing repository: {repo.full_name}")

                # Check core API rate limit before accessing repository contents
                rate_check_passed = check_rate_limit(g, 'core', retry_count)

                if not rate_check_passed:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        logger.error("Max retries exceeded for core API rate limit")
                        break
                    logger.info(f"Retrying core API call (attempt {retry_count})")
                    time.sleep(exponential_backoff(retry_count - 1))
                    continue

                # Reset retry count on successful API call
                retry_count = 0

                # Get the root directory contents of the repository
                repo_contents = repo.get_contents("")
                logger.debug(f"Found {len(repo_contents)} items in repository root")

                # Collect all Python files that need to be processed
                files_to_process = []

                # Iterate through root contents to find Python files and subdirectories
                for content in repo_contents:
                    if content.type == "file" and content.name.endswith(".py"):
                        # Direct Python file in root directory
                        files_to_process.append(content)
                    elif content.type == "dir":
                        # Directory - recursively get Python files from subdirectories
                        # We limit depth to 1 level to avoid excessive API calls
                        try:
                            # Check rate limit before subdirectory access
                            rate_check_passed = check_rate_limit(g, 'core')

                            if not rate_check_passed:
                                logger.warning(f"Skipping subdirectory {content.path} due to rate limits")
                                continue

                            subdir_contents = repo.get_contents(content.path)
                            # Add Python files from this subdirectory
                            for subcontent in subdir_contents:
                                if subcontent.type == "file" and subcontent.name.endswith(".py"):
                                    files_to_process.append(subcontent)
                        except Exception as e:
                            # Log warning but continue - some subdirs may be inaccessible
                            logger.warning(f"Could not access subdirectory {content.path}: {e}")
                            continue

                logger.debug(f"Collected {len(files_to_process)} Python files to process")

                # Process each collected Python file
                for py_file in files_to_process:
                    try:
                        # Decode the file content from bytes to string using UTF-8 encoding
                        py_file_content = py_file.decoded_content.decode("utf-8")

                        # Extract individual function definitions from the file content
                        functions = extract_functions_from_content(py_file_content)

                        # Process each extracted function
                        for func in functions:
                            # Filter out very short functions (likely stubs or incomplete)
                            if len(func.strip()) > 50:  # Minimum 50 characters for substantial functions
                                # Generate MD5 checksum for deduplication across repositories
                                checksum = hashlib.md5(func.strip().encode('utf-8')).hexdigest()
                                # Create structured data object for this function snippet
                                snippet_data = {
                                    'repository': repo.full_name,        # e.g., 'owner/repo-name'
                                    'file_path': py_file.path,           # path within repository
                                    'function_code': func.strip(),       # complete function code
                                    'checksum': checksum,                # for duplicate detection
                                    'stars': repo.stargazers_count,      # repository popularity
                                    'language': 'python',                # fixed value
                                    'scraped_at': datetime.now().isoformat()  # timestamp
                                }
                                # Add to our collection of all snippets
                                all_code_snippets.append(snippet_data)

                        # Log progress for this file
                        logger.debug(f"Extracted {len(functions)} functions from {py_file.path}")

                    except Exception as e:
                        # Log file processing errors but continue with other files
                        logger.warning(f"Error processing file {py_file.path}: {e}")
                        continue

                # Log total functions extracted from this repository
                repo_functions = len([s for s in all_code_snippets if s['repository'] == repo.full_name])
                logger.info(f"Extracted {repo_functions} total function snippets from {repo.full_name}")

                processed_repos += 1
                progress.update_progress(repo.full_name)

            except Exception as e:
                # Handle repository-level errors (API limits, access denied, etc.)
                logger.error(f"Error processing repository {repo.full_name}: {e}")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    logger.info(f"Retrying repository processing (attempt {retry_count})")
                    time.sleep(exponential_backoff(retry_count - 1))
                    continue
                else:
                    logger.error(f"Max retries exceeded for repository {repo.full_name}")
                    retry_count = 0
                    continue

    except Exception as e:
        # Handle search-level errors (network issues, API changes, etc.)
        logger.error(f"Error during repository search: {e}")
        return
    
    # Remove duplicate code snippets using checksum-based deduplication
    logger.info("Removing duplicate code snippets based on checksum...")
    unique_snippet_data = []
    seen_checksums = set()  # Set for O(1) lookup performance

    # Iterate through all collected snippets and keep only unique ones
    for snippet in all_code_snippets:
        if snippet['checksum'] not in seen_checksums:
            seen_checksums.add(snippet['checksum'])
            unique_snippet_data.append(snippet)

    # Get final statistics
    final_stats = progress.get_final_stats()

    # Save the final deduplicated data to JSON file
    try:
        logger.info("Saving results to JSON file...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(unique_snippet_data, f, indent=2, ensure_ascii=False)

        # Log comprehensive final statistics
        logger.info("=" * 60)
        logger.info("GITHUB SCRAPING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total repositories processed: {processed_repos}")
        logger.info(f"Total unique code snippets: {len(unique_snippet_data)}")
        logger.info(f"Total checksums computed: {len(seen_checksums)}")
        logger.info(f"Total processing time: {final_stats['total_time']}")
        logger.info(f"Average time per repository: {final_stats['avg_time_per_repo']:.2f} seconds")
        logger.info(f"Results saved to: {OUTPUT_FILE}")
        logger.info("=" * 60)

        # Report final rate limit status
        try:
            rate_limit = g.get_rate_limit()
            logger.info("Final API Usage:")
            logger.info(f"  Core API: {rate_limit.core.remaining}/{rate_limit.core.limit}")
            logger.info(f"  Search API: {rate_limit.search.remaining}/{rate_limit.search.limit}")
        except Exception as e:
            logger.debug(f"Could not retrieve final rate limit status: {e}")

    except Exception as e:
        # Handle file I/O errors (permissions, disk space, etc.)
        logger.error(f"Error saving data to file: {e}")
        logger.error("Scraping completed but results could not be saved")

def main():
    """
    Entry point for the GitHub code scraper application.

    This function serves as the main entry point when the script is run directly.
    It performs initial setup, validates prerequisites, executes the scraping
    process, and provides final status reporting.

    Execution Flow:
        1. Log the start of the scraping process
        2. Validate that the GitHub token environment variable is set
        3. Call scrape_github() to perform the actual scraping
        4. Log completion status and final rate limit information
        5. Handle any exceptions that occur during execution

    Prerequisites:
        - GITHUB_TOKEN environment variable must be set with a valid GitHub
          personal access token that has appropriate permissions
        - Network connectivity to GitHub API
        - Write permissions for output file creation

    Error Handling:
        - Checks for missing GitHub token before starting
        - Catches and logs any exceptions during scraping
        - Provides clear error messages for common issues

    Logging:
        - Logs start and completion of scraping process
        - Reports final API rate limit status
        - Includes timestamps and log levels for monitoring

    Returns:
        None: This is the main entry point and doesn't return values.
              Success/failure is indicated through logging and exit codes.

    Usage:
        python github_scraper.py
        (assuming GITHUB_TOKEN is set in environment)
    """
    logger.info("Starting GitHub code scraper...")

    # Validate that the required GitHub token environment variable exists
    if GITHUB_TOKEN_ENV_VAR not in os.environ:
        logger.error(f"Please set the {GITHUB_TOKEN_ENV_VAR} environment variable")
        return

    try:
        # Execute the main scraping logic
        scrape_github()
        logger.info("GitHub scraping completed successfully!")
    except KeyboardInterrupt:
        logger.info("GitHub scraping interrupted by user")
        logger.info("Partial results may have been saved")
    except Exception as e:
        # Catch and log any unexpected errors during execution
        logger.error(f"GitHub scraping failed: {e}")
        logger.error("Check your GitHub token and network connection")

def init_scraper():
    """
    Initialize the GitHub scraper component.

    This function performs basic initialization checks for the scraper
    including verifying dependencies and setting up the GitHub API client.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing GitHub scraper component...")

        # Check if required dependencies are available
        try:
            from PyGithub import Github
            logger.debug("PyGithub library available")
        except ImportError:
            logger.warning("PyGithub library not available - scraper functionality limited")
            return False

        # Check environment variables
        if GITHUB_TOKEN_ENV_VAR in os.environ:
            logger.debug("GitHub token found in environment")
        else:
            logger.warning(f"GitHub token not found - set {GITHUB_TOKEN_ENV_VAR} to use scraper")

        # Initialize GitHub client (will fail gracefully if token not available)
        try:
            g = Github(os.environ.get(GITHUB_TOKEN_ENV_VAR, ''))
            logger.debug("GitHub client initialized")
        except Exception as e:
            logger.debug(f"GitHub client initialization deferred: {e}")

        logger.info("GitHub scraper component initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize GitHub scraper: {e}")
        return False

if __name__ == "__main__":
    main()
