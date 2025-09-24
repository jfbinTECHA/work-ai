# Import necessary libraries
import os
import json
import logging
import time
import hashlib
from itertools import chain
from PyGithub import Github
from typing import List, Set
"""
GitHub Code Scraper Module

This module provides functionality to scrape Python code snippets from GitHub repositories.
It uses the GitHub API to search for popular Python repositories, extract function-level
code segments from Python files, and save them to a JSON file for further analysis or training.

Key Features:
- Searches GitHub repositories by language (Python) and sorts by star count
- Extracts individual function definitions from Python source files
- Handles GitHub API rate limiting automatically
- Removes duplicate code snippets based on MD5 checksums
- Saves extracted data with metadata (repository, file path, star count, etc.)

Usage:
    Set the GITHUB_TOKEN environment variable with a valid GitHub personal access token.
    Run the script directly: python github_scraper.py

Dependencies:
- PyGithub: For GitHub API interaction
- Standard library: os, json, logging, time, hashlib, itertools

Output:
- scraped_code_snippets.json: JSON file containing unique function snippets with metadata

Rate Limiting:
- Monitors both core and search API rate limits
- Automatically waits when approaching limits to avoid API errors
- Logs rate limit status for monitoring

Error Handling:
- Gracefully handles API errors, missing files, and encoding issues
- Continues processing other repositories/files when individual items fail
- Logs all errors with context for debugging

Author: AI Zetta Team
"""

# Configure logging with timestamp and level formatting for detailed monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def check_rate_limit(g: Github, api_type: str = 'core') -> None:
    """
    Check and manage GitHub API rate limits to prevent API errors.

    This function monitors the remaining API calls for the specified API type
    (core or search) and automatically waits until the rate limit resets if
    the remaining calls drop below a safety threshold. This prevents the
    script from hitting GitHub's rate limits and getting temporarily blocked.

    Args:
        g (Github): Authenticated PyGithub Github instance
        api_type (str): Type of API to check. Either 'core' (for repository
                       operations like getting contents) or 'search' (for
                       repository search operations). Defaults to 'core'.

    Returns:
        None: This function doesn't return a value but may cause the program
              to sleep if rate limits are approached.

    Behavior:
        - Retrieves current rate limit status from GitHub API
        - If remaining calls < 10, calculates wait time until reset
        - Sleeps for the calculated time plus 1 second buffer
        - Logs rate limit status and wait times for monitoring
        - Handles edge cases where reset time might be in the past

    Rate Limit Details:
        - Core API: Used for repository contents, commits, etc. (5000/hour for authenticated users)
        - Search API: Used for repository searches (30/hour for authenticated users)
        - Reset times are provided by GitHub and represent when limits reset
    """
    rate_limit = g.get_rate_limit()
    if api_type == 'core':
        remaining = rate_limit.core.remaining
        reset_time = rate_limit.core.reset
    elif api_type == 'search':
        remaining = rate_limit.search.remaining
        reset_time = rate_limit.search.reset
    else:
        logger.warning(f"Unknown API type: {api_type}")
        return

    if remaining < 10:  # Buffer to avoid hitting limit
        wait_time = reset_time.timestamp() - time.time()
        if wait_time > 0:
            logger.info(f"Rate limit low ({remaining} remaining for {api_type}). Waiting {wait_time:.0f} seconds until reset.")
            time.sleep(wait_time + 1)  # +1 to be safe
        else:
            logger.warning("Rate limit reset time is in the past.")

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
    Main scraping function that orchestrates the entire GitHub code extraction process.

    This function performs a comprehensive search of GitHub repositories, extracts Python
    function definitions from source files, and saves the results to a JSON file. It
    handles rate limiting, error recovery, and data deduplication automatically.

    Process Overview:
        1. Search GitHub for Python repositories sorted by star count
        2. For each repository (up to MAX_REPOS limit):
           - Get repository contents recursively
           - Identify Python files (.py extension)
           - Extract function definitions from each Python file
           - Collect metadata (repository name, file path, star count)
        3. Remove duplicate functions based on MD5 checksums
        4. Save results to JSON file with comprehensive metadata

    Args:
        None: This function doesn't take parameters. It uses global constants
              for configuration (GITHUB_TOKEN, QUERY_LANGUAGE, etc.).

    Returns:
        None: Results are saved to OUTPUT_FILE. Success/failure is logged.

    Data Structure:
        Each extracted function is stored as a dictionary with:
        - repository: Full repository name (owner/repo)
        - file_path: Path to the source file within the repository
        - function_code: Complete function definition as string
        - checksum: MD5 hash for deduplication
        - stars: Repository star count
        - language: Always 'python'

    Rate Limiting:
        - Checks rate limits before each API call
        - Automatically waits when approaching limits
        - Handles both core and search API limits separately

    Error Handling:
        - Continues processing if individual files/repositories fail
        - Logs all errors with context for debugging
        - Gracefully handles API errors, encoding issues, and access problems

    Performance Considerations:
        - Limits repository count to prevent excessive API usage
        - Filters functions by minimum length (50 characters)
        - Uses checksums for efficient deduplication
        - Logs progress for monitoring long-running operations

    Output File:
        Creates 'scraped_code_snippets.json' with indented JSON format.
        Contains array of unique function objects with all metadata.
    """
    all_code_snippets = []
    processed_repos = 0
    
    try:
        # Check rate limit before making the search API call to avoid hitting limits
        check_rate_limit(g, 'search')

        # Perform GitHub repository search using the configured query and sort order
        # This returns a paginated result set that we can iterate over
        logger.info(f"Searching for repositories with query: {QUERY_LANGUAGE}")
        query = g.search_repositories(query=QUERY_LANGUAGE, sort=SORT_ORDER)

        # Process each repository from the search results
        for repo in query:
            # Enforce the maximum repository limit to control processing time and API usage
            if processed_repos >= MAX_REPOS:
                logger.info(f"Reached maximum repository limit: {MAX_REPOS}")
                break
                
            try:
                # Log which repository we're currently processing
                logger.info(f"Processing repository: {repo.full_name}")

                # Check core API rate limit before accessing repository contents
                check_rate_limit(g, 'core')

                # Get the root directory contents of the repository
                # This returns a list of ContentFile objects representing files and directories
                repo_contents = repo.get_contents("")
                logger.info(f"Found {len(repo_contents)} items in repository root")

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
                            check_rate_limit(g, 'core')
                            subdir_contents = repo.get_contents(content.path)
                            # Add Python files from this subdirectory
                            for subcontent in subdir_contents:
                                if subcontent.type == "file" and subcontent.name.endswith(".py"):
                                    files_to_process.append(subcontent)
                        except Exception as e:
                            # Log warning but continue - some subdirs may be inaccessible
                            logger.warning(f"Could not access subdirectory {content.path}: {e}")
                            continue

                logger.info(f"Collected {len(files_to_process)} Python files to process")

                # Process each collected Python file
                for py_file in files_to_process:
                    try:
                        # Decode the file content from bytes to string using UTF-8 encoding
                        # GitHub API returns content as base64-encoded bytes
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
                                    'language': 'python'                 # fixed value
                                }
                                # Add to our collection of all snippets
                                all_code_snippets.append(snippet_data)

                        # Log progress for this file
                        logger.info(f"Extracted {len(functions)} functions from {py_file.path}")

                    except Exception as e:
                        # Log file processing errors but continue with other files
                        logger.warning(f"Error processing file {py_file.path}: {e}")
                        continue

                # Log total functions extracted from this repository
                repo_functions = len([s for s in all_code_snippets if s['repository'] == repo.full_name])
                logger.info(f"Extracted {repo_functions} total function snippets from {repo.full_name}")

                processed_repos += 1
                logger.info(f"Completed repository {processed_repos}/{MAX_REPOS}: {repo.full_name}")
                
            except Exception as e:
                # Handle repository-level errors (API limits, access denied, etc.)
                # Continue with next repository rather than failing completely
                logger.error(f"Error processing repository {repo.full_name}: {e}")
                continue
    
    except Exception as e:
        # Handle search-level errors (network issues, API changes, etc.)
        # Return early since we can't proceed without search capability
        logger.error(f"Error during repository search: {e}")
        return
    
    # Remove duplicate code snippets using checksum-based deduplication
    # This ensures we don't store identical functions from different repositories
    logger.info("Removing duplicate code snippets based on checksum...")
    unique_snippet_data = []
    seen_checksums = set()  # Set for O(1) lookup performance

    # Iterate through all collected snippets and keep only unique ones
    for snippet in all_code_snippets:
        if snippet['checksum'] not in seen_checksums:
            seen_checksums.add(snippet['checksum'])
            unique_snippet_data.append(snippet)

    # Save the final deduplicated data to JSON file
    try:
        # Write to file with proper encoding and formatting
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(unique_snippet_data, f, indent=2, ensure_ascii=False)

        # Log final statistics
        logger.info(f"Successfully saved {len(unique_snippet_data)} unique code snippets to {OUTPUT_FILE}")
        logger.info(f"Total repositories processed: {processed_repos}")
        logger.info(f"Total checksums computed: {len(seen_checksums)}")

    except Exception as e:
        # Handle file I/O errors (permissions, disk space, etc.)
        logger.error(f"Error saving data to file: {e}")

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

        # Report final API usage statistics for monitoring
        rate_limit = g.get_rate_limit()
        logger.info(f"Final rate limits - Core: {rate_limit.core.remaining}/{rate_limit.core.limit}, Search: {rate_limit.search.remaining}/{rate_limit.search.limit}")
    except Exception as e:
        # Catch and log any unexpected errors during execution
        logger.error(f"GitHub scraping failed: {e}")

if __name__ == "__main__":
    main()
