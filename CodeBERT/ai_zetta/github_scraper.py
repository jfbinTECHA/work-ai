# Import necessary libraries
import os
import json
import logging
import time
import hashlib
from itertools import chain
from PyGithub import Github
from typing import List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for GitHub token and repository query parameters
GITHUB_TOKEN_ENV_VAR = 'GITHUB_TOKEN'
QUERY_LANGUAGE = 'language:python'
SORT_ORDER = 'stars'
MAX_REPOS = 100  # Limit number of repositories to process
OUTPUT_FILE = 'scraped_code_snippets.json'

# Initialize GitHub object with access token
try:
    g = Github(os.environ[GITHUB_TOKEN_ENV_VAR])
    logger.info("GitHub API initialized successfully")
except KeyError:
    logger.error(f"Environment variable {GITHUB_TOKEN_ENV_VAR} not found")
    raise
except Exception as e:
    logger.error(f"Failed to initialize GitHub API: {e}")
    raise

def check_rate_limit(g: Github, api_type: str = 'core') -> None:
    """
    Check GitHub API rate limits and wait if necessary.
    Args:
        g: GitHub instance
        api_type: 'core' or 'search'
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
    Filter out duplicate code snippets based on exact string matching.
    Args:
        code_snippets (list): List of strings representing code snippets.
    Returns:
        list: List of unique code snippets.
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
    Extract function definitions from Python file content.
    Args:
        py_file_content (str): Content of a Python file
    Returns:
        list: List of function code snippets
    """
    functions = []
    
    # Split Python file into functions
    parts = py_file_content.split("\ndef ")
    
    # Process the first part (might contain a function without leading \n)
    if parts[0].strip().startswith("def "):
        functions.append(parts[0].strip())
    
    # Process remaining functions
    for func in parts[1:]:
        if func.strip():  # Only add non-empty functions
            # Reconstruct the function with proper def keyword
            complete_func = "def " + func.strip()
            
            # Extract only the function definition (stop at next function or class)
            lines = complete_func.split('\n')
            function_lines = []
            indent_level = None
            
            for line in lines:
                if line.strip() == "":
                    function_lines.append(line)
                    continue
                    
                current_indent = len(line) - len(line.lstrip())
                
                # First non-empty line sets the base indent level
                if indent_level is None and line.strip():
                    indent_level = current_indent
                    function_lines.append(line)
                # Continue adding lines that are part of this function
                elif current_indent > indent_level or (current_indent == indent_level and not line.strip().startswith(('def ', 'class ', '@'))):
                    function_lines.append(line)
                # Stop when we hit another function/class at the same level
                elif current_indent <= indent_level and line.strip().startswith(('def ', 'class ')):
                    break
                else:
                    function_lines.append(line)
            
            if function_lines:
                functions.append('\n'.join(function_lines))
    
    return functions

def scrape_github() -> None:
    """
    Scrape GitHub repositories for Python code snippets, extracting function-level code segments.
    Args:
        None
    Returns:
        None
    """
    all_code_snippets = []
    processed_repos = 0
    
    try:
        # Check rate limit before search
        check_rate_limit(g, 'search')

        # Search for Python repositories sorted by popularity
        logger.info(f"Searching for repositories with query: {QUERY_LANGUAGE}")
        query = g.search_repositories(query=QUERY_LANGUAGE, sort=SORT_ORDER)

        # Iterate over found repositories
        for repo in query:
            if processed_repos >= MAX_REPOS:
                logger.info(f"Reached maximum repository limit: {MAX_REPOS}")
                break
                
            try:
                logger.info(f"Processing repository: {repo.full_name}")

                # Check rate limit before getting contents
                check_rate_limit(g, 'core')

                # Get repository contents
                repo_contents = repo.get_contents("")
                logger.info(f"Found {len(repo_contents)} items in repository root")

                # Process files in the repository
                files_to_process = []
                
                # Handle both files and directories in root
                for content in repo_contents:
                    if content.type == "file" and content.name.endswith(".py"):
                        files_to_process.append(content)
                    elif content.type == "dir":
                        # Recursively get Python files from subdirectories (limit depth)
                        try:
                            check_rate_limit(g, 'core')
                            subdir_contents = repo.get_contents(content.path)
                            for subcontent in subdir_contents:
                                if subcontent.type == "file" and subcontent.name.endswith(".py"):
                                    files_to_process.append(subcontent)
                        except Exception as e:
                            logger.warning(f"Could not access subdirectory {content.path}: {e}")
                            continue

                logger.info(f"Collected {len(files_to_process)} Python files to process")

                # Process Python files
                for py_file in files_to_process:
                    try:
                        # Get decoded Python file content
                        py_file_content = py_file.decoded_content.decode("utf-8")
                        
                        # Extract functions from the file
                        functions = extract_functions_from_content(py_file_content)
                        
                        # Add functions to our collection
                        for func in functions:
                            if len(func.strip()) > 50:  # Only include substantial functions
                                checksum = hashlib.md5(func.strip().encode('utf-8')).hexdigest()
                                snippet_data = {
                                    'repository': repo.full_name,
                                    'file_path': py_file.path,
                                    'function_code': func.strip(),
                                    'checksum': checksum,
                                    'stars': repo.stargazers_count,
                                    'language': 'python'
                                }
                                all_code_snippets.append(snippet_data)
                        
                        logger.info(f"Extracted {len(functions)} functions from {py_file.path}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing file {py_file.path}: {e}")
                        continue

                # Log total functions extracted from this repository
                repo_functions = len([s for s in all_code_snippets if s['repository'] == repo.full_name])
                logger.info(f"Extracted {repo_functions} total function snippets from {repo.full_name}")

                processed_repos += 1
                logger.info(f"Completed repository {processed_repos}/{MAX_REPOS}: {repo.full_name}")
                
            except Exception as e:
                logger.error(f"Error processing repository {repo.full_name}: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error during repository search: {e}")
        return
    
    # Remove duplicate snippets based on checksum
    logger.info("Removing duplicate code snippets based on checksum...")
    unique_snippet_data = []
    seen_checksums = set()

    for snippet in all_code_snippets:
        if snippet['checksum'] not in seen_checksums:
            seen_checksums.add(snippet['checksum'])
            unique_snippet_data.append(snippet)
    
    # Save the scraped data
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(unique_snippet_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(unique_snippet_data)} unique code snippets to {OUTPUT_FILE}")
        logger.info(f"Total repositories processed: {processed_repos}")
        logger.info(f"Total checksums computed: {len(seen_checksums)}")
        
    except Exception as e:
        logger.error(f"Error saving data to file: {e}")

def main():
    """
    Main function to run the GitHub scraper.
    """
    logger.info("Starting GitHub code scraper...")
    
    # Check if GitHub token is available
    if GITHUB_TOKEN_ENV_VAR not in os.environ:
        logger.error(f"Please set the {GITHUB_TOKEN_ENV_VAR} environment variable")
        return
    
    try:
        scrape_github()
        logger.info("GitHub scraping completed successfully!")

        # Log final rate limits
        rate_limit = g.get_rate_limit()
        logger.info(f"Final rate limits - Core: {rate_limit.core.remaining}/{rate_limit.core.limit}, Search: {rate_limit.search.remaining}/{rate_limit.search.limit}")
    except Exception as e:
        logger.error(f"GitHub scraping failed: {e}")

if __name__ == "__main__":
    main()
