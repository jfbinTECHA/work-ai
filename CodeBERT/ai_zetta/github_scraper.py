import os
import hashlib
import logging
from github import Github

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GITHUB_TOKEN_ENV_VAR = 'GITHUB_TOKEN'
QUERY_STRING = "language:python"

def get_github_api():
    """Initialize the GitHub API."""
    token = os.environ.get(GITHUB_TOKEN_ENV_VAR)
    if not token:
        raise ValueError(f"Environment variable {GITHUB_TOKEN_ENV_VAR} not set.")
    g = Github(token)
    return g

def search_repos(g, query):
    """Search repositories matching the given query."""
    return g.search_repositories(query=query, sort="stars")

def calculate_checksum(code_snippet):
    """Calculate the SHA256 checksum of the given code snippet."""
    h = hashlib.sha256()
    h.update(code_snippet.encode('utf-8'))
    return h.hexdigest()

def main():
    # Initialize GitHub API
    g = get_github_api()

    # Search repositories
    logger.info(f"Searching for repositories matching '{QUERY_STRING}'")
    repos = search_repos(g, QUERY_STRING)

    # Process repositories
    unique_snippets = set()
    for repo in repos[:5]:  # Limit to first 5 repos for demo
        logger.info(f"Processing repository: {repo.full_name}")

        # Dummy snippet (replace with real extraction later)
        dummy_snippet = 'print("Hello, world!")'
        checksum = calculate_checksum(dummy_snippet)
        if checksum not in unique_snippets:
            unique_snippets.add(checksum)
            logger.info(f"New snippet found in {repo.full_name}: {dummy_snippet}")
            logger.info(f"Checksum: {checksum}")

if __name__ == "__main__":
    main()
