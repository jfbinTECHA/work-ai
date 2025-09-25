
import os
import hashlib
import logging
from github import Github

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_TOKEN_ENV_VAR = 'GITHUB_TOKEN'
QUERY_STRING = "language:python"

def get_github_api():
    """Initialize the GitHub API."""
    token = os.environ.get(GITHUB_TOKEN_ENV_VAR)
    if not token:
        raise ValueError(f"{GITHUB_TOKEN_ENV_VAR} environment variable not set")
    return Github(token)

def calculate_checksum(code_snippet: str) -> str:
    """Calculate SHA256 checksum of a string."""
    h = hashlib.sha256()
    h.update(code_snippet.encode('utf-8'))
    return h.hexdigest()

def main():
    g = get_github_api()
    logger.info(f"Searching for repositories matching '{QUERY_STRING}'")
    repos = g.search_repositories(query=QUERY_STRING, sort="stars")

    unique_snippets = set()
    for repo in repos[:5]:  # Limit to first 5 for testing
        logger.info(f"Processing repository: {repo.full_name}")

        # Example dummy snippet
        dummy_snippet = "print('Hello world')"
        checksum = calculate_checksum(dummy_snippet)
        unique_snippets.add(checksum)

        logger.info(f"Snippet checksum: {checksum}")

if __name__ == "__main__":
    main()
