# Helper function to filter unique code snippets
def get_unique_snippets(snippets):
    unique_snippets = []
    seen = set()
    for snippet in snippets:
        # Use the complete_snippet as the uniqueness key
        key = snippet["complete_snippet"]
        if key not in seen:
            seen.add(key)
            unique_snippets.append(snippet)
    return unique_snippets
"""
GitHub Scraper for Python Code Snippets using PyGithub
- Authenticates with GitHub
- Searches for popular Python repositories
- Clones or downloads files
- Extracts Python code snippets from .py files
- Saves results to a JSON file (schema: language, incomplete_snippet, complete_snippet)

Note: You need a GitHub personal access token for higher rate limits.
"""

from github import Github
import requests
import os
import json

# Set your GitHub personal access token here (or use environment variable)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
g = Github(GITHUB_TOKEN)

# Parameters
SEARCH_QUERY = 'language:python stars:>1000'
MAX_REPOS = 3  # Limit for demo; increase as needed
OUTPUT_FILE = 'scraped_python_snippets.json'

results = []

# Search for popular Python repositories
for repo in g.search_repositories(query=SEARCH_QUERY)[:MAX_REPOS]:
    print(f"Scraping repo: {repo.full_name}")
    try:
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            elif file_content.path.endswith('.py'):
                # Download raw file content
                raw_url = file_content.download_url
                code = requests.get(raw_url).text
                # For demo: treat the whole file as a complete snippet
                # In practice, parse for incomplete/complete pairs
                results.append({
                    "language": "python",
                    "incomplete_snippet": code.split('\n')[0],  # First line as placeholder
                    "complete_snippet": code
                })
    except Exception as e:
        print(f"Error scraping {repo.full_name}: {e}")

# Filter for unique code snippets before saving
unique_results = get_unique_snippets(results)

# Save unique results
with open(OUTPUT_FILE, 'w') as f:
    json.dump(unique_results, f, indent=2)

print(f"Scraping complete. Saved {len(unique_results)} unique snippets to {OUTPUT_FILE}.")
