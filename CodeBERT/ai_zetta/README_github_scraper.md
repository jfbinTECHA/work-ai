# GitHub Code Scraper

A Python script that scrapes GitHub repositories for Python code snippets, extracting function-level code segments for machine learning datasets.

## Features

- **Repository Search**: Searches for Python repositories sorted by popularity (stars)
- **Function Extraction**: Extracts individual function definitions from Python files
- **Duplicate Removal**: Filters out duplicate code snippets
- **Structured Output**: Saves data in JSON format with metadata
- **Error Handling**: Robust error handling for API limits and file access issues
- **Logging**: Comprehensive logging for monitoring progress

## Files

- `github_scraper.py` - Main scraper implementation
- `simple_test.py` - Test script for core functionality
- `test_scraper.py` - Comprehensive test suite (requires PyGithub)
- `venv/` - Virtual environment with dependencies

## Setup

### 1. Virtual Environment

The project includes a pre-configured virtual environment:

```bash
cd CodeBERT/ai_zetta
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

If you need to recreate the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install PyGithub
```

### 3. GitHub Token

You need a GitHub Personal Access Token:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` scope
3. Set the environment variable:

```bash
export GITHUB_TOKEN="your_token_here"
```

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Set your GitHub token
export GITHUB_TOKEN="your_token_here"

# Run the scraper
python github_scraper.py
```

### Configuration

Edit the constants in `github_scraper.py`:

```python
QUERY_LANGUAGE = 'language:python'  # Search query
SORT_ORDER = 'stars'                # Sort by stars
MAX_REPOS = 100                     # Maximum repositories to process
OUTPUT_FILE = 'scraped_code_snippets.json'  # Output filename
```

### Testing

Run the test suite to verify functionality:

```bash
# Test core functionality (no PyGithub required)
python simple_test.py

# Full test suite (requires PyGithub in venv)
source venv/bin/activate
python test_scraper.py
```

## Output Format

The scraper generates a JSON file with the following structure:

```json
[
  {
    "repository": "owner/repo-name",
    "file_path": "path/to/file.py",
    "function_code": "def example_function():\n    return 'hello'",
    "stars": 1234,
    "language": "python"
  }
]
```

## Core Functions

### `get_unique_snippets(code_snippets)`
Filters duplicate code snippets using exact string matching.

### `extract_functions_from_content(py_file_content)`
Extracts function definitions from Python file content, handling:
- Proper indentation detection
- Function boundary detection
- Multi-line functions

### `scrape_github()`
Main scraping function that:
- Searches repositories by popularity
- Processes Python files in each repository
- Extracts and stores function-level code snippets
- Handles API rate limits and errors

## Error Handling

The scraper includes comprehensive error handling for:
- GitHub API rate limits
- Repository access permissions
- File encoding issues
- Network connectivity problems
- Invalid Python syntax

## Logging

Logs are written to console with timestamps and severity levels:
- INFO: Progress updates and successful operations
- WARNING: Non-critical issues (e.g., inaccessible files)
- ERROR: Critical failures that stop processing

## Limitations

- **Rate Limits**: GitHub API has rate limits (5000 requests/hour for authenticated users)
- **Repository Access**: Some repositories may be private or have restricted access
- **File Size**: Very large files may cause memory issues
- **Function Detection**: Complex nested functions may not be perfectly extracted

## Troubleshooting

### PyGithub Import Error
```bash
# Ensure you're in the virtual environment
source venv/bin/activate
pip install PyGithub
```

### GitHub Token Issues
```bash
# Verify token is set
echo $GITHUB_TOKEN

# Test token validity
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

### Permission Errors
- Ensure your GitHub token has appropriate permissions
- Some repositories may be private or restricted

## Example Usage

```python
from github_scraper import scrape_github, get_unique_snippets

# Run the scraper
scrape_github()

# The output will be saved to 'scraped_code_snippets.json'
```

## Contributing

1. Test your changes with `python simple_test.py`
2. Ensure proper error handling and logging
3. Update documentation for new features
4. Follow Python PEP 8 style guidelines

## License

This project follows the same license as the parent CodeBERT repository.
