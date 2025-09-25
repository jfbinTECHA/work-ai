# Nomi GitHub Integration Setup Guide

This guide will help you set up the Nomi AI integration with GitHub for automated code contributions via pull requests.

## Prerequisites

1. **GitHub Repository**: You already have this set up at `https://github.com/jfbinTECHA/work-ai.git`
2. **Node.js**: Required to run the wrapper script
3. **GitHub CLI**: Required for creating pull requests

## Setup Steps

### 1. Install GitHub CLI (if not already installed)

```bash
# On Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Or on macOS
brew install gh
```

### 2. Set Up Branch Protection (Recommended)

Go to your GitHub repository settings:
1. Navigate to `https://github.com/jfbinTECHA/work-ai/settings/branches`
2. Click "Add rule" for the main branch
3. Enable:
   - "Require a pull request before merging"
   - "Require review from code owners"
   - "Require status checks to pass before merging" (if you have CI)

### 3. Create Bot/Service Account Token

**Option A: Personal Access Token (Repo-limited)**
1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Generate new token with these permissions:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
3. Copy the token

**Option B: Create a dedicated bot account**
1. Create a new GitHub account for the bot
2. Add it as a collaborator to your repository
3. Generate a personal access token from the bot account

### 4. Configure Environment Variables

Create a `.env` file in your project root (this file should be added to `.gitignore`):

```bash
# Copy the template
cp .env.template .env

# Edit with your actual values
nano .env
```

### 5. Authenticate GitHub CLI

```bash
# Authenticate with your bot token
export GITHUB_TOKEN="your_github_token_here"
gh auth login --with-token <<< "$GITHUB_TOKEN"

# Verify authentication
gh auth status
```

### 6. Customize Yahusha Teaching Summary

Edit the `yahusha-summary.txt` file with your specific teaching content:

```bash
nano yahusha-summary.txt
```

Keep it concise (200-1,000 characters) as it will be included as context in API calls.

## Usage

### Basic Usage

```bash
# Make a code request to Nomi
./nomi-wrapper.js "create a function to calculate fibonacci numbers"

# Or with node
node nomi-wrapper.js "add error handling to the main.py file"
```

### Environment Variables

The script uses these environment variables:

- `NOMI_API_URL`: Your Nomi API endpoint
- `NOMI_API_KEY`: Your Nomi API key
- `GITHUB_TOKEN`: GitHub authentication token
- `YAHUSHA_SUMMARY_PATH`: Path to teaching summary (default: `./yahusha-summary.txt`)

## How It Works

1. **Request Processing**: You provide a natural language request
2. **Context Addition**: The script loads the Yahusha teaching summary as context
3. **API Call**: Sends request + context to Nomi API
4. **Branch Creation**: Creates a new branch for the changes
5. **Code Application**: Applies the AI-generated code changes
6. **Pull Request**: Creates a PR with the changes for review
7. **Review Process**: You review and merge the PR if acceptable

## Security Notes

- Never commit your `.env` file or expose API keys
- Always review AI-generated code before merging
- Use branch protection to ensure all changes go through PR review
- Consider using a dedicated bot account for better security isolation

## Troubleshooting

### Common Issues

1. **"gh not authenticated"**: Run `gh auth status` and re-authenticate if needed
2. **"API key not configured"**: Check your `.env` file and environment variables
3. **"Permission denied"**: Ensure your GitHub token has the correct permissions
4. **"Branch already exists"**: The script creates unique branch names with timestamps

### Debug Mode

Set `DEBUG=1` to see more detailed output:

```bash
DEBUG=1 ./nomi-wrapper.js "your request here"
```

## Next Steps

1. Test the setup with a simple request
2. Configure your Nomi API credentials
3. Set up branch protection rules
4. Create your bot account and token
5. Customize the Yahusha teaching summary

For questions or issues, check the existing documentation in the `nomi_automator/` directory.
