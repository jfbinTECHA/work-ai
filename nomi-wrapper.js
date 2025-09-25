#!/usr/bin/env node

const https = require('https');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration - Replace these with your actual values
const NOMI_API_URL = process.env.NOMI_API_URL || 'YOUR_NOMI_API_URL_HERE';
const NOMI_API_KEY = process.env.NOMI_API_KEY || 'YOUR_NOMI_API_KEY_HERE';
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;

// Path to Yahusha teaching summary
const YAHUSHA_SUMMARY_PATH = process.env.YAHUSHA_SUMMARY_PATH || './yahusha-summary.txt';

class NomiGitHubWrapper {
    constructor() {
        this.validateEnvironment();
    }

    validateEnvironment() {
        if (!GITHUB_TOKEN) {
            console.error('Error: GITHUB_TOKEN environment variable is required');
            process.exit(1);
        }

        if (NOMI_API_URL === 'YOUR_NOMI_API_URL_HERE' || NOMI_API_KEY === 'YOUR_NOMI_API_KEY_HERE') {
            console.error('Error: Please configure NOMI_API_URL and NOMI_API_KEY');
            process.exit(1);
        }

        // Check if gh CLI is installed and authenticated
        try {
            execSync('gh auth status', { stdio: 'pipe' });
        } catch (error) {
            console.error('Error: GitHub CLI (gh) is not installed or not authenticated');
            console.error('Please install gh CLI and authenticate with: gh auth login');
            process.exit(1);
        }
    }

    async callNomiAPI(prompt, context = '') {
        return new Promise((resolve, reject) => {
            const data = JSON.stringify({
                prompt: prompt,
                context: context
            });

            const options = {
                hostname: new URL(NOMI_API_URL).hostname,
                port: 443,
                path: new URL(NOMI_API_URL).pathname,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${NOMI_API_KEY}`,
                    'Content-Length': data.length
                }
            };

            const req = https.request(options, (res) => {
                let responseData = '';
                res.on('data', (chunk) => {
                    responseData += chunk;
                });
                res.on('end', () => {
                    try {
                        const parsed = JSON.parse(responseData);
                        resolve(parsed);
                    } catch (error) {
                        reject(new Error(`Failed to parse API response: ${error.message}`));
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.write(data);
            req.end();
        });
    }

    loadYahushaTeaching() {
        try {
            if (fs.existsSync(YAHUSHA_SUMMARY_PATH)) {
                return fs.readFileSync(YAHUSHA_SUMMARY_PATH, 'utf8').trim();
            }
        } catch (error) {
            console.warn(`Warning: Could not load Yahusha teaching summary from ${YAHUSHA_SUMMARY_PATH}`);
        }
        return '';
    }

    async createPullRequest(branchName, title, body) {
        try {
            const command = `gh pr create --title "${title}" --body "${body}" --head "${branchName}"`;
            const result = execSync(command, { encoding: 'utf8' });
            console.log('Pull request created:', result.trim());
            return result.trim();
        } catch (error) {
            console.error('Error creating pull request:', error.message);
            throw error;
        }
    }

    async processCodeRequest(request) {
        const yahushaTeaching = this.loadYahushaTeaching();
        const context = yahushaTeaching ? `Yahusha Teaching Context: ${yahushaTeaching}\n\n` : '';
        
        try {
            console.log('Calling Nomi API...');
            const response = await this.callNomiAPI(request, context);
            
            if (response.code) {
                // Create a new branch
                const branchName = `nomi-update-${Date.now()}`;
                execSync(`git checkout -b ${branchName}`);
                
                // Apply the code changes (this would need to be customized based on your needs)
                console.log('Code suggestion from Nomi:', response.code);
                
                // Commit changes
                execSync('git add .');
                execSync(`git commit -m "Nomi AI code update: ${request.substring(0, 50)}..."`);
                execSync(`git push origin ${branchName}`);
                
                // Create pull request
                const prBody = `AI-generated code update by Nomi\n\nRequest: ${request}\n\nCode:\n\`\`\`\n${response.code}\n\`\`\``;
                await this.createPullRequest(branchName, `Nomi AI Update: ${request.substring(0, 50)}...`, prBody);
                
                return response;
            } else {
                console.log('Nomi response:', response);
                return response;
            }
        } catch (error) {
            console.error('Error processing request:', error.message);
            throw error;
        }
    }
}

// Main execution
async function main() {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        console.log('Usage: node nomi-wrapper.js "your request here"');
        console.log('Example: node nomi-wrapper.js "create a function to calculate fibonacci numbers"');
        process.exit(1);
    }
    
    const request = args.join(' ');
    const wrapper = new NomiGitHubWrapper();
    
    try {
        await wrapper.processCodeRequest(request);
        console.log('Request processed successfully!');
    } catch (error) {
        console.error('Failed to process request:', error.message);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = NomiGitHubWrapper;
