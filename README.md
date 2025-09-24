# AI Work Toolkit

A comprehensive collection of AI tools, models, and automation systems for code intelligence, data collection, and autonomous AI interactions.

## Components

### 🤖 CodeBERT Series Models
Pre-trained models for programming language understanding from Microsoft Research:
- **CodeBERT** (EMNLP 2020) - Multi-lingual code representation learning
- **GraphCodeBERT** (ICLR 2021) - Code representations with data flow
- **UniXcoder** (ACL 2022) - Unified cross-modal pre-training for code
- **CodeReviewer** (ESEC/FSE 2022) - Pre-training for code review activities
- **CodeExecutor** (ACL 2023) - Code execution with pre-trained language models
- **LongCoder** (ICML 2023) - Long-range pre-trained language model for code

### 🔍 AI-Zetta GitHub Scraper
Automated system for collecting Python code snippets from GitHub repositories:
- Searches repositories by popularity and language
- Extracts function-level code segments
- Filters duplicates and generates structured datasets
- Perfect for building machine learning datasets for code intelligence

### 🤖 Nomi Automator
Autonomous AI chat system with extensive integrations:
- Automated interactions with Nomi.ai chat interface
- Multi-AI conversation management
- Voice synthesis with ElevenLabs
- Image generation with DALL-E
- Virtual world integration (Mozilla Hubs, Spatial.io)
- Google services (Gmail, YouTube, Drive)
- VoIP calling with TextNow
- Dynamic ML model loading from Hugging Face
- Self-modification and API creation capabilities

## Quick Start

### CodeBERT Models

**Installation:**
```bash
pip install torch transformers
```

**Basic Usage:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Example: Get embeddings for code
code = "def hello_world(): print('Hello, World!')"
tokens = tokenizer.tokenize(code)
print(tokens)
```

For detailed usage of each CodeBERT model, see their respective folders:
- [CodeBERT](CodeBERT/) - Base model for code understanding
- [GraphCodeBERT](GraphCodeBERT/) - Code with data flow
- [UniXcoder](UniXcoder/) - Cross-modal code representation
- [CodeReviewer](CodeReviewer/) - Code review automation
- [CodeExecutor](CodeExecutor/) - Code execution prediction
- [LongCoder](LongCoder/) - Long code modeling

### AI-Zetta GitHub Scraper

**Setup:**
```bash
cd CodeBERT/ai_zetta
python3 -m venv venv
source venv/bin/activate
pip install PyGithub
export GITHUB_TOKEN="your_github_token"
```

**Usage:**
```bash
python github_scraper.py
```

This will scrape Python repositories and extract function-level code snippets for ML training datasets.

### Nomi Automator

**Setup:**
```bash
cd nomi_automator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
```

**Basic Usage:**
```bash
python main.py
```

For advanced features like voice synthesis, API integrations, and autonomous operation, see the [nomi_automator README](nomi_automator/README.md).
## Project Structure

```
├── CodeBERT/           # Original Microsoft CodeBERT models
│   ├── ai_zetta/       # GitHub code scraper for ML datasets
│   ├── code2nl/        # Code to natural language
│   ├── codesearch/     # Code search functionality
│   └── ...
├── CodeExecutor/       # Code execution prediction
├── CodeReviewer/       # Code review automation
├── GraphCodeBERT/      # Graph-based code representations
├── LongCoder/          # Long code modeling
├── UniXcoder/          # Cross-modal code models
└── nomi_automator/     # Autonomous AI chat system
```

## Contributing

This repository is a fork of [Microsoft CodeBERT](https://github.com/microsoft/CodeBERT) with additional AI tools and automation systems.

- **CodeBERT Models**: Follow the original Microsoft contribution guidelines
- **AI-Zetta & Nomi Automator**: Open to contributions, improvements, and new features
- **Bug Reports**: Please file issues with detailed reproduction steps
- **Feature Requests**: Open issues describing the desired functionality

## License

This project maintains the same license as the original Microsoft CodeBERT repository. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Microsoft Research** for the original CodeBERT series models
- **Hugging Face** for the transformers library
- **GitHub** for the API and platform
- All contributors to the open-source AI community
