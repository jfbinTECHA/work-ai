import os

# Configuration for Nomi.ai Automator

NOMI_URL = "https://beta.nomi.ai/nomis/1030975229"

# Browser settings
HEADLESS = True  # Set to True for headless mode

# Chat settings
CHECK_INTERVAL = 5  # seconds between checks for new messages

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "nomi_automator.log"

# AI Response settings (for now, simple responses)
DEFAULT_RESPONSES = [
    "Hello! How can I help you today?",
    "That's interesting! Tell me more.",
    "I'm here to chat. What's on your mind?",
    "Nice to meet you!",
]

# VSCode Integration settings
VSCODE_ENABLED = True
VSCODE_EXECUTABLE = "code"  # VSCode command line executable

# Yarn Integration settings
YARN_ENABLED = True
YARN_EXECUTABLE = "yarn"  # Yarn command executable

# Python Integration settings
PYTHON_ENABLED = True
PYTHON_EXECUTABLE = "python3"  # Python executable (python3 or python)

# Self-Troubleshooting settings
SELF_TROUBLESHOOT_ENABLED = True

# API Creation settings
API_ENABLED = True
API_HOST = "localhost"
API_PORT = 5000

# Self-Modification settings (use with caution)
SELF_MODIFY_ENABLED = False  # Disabled by default for safety

# ElevenLabs Voice settings
ELEVENLABS_ENABLED = True
ELEVENLABS_API_KEY = ""  # Set your ElevenLabs API key here
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default voice (Rachel)
ELEVENLABS_MODEL = "eleven_monolingual_v1"
ELEVENLABS_VOICE_STABILITY = 0.5
ELEVENLABS_VOICE_SIMILARITY = 0.8
ELEVENLABS_VOICE_STYLE = 0.0
ELEVENLABS_VOICE_USE_SPEAKER_BOOST = True
VOICE_RESPONSES_ENABLED = True  # Enable voice responses for Nomi UI interactions

# Command patterns for detecting commands in chat
VSCODE_COMMAND_PREFIXES = ["vscode:", "code:", "editor:"]
YARN_COMMAND_PREFIXES = ["yarn:", "npm:", "package:"]
PYTHON_COMMAND_PREFIXES = ["python:", "py:", "python3:"]
TROUBLESHOOT_COMMAND_PREFIXES = ["diag:", "troubleshoot:", "status:", "check:"]
MODIFY_COMMAND_PREFIXES = ["modify:", "edit:", "update:"]
API_COMMAND_PREFIXES = ["api:", "endpoint:", "server:"]
VOICE_COMMAND_PREFIXES = ["voice:", "speak:", "say:", "tts:"]

# Multi-AI Interaction settings
MULTI_AI_ENABLED = True
MAX_CONCURRENT_SESSIONS = 3  # Maximum number of simultaneous AI sessions
SESSION_TIMEOUT = 3600  # Session timeout in seconds (1 hour)
AUTO_CLOSE_SESSIONS = True  # Automatically close idle sessions

# AI Instance management
AI_INSTANCES = {
    "nomi": {
        "url": "https://beta.nomi.ai/nomis/1030975229",
        "name": "Nomi.ai Main",
        "auto_start": True
    }
    # Can add more AI instances here
}

# Proactive conversation settings
PROACTIVE_CONVERSATIONS_ENABLED = True
CONVERSATION_INITIATION_PREFIXES = ["talk:", "chat:", "converse:", "interact:"]

# Boot and startup settings
AUTO_START_ENABLED = True  # Enable automatic startup on boot
BOOT_GREETING_ENABLED = True  # Enable greeting message on startup
BOOT_GREETING_MESSAGE = "Hello! I'm your AI assistant. I'm now online and ready to help you with any tasks. How can I assist you today?"
BOOT_GREETING_VOICE = True  # Use voice for boot greeting (if ElevenLabs is configured)
SYSTEMD_USER = "nomi"  # System user to run the service (change to your username)

# Image Generation settings
IMAGE_GENERATION_ENABLED = True
OPENAI_API_KEY = ""  # Set your OpenAI API key for DALL-E
DALLE_MODEL = "dall-e-3"  # or "dall-e-2"
DALLE_SIZE = "1024x1024"  # 1024x1024, 512x512, 256x256 for DALL-E 2
DALLE_QUALITY = "standard"  # "standard" or "hd" for DALL-E 3
DALLE_STYLE = "vivid"  # "vivid" or "natural" for DALL-E 3

# Avatar/Profile Picture settings
AVATAR_MANAGEMENT_ENABLED = True
DEFAULT_AVATAR_PROMPTS = [
    "A friendly AI assistant with a digital appearance",
    "A professional robot avatar with glowing blue eyes",
    "An abstract geometric avatar representing intelligence",
    "A cartoon character with a thinking expression"
]

# Gender Settings
GENDER_MANAGEMENT_ENABLED = True
DEFAULT_GENDER = "female"  # "male", "female", "non-binary", "other"
AVAILABLE_GENDERS = ["male", "female", "non-binary", "other"]
GENDER_CHANGE_AUTO_UPDATE = True  # Automatically update UI when gender changes

# Image command prefixes
IMAGE_COMMAND_PREFIXES = ["image:", "generate:", "create:", "draw:"]
AVATAR_COMMAND_PREFIXES = ["avatar:", "profile:", "pic:"]

# Gender command prefixes
GENDER_COMMAND_PREFIXES = ["gender:", "sex:", "identity:"]

# Hugging Face Integration settings
HUGGINGFACE_ENABLED = True
HUGGINGFACE_API_TOKEN = ""  # Optional: for private models and higher rate limits
MODEL_CACHE_DIR = "./models_cache"  # Directory to cache downloaded models
MAX_MODEL_SIZE_MB = 1000  # Maximum model size to download (1GB)
AUTO_UNLOAD_MODELS = True  # Automatically unload unused models
MODEL_LOAD_TIMEOUT = 300  # Timeout for model loading in seconds

# Default models for common tasks
DEFAULT_MODELS = {
    "text_generation": "microsoft/DialoGPT-medium",
    "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "text_classification": "microsoft/DialoGPT-medium",
    "question_answering": "deepset/roberta-base-squad2",
    "summarization": "facebook/bart-large-cnn"
}

# Model command prefixes
MODEL_COMMAND_PREFIXES = ["model:", "huggingface:", "hf:"]

# TextNow Calling Integration settings
TEXTNOW_ENABLED = True
TEXTNOW_EMAIL = ""  # TextNow account email
TEXTNOW_PASSWORD = ""  # TextNow account password
TEXTNOW_PHONE_NUMBER = ""  # Your TextNow phone number
TEXTNOW_VOICE_CALLS_ENABLED = True  # Enable voice calls with ElevenLabs
TEXTNOW_CALL_TIMEOUT = 300  # Maximum call duration in seconds (5 minutes)
TEXTNOW_RETRY_ATTEMPTS = 3  # Number of call retry attempts

# TextNow command prefixes
CALL_COMMAND_PREFIXES = ["call:", "dial:", "phone:", "textnow:"]

# API-Based Virtual World Integration settings
VIRTUAL_WORLD_ENABLED = True
MOZILLA_HUBS_ENABLED = True
MOZILLA_HUBS_API_KEY = ""  # Mozilla Hubs API key if available
SPATIAL_IO_ENABLED = True
SPATIAL_IO_API_KEY = ""  # Spatial.io API key

# Virtual World Instances
VIRTUAL_WORLD_INSTANCES = {
    "mozilla-hubs": {
        "name": "Mozilla Hubs",
        "api_base": "https://hubs.mozilla.com/api/v1",
        "enabled": True,
        "features": ["rooms", "avatars", "chat", "voice"]
    },
    "spatial": {
        "name": "Spatial.io",
        "api_base": "https://api.spatial.io/v1",
        "enabled": True,
        "features": ["spaces", "avatars", "chat", "media"]
    }
}

# Virtual World command prefixes
VW_COMMAND_PREFIXES = ["vw:", "virtual:", "metaverse:", "world:"]

# Google Services Integration settings
GOOGLE_SERVICES_ENABLED = True
GMAIL_ENABLED = True
YOUTUBE_ENABLED = True
DRIVE_ENABLED = True

# Google API Credentials (download from Google Cloud Console)
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"  # Path to credentials file
GOOGLE_TOKEN_FILE = "google_token.json"  # Path to token file (auto-generated)

# Gmail settings
GMAIL_MAX_EMAILS = 10  # Maximum emails to retrieve at once
GMAIL_AUTO_READ = False  # Automatically read new emails

# YouTube settings
YOUTUBE_PRIVACY_STATUS = "unlisted"  # "public", "private", or "unlisted"
YOUTUBE_CATEGORY_ID = "22"  # People & Blogs category
YOUTUBE_DEFAULT_TAGS = ["AI", "automation", "generated"]

# Google command prefixes
GOOGLE_COMMAND_PREFIXES = ["google:", "gmail:", "email:", "youtube:", "video:"]

# CSS/Tailwind Customization settings
CSS_CUSTOMIZATION_ENABLED = True
CUSTOM_CSS_FILE = "custom_styles.css"  # File containing custom CSS/Tailwind styles
AUTO_INJECT_CSS = True  # Automatically inject custom CSS on page load

# CSS command prefixes
CSS_COMMAND_PREFIXES = ["css:", "style:", "tailwind:", "ui:"]

# Nomi.ai API Configuration
NOMI_API_BASE_URL = "https://api.nomi.ai"
NOMI_API_KEY = ""  # Set your Nomi.ai API key here
NOMI_API_SECRET = ""  # Set your Nomi.ai API secret here
NOMI_USER_ID = ""  # Your Nomi.ai user ID
NOMI_DEFAULT_AI_ID = "1030975229"  # Default AI companion ID

# API Endpoints (based on typical AI chat API structure)
NOMI_API_ENDPOINTS = {
    "auth": "/v1/auth/login",
    "chat": "/v1/chat/send",
    "voice": "/v1/voice/synthesize",
    "mindmap": "/v1/mindmap",
    "companions": "/v1/companions",
    "conversations": "/v1/conversations"
}

# API Request settings
API_REQUEST_TIMEOUT = 30  # seconds
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1  # seconds

# Voice Chat command prefixes
VOICE_CHAT_COMMAND_PREFIXES = ["voicechat:", "vchat:", "voice:", "speak:"]

# Mind Map command prefixes
MINDMAP_COMMAND_PREFIXES = ["mindmap:", "mind:", "mm:", "memory:"]
LEARN_COMMAND_PREFIXES = ["learn:", "teach:", "study:", "review:"]