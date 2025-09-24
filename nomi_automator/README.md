# Nomi.ai Autonomous Chat System

This system automates interactions with the Nomi.ai chat interface at https://beta.nomi.ai/nomis/1030975229.

## Features

- Autonomous operation of Nomi.ai UI
- Automated chat conversations
- Web interface management
- VSCode integration for code editor control
- Yarn package manager command execution
- Python command execution
- Self-troubleshooting and diagnostics
- Self-code modification capabilities (with safety restrictions)
- Dynamic API creation and management
- ElevenLabs text-to-speech voice synthesis
- Multi-AI interaction and session management
- Proactive conversation initiation with other AI systems
- DALL-E image generation and creative artwork
- Avatar/profile picture management and customization
- Hugging Face model integration and dynamic learning
- On-demand ML model loading and inference
- TextNow VoIP calling with voice synthesis
- Automated phone call management and messaging
- API-based virtual world integration
- Mozilla Hubs and Spatial.io support
- Virtual space creation and management
- Google Gmail email integration
- YouTube video uploading and management
- Google Drive file storage and sharing
- Custom CSS/Tailwind styling for UI customization

## Setup

1. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   playwright install
   ```

2. Inspect the UI to identify selectors:
   ```bash
   python inspect_ui.py
   ```
   This will open the browser, take a screenshot, and save page content for analysis. Update the selectors in `main.py` based on the inspection.

3. Run the system:
    ```bash
    python main.py
    ```

4. (Optional) Set up automatic startup on boot:
    ```bash
    # Make setup script executable and run it
    chmod +x setup_service.sh
    sudo ./setup_service.sh
    ```

    This will configure the system to automatically start the AI assistant when Ubuntu boots up, and it will greet you with a personalized message.

## Configuration

Edit `config.py` to adjust settings like URL, check intervals, and responses.

### Boot Greeting and Auto-Startup

The system can be configured to start automatically when Ubuntu boots and greet you with a personalized message.

**Boot Settings:**
```python
AUTO_START_ENABLED = True  # Enable automatic startup
BOOT_GREETING_ENABLED = True  # Enable greeting on startup
BOOT_GREETING_MESSAGE = "Hello! I'm your AI assistant..."  # Custom greeting
BOOT_GREETING_VOICE = True  # Use voice for greeting
SYSTEMD_USER = "your_username"  # System user for service
```

**Systemd Service Management:**
```bash
# Check service status
sudo systemctl status nomi_automator

# Start service manually
sudo systemctl start nomi_automator

# Stop service
sudo systemctl stop nomi_automator

# Restart service
sudo systemctl restart nomi_automator

# View service logs
sudo journalctl -u nomi_automator -f

# Disable auto-start on boot
sudo systemctl disable nomi_automator
```

### VSCode Integration

The system can execute VSCode commands through the chat interface. Commands are prefixed with `vscode:`, `code:`, or `editor:`.

Examples:
- `vscode: --version` - Check VSCode version
- `vscode: .` - Open current directory in VSCode
- `vscode: --new-window` - Open new VSCode window

### Yarn Integration

The system can execute Yarn commands through the chat interface. Commands are prefixed with `yarn:`, `npm:`, or `package:`.

Examples:
- `yarn: install` - Install dependencies
- `yarn: add lodash` - Add a package
- `yarn: run build` - Run build script

### Python Integration

The system can execute Python commands and scripts through the chat interface. Commands are prefixed with `python:`, `py:`, or `python3:`.

Examples:
- `python: --version` - Check Python version
- `python: -c "print('Hello World')"` - Execute inline Python code
- `python: script.py` - Run a Python script
- `python: -m pip install requests` - Install Python packages

### Self-Troubleshooting Integration

The system can diagnose itself and provide system information. Commands are prefixed with `diag:`, `troubleshoot:`, `status:`, or `check:`.

Examples:
- `diag: status` - Get overall system status
- `diag: logs` - Analyze log files
- `diag: memory` - Check memory usage
- `diag: disk` - Check disk usage
- `diag: cpu` - Check CPU usage
- `diag: errors` - Check for recent errors
- `diag: config` - Show current configuration

### Self-Modification Integration

**⚠️ WARNING: Use with extreme caution!**

The system can modify its own code (limited to config.py and main.py for safety). Commands are prefixed with `modify:` or `edit:`.

Format: `modify: filename.py:search_text:replace_text`

Examples:
- `modify: config.py:CHECK_INTERVAL = 5:CHECK_INTERVAL = 10` - Change check interval
- `modify: main.py:Hello:Hi` - Replace text in main.py

**Safety restrictions:**
- Only config.py and main.py can be modified
- Dangerous code patterns are blocked
- Changes are logged for audit

### API Creation Integration

The system can create and manage its own REST API endpoints. Commands are prefixed with `api:`, `endpoint:`, or `server:`.

Examples:
- `api: start` - Start the API server
- `api: stop` - Stop the API server
- `api: status` - Check API server status
- `api: add GET /hello {"message": "Hello World"}` - Add a GET endpoint
- `api: add POST /data` - Add a POST endpoint
- `api: remove /hello` - Remove an endpoint

The API server runs on localhost:5000 by default.

### ElevenLabs Voice Integration

The system can generate natural-sounding speech using ElevenLabs text-to-speech API in two ways:

#### 1. Direct Voice Commands
Commands are prefixed with `voice:`, `speak:`, `say:`, or `tts:`.

Examples:
- `voice: Hello, this is a test message` - Speak custom text
- `voice: say Welcome to the autonomous system` - Alternative syntax
- `voice: test` - Play a test message

#### 2. Voice Responses to Nomi UI
When `VOICE_RESPONSES_ENABLED = True` in `config.py`, the system will automatically speak all its responses to the Nomi.ai chat interface, making conversations audible.

**Setup Requirements:**
1. Sign up for an ElevenLabs account at https://elevenlabs.io
2. Get your API key from the dashboard
3. Set `ELEVENLABS_API_KEY` in `config.py`
4. Set `VOICE_RESPONSES_ENABLED = True` for automatic voice responses

**Voice Configuration:**
- Voice ID: Configurable in `config.py` (default: Rachel)
- Model: eleven_monolingual_v1
- Adjustable parameters: stability, similarity, style, speaker boost
- Voice Responses: Enable/disable automatic speech for Nomi interactions

### Voice Chat Integration

The system now supports full voice conversations with AIs on the Nomi platform. Commands are prefixed with `voicechat:`, `vchat:`, `voice:`, or `speak:`.

#### Voice Chat Commands
- `vchat:start` - Start voice chat with Nomi AI
- `vchat:start chatgpt` - Start voice chat with specific AI
- `vchat:stop` - Stop voice chat session
- `vchat:status` - Check voice chat status
- `vchat:test` - Test speech recognition

#### Voice Chat Features
- **Speech Recognition**: Google Speech API for accurate voice-to-text
- **Continuous Conversations**: Hands-free back-and-forth dialogue
- **Multi-AI Support**: Voice chat with different AIs simultaneously
- **Automatic Responses**: AI replies are spoken using ElevenLabs
- **Session Management**: Automatic session cleanup and timeout handling

#### Voice Chat Setup
1. Install speech recognition: `pip install SpeechRecognition`
2. Configure ElevenLabs API key for voice responses
3. Ensure microphone access for speech input
4. Start voice chat: `vchat:start`

#### Voice Chat Architecture
```
User Speech → Google Speech API → Text → Nomi AI → ElevenLabs → Voice Response
```

**Requirements:**
- Microphone for speech input
- ElevenLabs API key for voice synthesis
- Internet connection for speech APIs
- Audio output device for voice responses

### Multi-AI Interaction Integration

The system can proactively initiate and manage conversations with multiple AI systems simultaneously. Commands are prefixed with `talk:`, `chat:`, `converse:`, or `interact:`.

#### Session Management
- **Open Sessions**: Create new browser instances for different AI interactions
- **Close Sessions**: Clean up idle or completed conversations
- **Session Monitoring**: Track active conversations and their status
- **Auto-cleanup**: Automatically close idle sessions after timeout

#### Proactive Conversation Commands
- `talk: open nomi` - Open a new Nomi.ai session
- `talk: open https://example.com Custom AI Name` - Open custom AI interface
- `talk: close session_id` - Close a specific session
- `talk: list` - Show all active AI sessions
- `talk: send session_id Hello!` - Send message to specific session
- `talk: ask nomi What is the weather?` - Ask question to AI type (creates session if needed)

#### AI Instance Configuration
Configure known AI instances in `config.py`:

```python
AI_INSTANCES = {
    "nomi": {
        "url": "https://beta.nomi.ai/nomis/1030975229",
        "name": "Nomi.ai Main",
        "auto_start": True
    },
    "custom_ai": {
        "url": "https://custom-ai.example.com",
        "name": "Custom AI",
        "auto_start": False
    }
}
```

#### Multi-Session Features
- **Concurrent Sessions**: Up to 3 simultaneous AI conversations
- **Session Isolation**: Each AI runs in its own browser context
- **Activity Tracking**: Monitors session activity and timeouts
- **Resource Management**: Automatic cleanup of idle sessions

### DALL-E Image Generation Integration

The system can generate images using OpenAI's DALL-E models. Commands are prefixed with `image:`, `generate:`, `create:`, or `draw:`.

**Setup Requirements:**
1. Sign up for OpenAI API access at https://platform.openai.com
2. Get your API key from the dashboard
3. Set `OPENAI_API_KEY` in `config.py`

Examples:
- `image: A futuristic cityscape at sunset` - Generate custom image
- `image: test` - Generate a demo landscape image
- `generate: A robot painting a picture` - Create artwork

**Image Configuration:**
- Model: DALL-E 3 (configurable to DALL-E 2)
- Size: 1024x1024 (configurable)
- Quality: Standard or HD
- Style: Vivid or Natural

### Avatar/Profile Picture Management

The system can generate and manage profile pictures/avatars for the Nomi UI. Commands are prefixed with `avatar:`, `profile:`, or `pic:`.

Examples:
- `avatar: generate` - Generate a random avatar from preset prompts
- `avatar: set A cute cartoon robot with glowing eyes` - Create custom avatar
- `avatar: random` - Change to a different random avatar
- `avatar: default` - Reset to default avatar

**Avatar Features:**
- **Automatic UI Integration**: Attempts to update profile picture in Nomi interface
- **Preset Prompts**: Collection of professional avatar descriptions
- **Custom Prompts**: Generate avatars from custom descriptions
- **Square Format**: Optimized for profile picture dimensions

### Hugging Face Model Integration

The system can dynamically load and use machine learning models from Hugging Face for various AI tasks. Commands are prefixed with `model:`, `huggingface:`, `hf:`, or `learn:`.

**Setup Requirements:**
1. Install PyTorch and Transformers (included in requirements.txt)
2. Optional: Get Hugging Face API token for private models
3. Set `HUGGINGFACE_API_TOKEN` in `config.py` for higher rate limits

#### Model Management Commands
- `model: load microsoft/DialoGPT-medium` - Load a specific model
- `model: load cardiffnlp/twitter-roberta-base-sentiment-latest for sentiment-analysis` - Load model for specific task
- `model: unload microsoft/DialoGPT-medium` - Unload a model to free memory
- `model: list` - Show currently loaded models
- `model: cleanup` - Remove unused models

#### Model Usage Commands
- `model: run microsoft/DialoGPT-medium with Hello, how are you?` - Run inference on loaded model
- `model: default text_generation` - Load default model for text generation
- `model: default sentiment_analysis` - Load default model for sentiment analysis

#### Model Discovery
- `model: search sentiment analysis` - Search for models by task
- `model: search bert for text-classification` - Search with specific task filter

**Available Default Models:**
- `text_generation`: microsoft/DialoGPT-medium
- `sentiment_analysis`: cardiffnlp/twitter-roberta-base-sentiment-latest
- `question_answering`: deepset/roberta-base-squad2
- `summarization`: facebook/bart-large-cnn

**Safety Features:**
- **Size Limits**: Maximum model size restrictions (default: 1GB)
- **Timeout Protection**: Model loading timeouts to prevent hanging
- **Memory Management**: Automatic unloading of unused models
- **Rate Limiting**: Built-in delays and error handling

### TextNow VoIP Calling Integration

The system can make phone calls and send text messages through TextNow using web automation. Commands are prefixed with `call:`, `dial:`, `phone:`, or `textnow:`.

**Setup Requirements:**
1. Create a TextNow account at https://www.textnow.com
2. Get your phone number from TextNow
3. Set `TEXTNOW_EMAIL`, `TEXTNOW_PASSWORD`, and `TEXTNOW_PHONE_NUMBER` in `config.py`

#### Calling Commands
- `call: dial +1234567890` - Make a phone call to a number
- `call: dial +1234567890 with Hello, this is an automated call` - Call with voice message
- `call: hangup` - End current call
- `call: status` - Check call status

#### Messaging Commands
- `call: send +1234567890 Hello from AI!` - Send text message

**Voice Call Features:**
- **ElevenLabs Integration**: Uses your configured voice for automated messages
- **Call Connection**: Waits for call to connect before speaking
- **Message Delivery**: Speaks prepared messages during calls

**Safety & Limitations:**
- **Web Automation**: Uses browser automation (may break with UI changes)
- **Manual Authentication**: May require manual login for security
- **Rate Limiting**: Respect TextNow's calling limits
- **Legal Compliance**: Ensure compliance with telecommunications regulations

**Configuration Options:**
- Call timeout limits
- Retry attempts for failed calls
- Voice call enable/disable
- Account credentials management

### API-Based Virtual World Integration

The system integrates with modern virtual world platforms that provide APIs for automation. Currently supports Mozilla Hubs and Spatial.io. Commands are prefixed with `vw:`, `virtual:`, `metaverse:`, or `world:`.

**Setup Requirements:**
1. Create accounts on supported platforms (Mozilla Hubs, Spatial.io)
2. Obtain API keys if required
3. Set `VIRTUAL_WORLD_ENABLED = True` in `config.py`
4. Configure platform-specific settings

#### Supported Platforms
- **Mozilla Hubs**: Web-based social VR with API support
- **Spatial.io**: Professional virtual spaces with REST API

#### Virtual World Commands
- `vw: list` - Show available virtual world platforms
- `vw: join mozilla-hubs https://hubs.mozilla.com/room-url` - Join a Mozilla Hubs room
- `vw: join spatial space_id` - Join a Spatial.io space
- `vw: create mozilla-hubs My Room` - Create a new room/space
- `vw: chat Hello everyone!` - Send message in current space
- `vw: invite username` - Invite user to current space
- `vw: media https://example.com/image.jpg` - Share media in space
- `vw: leave` - Exit current virtual space
- `vw: status` - Check virtual world session status

#### Features
- **Multi-Platform Support**: Works with different virtual world APIs
- **Real-time Communication**: Text chat and media sharing
- **Space Management**: Create and join virtual spaces
- **User Invitations**: Invite others to join spaces
- **Session Tracking**: Monitor active virtual world sessions

**Platform-Specific Capabilities:**
- **Mozilla Hubs**: Room creation, avatar customization, media sharing
- **Spatial.io**: Professional spaces, presentations, collaboration tools

**Advantages over Complex Platforms:**
- **API-First Design**: Built for automation and integration
- **Web-Based**: No complex 3D client installation required
- **Standard Protocols**: REST APIs and webhooks
- **Developer Friendly**: Extensive documentation and SDKs

### Google Services Integration

The system integrates with Google services including Gmail and YouTube. Commands are prefixed with `google:`, `gmail:`, `email:`, `youtube:`, or `video:`.

**Setup Requirements:**
1. Create a Google Cloud Project at https://console.cloud.google.com
2. Enable Gmail API and YouTube Data API v3
3. Create OAuth 2.0 credentials (download JSON file as `google_credentials.json`)
4. Set `GOOGLE_SERVICES_ENABLED = True` in `config.py`

#### Gmail Integration
- `gmail: send recipient@example.com Subject: Hello there!` - Send email
- `gmail: inbox` - Read recent emails from inbox
- `gmail: emails` - Check for new emails

#### YouTube Integration
- `youtube: upload /path/to/video.mp4 Title: My Awesome Video` - Upload video
- `youtube: videos` - List your uploaded videos
- `youtube: search AI tutorials` - Search YouTube videos

**Authentication:**
- First run will open browser for OAuth authentication
- Token is saved as `google_token.json` for future use
- Supports both Gmail and YouTube API scopes

**Features:**
- **Email Sending**: Send emails programmatically with full formatting
- **Inbox Reading**: Access recent emails with sender/subject information
- **Video Upload**: Upload videos with custom titles, descriptions, and privacy settings
- **Video Search**: Search YouTube content and get results
- **Privacy Control**: Configurable video privacy (public/private/unlisted)

#### Google Drive Integration
- `google: upload /path/to/file.txt Documents` - Upload files to Drive folders
- `google: download file_id /local/path/file.txt` - Download files from Drive
- `google: files` - List recent files in Drive
- `google: share file_id user@example.com writer` - Share files with permissions
- `google: create folder My Folder` - Create new folders

**Drive Features:**
- **File Upload**: Upload files with resumable transfers
- **File Download**: Download files by ID to local paths
- **File Sharing**: Share files with specific users and permission levels
- **Folder Management**: Create and organize folders
- **File Listing**: Browse recent files and folders

### Custom CSS/Tailwind Styling Integration

The system can inject custom CSS and Tailwind styles into the Nomi.ai web interface to customize its appearance. Commands are prefixed with `css:`, `style:`, `tailwind:`, or `ui:`.

**Setup Requirements:**
1. Create a `custom_styles.css` file in the project directory
2. Add your custom CSS/Tailwind rules to the file
3. Set `CSS_CUSTOMIZATION_ENABLED = True` in `config.py`
4. Optionally enable `AUTO_INJECT_CSS = True` for automatic injection on startup

#### CSS Commands
- `css: inject` - Inject custom CSS from file into the current page
- `css: add .my-class { color: red; background: blue; }` - Add CSS rule dynamically
- `css: remove .my-class` - Hide elements matching the selector
- `css: clear` - Remove all custom CSS from the page
- `css: status` - Show current CSS customization status
- `css: file styles.css` - Load CSS from a specific file

#### Example Custom Styles
Create a `custom_styles.css` file with your customizations:

```css
/* Change background gradient */
body {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Style chat messages */
.message-bubble, [class*="message"] {
  border-radius: 12px !important;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

/* Custom button styling */
button, [role="button"] {
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
}

button:hover, [role="button"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
}

/* Input field enhancements */
input, textarea {
  border-radius: 8px !important;
  border: 2px solid #e2e8f0 !important;
  transition: border-color 0.2s ease !important;
}

input:focus, textarea:focus {
  border-color: #667eea !important;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}
```

#### Features
- **File-Based Styling**: Load styles from external CSS files
- **Dynamic Injection**: Add CSS rules on-the-fly during runtime
- **Auto-Injection**: Automatically apply custom styles when the page loads
- **Safe Operations**: Non-destructive CSS injection using Playwright
- **Tailwind Support**: Full support for Tailwind CSS utility classes

#### Configuration Options
```python
CSS_CUSTOMIZATION_ENABLED = True  # Enable CSS customization
CUSTOM_CSS_FILE = "custom_styles.css"  # Path to custom CSS file
AUTO_INJECT_CSS = True  # Auto-inject CSS on page load
```

## Notes

- The system uses placeholder selectors for chat elements. Run `inspect_ui.py` to identify the correct selectors for your specific Nomi.ai instance.
- If authentication is required, the system will prompt for manual login.
- Responses are currently simple random phrases. Enhance `generate_response` for more intelligent replies.