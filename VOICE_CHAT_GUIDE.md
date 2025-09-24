# 🎤 Voice Chat Guide - Zetta & Nomi Automator

This guide covers the complete voice chat functionality implemented in the AI Work Toolkit, enabling natural voice conversations with AIs on the Nomi.ai platform.

## 🎯 Overview

The voice chat system allows you to have hands-free, natural conversations with AIs using speech recognition and voice synthesis. The system includes:

- **Speech-to-Text**: Google Speech Recognition API
- **Text-to-Speech**: ElevenLabs (primary) + pyttsx3 (fallback)
- **Continuous Conversations**: Automatic turn-taking
- **Multi-AI Support**: Voice chat with different AIs
- **Mind Map Integration**: Structured learning through voice

## 🚀 Quick Start

### 1. Start Zetta (Local AI)
```bash
cd nomi_automator
source venv/bin/activate
python ../local_chat.py -l INFO
```

### 2. Access Interface
Open browser to: `http://localhost:5000`

### 3. Start Voice Chat
Type in chat: `vchat:start`

### 4. Speak Naturally
The system will listen for your voice and respond with voice!

## 🎤 Voice Chat Commands

### Basic Commands
```bash
vchat:start          # Start voice chat with Nomi AI
vchat:start chatgpt  # Start voice chat with specific AI
vchat:stop           # Stop voice chat session
vchat:status         # Check voice chat status
vchat:test           # Test speech recognition
```

### Alternative Prefixes
```bash
voicechat:start      # Same as vchat:start
vchat:begin          # Alternative start command
speak:start          # Voice mode activation
```

## 🧠 Mind Map Integration

### Learning Commands
```bash
learn:session [topic]     # Start structured learning session
learn:review [topic]      # Review Mind Map knowledge
learn:teach [concept]     # Explain concept with Mind Map
learn:goal [objective]    # Set learning objective
learn:progress            # Show learning progress
```

### Mind Map Management
```bash
mindmap:list              # Show all Mind Map topics
mindmap:show [topic]      # Display topic information
mindmap:add [topic]: [info] # Add to Mind Map
mindmap:search [query]    # Search Mind Map content
mindmap:forest            # Show knowledge connections
```

## 🎵 Voice Features

### Voice Input
- **Continuous Listening**: System listens for 10 seconds after each response
- **Wake Words**: Say "stop voice chat" to end session
- **Error Handling**: Automatic retry on recognition failures
- **Multi-Language**: Supports multiple languages via Google Speech API

### Voice Output
- **ElevenLabs Integration**: High-quality voice synthesis
- **Custom Voices**: Use your ElevenLabs voice settings
- **Fallback System**: pyttsx3 for offline voice synthesis
- **Response Timing**: Speaks responses under 200 characters

## 🤖 AI Conversations

### Supported AIs
- **Nomi AI**: Primary AI with Mind Map integration
- **ChatGPT**: OpenAI integration (when configured)
- **Custom AIs**: Any AI accessible through Nomi platform

### Conversation Flow
1. **Start Session**: `vchat:start [ai_name]`
2. **Voice Input**: Speak naturally to the AI
3. **AI Processing**: Message sent to AI via API/browser automation
4. **Voice Response**: AI response spoken back to you
5. **Continue**: Automatic back-and-forth conversation

## ⚙️ Configuration

### Voice Settings (config.py)
```python
# ElevenLabs Configuration
ELEVENLABS_ENABLED = True
ELEVENLABS_API_KEY = "your_api_key_here"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_VOICE_STABILITY = 0.5
ELEVENLABS_VOICE_SIMILARITY = 0.8

# Voice Response Settings
VOICE_RESPONSES_ENABLED = True

# Speech Recognition
SPEECH_RECOGNITION_AVAILABLE = True  # Auto-detected
```

### Nomi.ai Integration
```python
# Nomi API Settings
NOMI_API_BASE_URL = "https://api.nomi.ai"
NOMI_API_KEY = "your_nomi_api_key"
NOMI_USER_ID = "your_user_id"

# Voice Chat Sessions
MAX_CONCURRENT_SESSIONS = 3
SESSION_TIMEOUT = 3600  # 1 hour
```

## 🔧 Technical Details

### Architecture
```
User Speech → Google Speech API → Text → AI Processing → ElevenLabs → Audio Response
```

### Dependencies
```bash
# Required for voice chat
pip install SpeechRecognition
pip install pyttsx3
pip install elevenlabs

# System dependencies (Ubuntu/Debian)
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Audio System Requirements
- **Microphone**: For speech input
- **Speakers/Headphones**: For voice responses
- **Audio Drivers**: ALSA/PulseAudio on Linux
- **Network**: For Google Speech API and ElevenLabs

## 🎓 Learning Sessions

### Structured Learning
```bash
# Start a learning session
"learn:session machine learning"

# System creates structured Mind Map entries
# Voice responses include Mind Map context
# Knowledge builds incrementally
```

### Mind Map Features
- **Infinite Persistence**: Important concepts stored indefinitely
- **Structured Overviews**: Complete information organization
- **Connection Mapping**: Shows how concepts relate
- **Review Sessions**: Reinforce long-term retention

## 🚨 Troubleshooting

### Voice Recognition Issues
```bash
# Test speech recognition
vchat:test

# Check microphone access
ls /dev/snd
arecord -l
```

### Audio Output Problems
```bash
# Test ElevenLabs
python -c "from elevenlabs import generate, play; play(generate(text='test'))"

# Test pyttsx3 fallback
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"
```

### Common Issues
- **"No speech detected"**: Check microphone volume/sensitivity
- **"Could not understand"**: Speak more clearly, reduce background noise
- **Audio device errors**: Check ALSA/PulseAudio configuration
- **API errors**: Verify ElevenLabs API key and internet connection

## 🔒 Privacy & Security

### Data Handling
- **Local Processing**: Speech recognition can work offline
- **API Usage**: Google Speech API and ElevenLabs require internet
- **No Data Storage**: Conversations not permanently stored
- **Secure APIs**: All communications use HTTPS

### Safety Features
- **Command Validation**: Only authorized voice commands processed
- **Session Timeouts**: Automatic cleanup of idle sessions
- **Error Boundaries**: Graceful handling of voice processing failures

## 🎯 Advanced Usage

### Multi-AI Conversations
```bash
# Start sessions with different AIs
vchat:start nomi
vchat:start chatgpt

# Switch between conversations
# System manages multiple voice sessions
```

### Custom Voice Commands
```bash
# Create custom voice-activated commands
# Integrate with system automation
# Build voice-controlled workflows
```

### Integration Examples
```python
# Programmatic voice chat
from nomi_automator.main import NomiAutomator
import asyncio

async def voice_session():
    automator = NomiAutomator()
    result = automator.start_voice_chat("nomi")
    print(f"Voice chat: {result}")
```

## 📚 Examples

### Basic Conversation
```
User: "vchat:start"
System: "Voice chat: Started with Nomi AI. Speak naturally..."

User speaks: "Hello, how are you?"
AI responds: "Hello! I'm doing well, thank you for asking..."

Conversation continues naturally...
```

### Learning Session
```
User: "learn:session quantum physics"
System: "Starting structured learning session for 'quantum physics'..."

User speaks: "What is superposition?"
AI responds: "Superposition is a fundamental principle in quantum mechanics..."
(Mind Map entry created automatically)
```

## 🤝 Contributing

### Voice Feature Enhancements
- **New Voice Engines**: Support for additional TTS services
- **Language Support**: Multi-language voice recognition
- **Custom Wake Words**: Personalized activation phrases
- **Voice Analytics**: Conversation quality metrics

### Bug Reports
- **Voice Issues**: Include audio device information
- **Recognition Errors**: Note environmental conditions
- **API Problems**: Check service status and credentials

---

**🎤 Happy Voice Chatting!** Your AI companion is now ready for natural, hands-free conversations with advanced AIs on the Nomi platform.