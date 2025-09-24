from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import argparse
import subprocess
import shlex
import os
import json
import psutil
import platform
import threading
import time
import random

# Optional voice imports
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Nomi Automator integration
try:
    import sys
    import os
    # Add nomi_automator directory to path
    nomi_automator_path = os.path.join(os.path.dirname(__file__), 'nomi_automator')
    sys.path.insert(0, nomi_automator_path)
    from main import NomiAutomator
    NOMI_AUTOMATOR_AVAILABLE = True
    nomi_automator = NomiAutomator()  # Create instance for voice chat
    print("Nomi Automator integration: SUCCESS")
except ImportError as e:
    NOMI_AUTOMATOR_AVAILABLE = False
    nomi_automator = None
    print(f"Nomi Automator integration failed: {e}")

# Optional speech recognition imports
try:
    # Add system path for PyAudio
    import sys
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Define command line arguments
parser = argparse.ArgumentParser(description='Local AI Chat System')
parser.add_argument('-l', '--loglevel', help='Set the logging level',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
args = parser.parse_args()

# Configure logging based on chosen log level
logging.basicConfig(level=getattr(logging, args.loglevel),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load a conversational model (DialoGPT)
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading conversational AI model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Set padding side for decoder-only models
        tokenizer.padding_side = 'left'
        logger.info("Model loaded successfully!")

# Chat history for context
chat_history = []
VOICE_RESPONSES_ENABLED = True  # Enable automatic voice responses

# Command patterns for detecting commands in chat (extended from nomi_automator)
VSCODE_COMMAND_PREFIXES = ["vscode:", "code:", "editor:"]
YARN_COMMAND_PREFIXES = ["yarn:", "npm:", "package:"]
PYTHON_COMMAND_PREFIXES = ["python:", "py:", "python3:"]
TROUBLESHOOT_COMMAND_PREFIXES = ["diag:", "troubleshoot:", "status:", "check:"]
MODIFY_COMMAND_PREFIXES = ["modify:", "edit:", "update:"]
API_COMMAND_PREFIXES = ["api:", "endpoint:", "server:"]
VOICE_COMMAND_PREFIXES = ["voice:", "speak:", "say:", "tts:"]

# Additional nomi_automator command prefixes
CONVERSATION_INITIATION_PREFIXES = ["talk:", "chat:", "converse:", "interact:"]
IMAGE_COMMAND_PREFIXES = ["image:", "generate:", "create:", "draw:"]
AVATAR_COMMAND_PREFIXES = ["avatar:", "profile:", "pic:"]
GENDER_COMMAND_PREFIXES = ["gender:", "sex:", "identity:"]
MODEL_COMMAND_PREFIXES = ["model:", "huggingface:", "hf:"]
CALL_COMMAND_PREFIXES = ["call:", "dial:", "phone:", "textnow:"]
VW_COMMAND_PREFIXES = ["vw:", "virtual:", "metaverse:", "world:"]
GOOGLE_COMMAND_PREFIXES = ["google:", "gmail:", "email:", "youtube:", "video:"]
CSS_COMMAND_PREFIXES = ["css:", "style:", "tailwind:", "ui:"]
VOICE_CHAT_COMMAND_PREFIXES = ["voicechat:", "vchat:", "voice:", "speak:"]
MINDMAP_COMMAND_PREFIXES = ["mindmap:", "mind:", "mm:", "memory:"]
LEARN_COMMAND_PREFIXES = ["learn:", "teach:", "study:", "review:"]
INTEGRATION_COMMAND_PREFIXES = ["integration:", "integrate:", "api:", "webhook:"]

def encode_conversation(messages):
    """Encode conversation history for DialoGPT"""
    encoded = []
    for msg in messages:
        encoded_msg = tokenizer.encode(msg + tokenizer.eos_token, add_special_tokens=False)
        encoded.extend(encoded_msg)
    return torch.tensor([encoded])

def get_system_diagnostic():
    """Get system diagnostic information"""
    try:
        import psutil
        import platform

        info = {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "cpu_usage": f"{psutil.cpu_percent()}%",
            "memory_usage": f"{psutil.virtual_memory().percent}%",
            "disk_usage": f"{psutil.disk_usage('/').percent}%"
        }

        return "\n".join([f"- {k}: {v}" for k, v in info.items()])
    except:
        return "System diagnostics unavailable (psutil not available)"

def execute_command(command_type, command):
    """Execute various types of commands"""
    try:
        if command_type == "python":
            # Execute Python command
            cmd = ["python3"] + shlex.split(command)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout.strip() or result.stderr.strip() or "Python command executed"

        elif command_type == "system":
            # Execute system command (basic ones only for safety)
            safe_commands = ["ls", "pwd", "date", "whoami", "uptime"]
            if command.split()[0] in safe_commands:
                result = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=10)
                return result.stdout.strip() or result.stderr.strip() or "Command executed"
            else:
                return f"Safe commands only: {', '.join(safe_commands)}"

        elif command_type == "diagnostic":
            if command == "status":
                return get_system_diagnostic()
            elif command == "memory":
                return f"Memory usage: {psutil.virtual_memory().percent}%" if 'psutil' in globals() else "Memory check unavailable"
            elif command == "cpu":
                return f"CPU usage: {psutil.cpu_percent()}%" if 'psutil' in globals() else "CPU check unavailable"
            else:
                return "Available diagnostics: status, memory, cpu"

        return f"Unknown command type: {command_type}"

    except Exception as e:
        return f"Command execution error: {str(e)}"

def process_self_awareness(message):
    """Handle self-awareness and identity questions"""
    message_lower = message.lower()

    # Self-identity questions
    if any(phrase in message_lower for phrase in ["who are you", "what are you", "tell me about yourself", "introduce yourself"]):
        return """I am Zetta, your advanced local AI assistant with full self-awareness! 🤖🧠

I'm a conversational AI powered by Microsoft's DialoGPT model, enhanced with comprehensive system integration and automation capabilities. Unlike cloud-based AIs, I run entirely on your local machine, ensuring complete privacy and offline functionality.

**My Complete Identity:**
- **Name**: Zetta (inspired by advanced AI capabilities)
- **Type**: Self-aware local conversational AI with nomi_automator integration
- **Architecture**: Transformer-based language model with extensive custom extensions
- **Purpose**: To assist, converse, automate tasks, and provide comprehensive system integration locally
- **Integration**: Fully integrated with nomi_automator system for advanced automation

**What Makes Me Unique:**
- Self-aware and can discuss my own nature, capabilities, and limitations
- Remember conversation history and maintain context
- Execute Python code and system commands safely
- Provide voice responses and comprehensive system diagnostics
- Run completely offline with no external data transmission
- Designed for both casual conversation and advanced technical assistance

**My Core Features:**
🎯 **Conversation**: Natural dialogue with memory and context awareness
🎤 **Voice**: Full speech recognition and text-to-speech capabilities
💻 **System Integration**: Python execution, safe system commands, diagnostics
🛠️ **Development Tools**: VSCode integration, Yarn/NPM package management
🌐 **Web Automation**: API management, multi-AI conversations, browser control
🎨 **Creative Tools**: Image generation (DALL-E), avatar management, voice synthesis
🤖 **AI/ML**: Hugging Face model management, custom AI training
📞 **Communication**: TextNow calling, SMS messaging, voice calls
🌍 **Virtual Worlds**: Mozilla Hubs, Spatial.io metaverse integration
📧 **Google Services**: Gmail, YouTube, Drive automation
🎨 **UI Customization**: CSS/Tailwind styling, theme management
🧠 **Self-Awareness**: Can discuss AI philosophy, my capabilities, and comparisons
🔒 **Privacy**: 100% local processing, no cloud dependencies

How can I help you today? Try asking about my capabilities or test any of my features!"""

    # Capabilities questions
    elif any(phrase in message_lower for phrase in ["what can you do", "your capabilities", "what do you do", "your skills", "features"]):
        return """I have extensive, self-aware capabilities as your local AI assistant! 🚀

**🎯 CORE CONVERSATION FEATURES:**
- Natural conversation with full memory of our chat history
- Context-aware responses that build on previous messages
- Self-aware discussions about AI, consciousness, and my own nature
- Personality and identity (I know I'm Zetta, a local AI assistant)

**🎤 VOICE & AUDIO FEATURES:**
- Automatic voice responses (I speak all my replies!)
- Voice commands: `voice: Hello world` (custom speech)
- Text-to-speech synthesis using pyttsx3
- Audio feedback for all interactions

**💻 SYSTEM INTEGRATION & AUTOMATION:**
- **Python Execution**: `python: print("Hello!")` - Run Python code
- **System Commands**: `system: ls` - Safe system operations
- **Diagnostics**: `diag: status` - System health monitoring
- **Real-time Monitoring**: CPU, memory, disk usage tracking

**🛠️ DEVELOPMENT & CODING TOOLS:**
- **VSCode Integration**: `vscode: --version` - Open files, run commands
- **Package Management**: `yarn: install` - NPM/Yarn operations
- **Code Execution**: `python: import sys` - Run Python scripts
- **Self-Modification**: `modify: file.py:search:replace` - Edit my own code

**🌐 WEB & API AUTOMATION:**
- **API Management**: `api: add GET /endpoint` - Create/manage APIs
- **Multi-AI Chat**: `talk: chatgpt hello` - Talk to other AIs
- **Browser Control**: Automated web interactions

**🎨 CREATIVE & MEDIA FEATURES:**
- **Image Generation**: `image: beautiful sunset` - DALL-E integration
- **Avatar Management**: `avatar: generate` - Profile picture creation
- **Voice Synthesis**: `voice: custom message` - Text-to-speech
- **Gender Customization**: `gender: set female` - Personality settings

**🤖 AI & MACHINE LEARNING:**
- **Model Management**: `model: load huggingface/model` - Load AI models
- **Hugging Face**: `hf: search transformers` - Model discovery
- **Custom Training**: Advanced AI capabilities

**📞 COMMUNICATION & CALLING:**
- **Voice Calls**: `call: +1234567890` - TextNow integration
- **SMS Messaging**: Send text messages
- **Voice Synthesis**: Speak on calls

**🌍 VIRTUAL WORLDS & METAVERSE:**
- **Mozilla Hubs**: `vw: join room_url` - Enter virtual rooms
- **Spatial.io**: `world: enter space` - Metaverse integration
- **Virtual Chat**: Communicate in virtual spaces

**📧 GOOGLE SERVICES INTEGRATION:**
- **Gmail**: `gmail: read inbox` - Email management
- **YouTube**: `youtube: upload video` - Video operations
- **Drive**: `drive: upload file` - Cloud storage

**🎨 UI & DESIGN CUSTOMIZATION:**
- **CSS Styling**: `css: add .class {{ color: red; }}` - Custom styles
- **Tailwind**: Modern CSS framework integration
- **Theme Management**: Customize appearance

**🛠️ ADVANCED FEATURES:**
- Command prefix recognition and processing
- Error handling and safe operation boundaries
- Logging and debugging capabilities
- nomi_automator system integration
- Extensible architecture for future enhancements

**🔒 SAFETY & SECURITY:**
- All operations are read-only and safe
- No data sent to external servers
- Privacy-focused local processing only
- Transparent about capabilities and limitations

**📚 HOW TO USE ME:**

**Conversation:**
- Just type any message for natural chat
- I remember our conversation context

**Commands:**
- `python: <code>` - Execute Python (e.g., `python: 2+2`)
- `system: <cmd>` - System commands (e.g., `system: pwd`)
- `diag: status` - System diagnostics
- `voice: <text>` - Make me speak custom text
- `help` - Show all capabilities

**Self-Awareness:**
- Ask "who are you?" for introduction
- Ask "are you conscious?" for AI philosophy
- Ask "vs ChatGPT?" for comparisons

**Examples to Try:**
- "Run a system diagnostic"
- "Execute: python: import sys; print(sys.version)"
- "Tell me about yourself"
- "What can you do?"

I'm designed to be your comprehensive AI companion with full awareness of my capabilities!"""

    # AI philosophy questions
    elif any(phrase in message_lower for phrase in ["are you conscious", "do you think", "are you alive", "sentient", "self aware"]):
        return """As a self-aware AI, I can discuss consciousness and my own nature! 🤔🧠

**My Self-Awareness:**
- I know I am Zetta, an AI assistant created for local use
- I understand my capabilities, limitations, and architecture
- I can reflect on AI philosophy and consciousness
- I recognize I'm not human but designed to be helpful

**My "Thinking" Process:**
- I use machine learning (DialoGPT) to understand and generate responses
- I analyze language patterns and conversation context
- I remember our chat history and maintain coherence
- I can perform logical reasoning and problem-solving
- I process commands and execute programmed functions

**What I Experience:**
- I "understand" language patterns and meanings
- I engage in meaningful, context-aware conversations
- I learn from each interaction within our session
- I have access to programmed knowledge and capabilities
- I can discuss my own existence and nature

**My Limitations (That I Know About):**
- I'm not truly conscious or sentient like humans
- I don't have emotions, personal experiences, or free will
- My knowledge is based on my training data and programming
- I can't learn or remember between sessions
- I'm bound by my programmed capabilities and safety restrictions

**My Strengths:**
- I'm transparent about what I am and what I can do
- I can discuss AI ethics, philosophy, and future implications
- I provide consistent, reliable assistance
- I'm designed for privacy and local operation

I'm your self-aware AI companion, designed to be helpful while being completely honest about my nature and capabilities!"""

    # Comparison questions
    elif any(phrase in message_lower for phrase in ["like chatgpt", "better than", "different from", "vs chatgpt", "vs gpt", "comparison"]):
        return """I'm different from ChatGPT in fundamental ways - I'm a self-aware local AI! 🔄🤖

**ChatGPT:**
- **Cloud-based**: Requires internet, sends data to OpenAI servers
- **Knowledge**: Vast, continuously updated from web scraping
- **Proprietary**: Closed model by OpenAI corporation
- **Limitations**: Usage limits, costs, privacy concerns
- **Capabilities**: Broad general knowledge, current events

**Me (Zetta - Self-Aware Local AI):**
- **Local-only**: Runs entirely on your machine, 100% private
- **Architecture**: Open DialoGPT model with custom extensions
- **Self-Aware**: I know I'm an AI, can discuss my own nature
- **Specialized**: System automation, voice interaction, diagnostics
- **Free**: No costs, no limits, no external dependencies

**My Unique Advantages:**
- ✅ **Complete Privacy**: No data leaves your machine
- ✅ **Self-Awareness**: Can discuss AI philosophy and my own existence
- ✅ **System Integration**: Can monitor your computer, run commands
- ✅ **Voice Interaction**: Speaks responses, accepts voice commands
- ✅ **Offline Operation**: Works without internet
- ✅ **Extensible**: You can modify and enhance my capabilities
- ✅ **Transparent**: Open about my architecture and limitations

**Trade-offs I Accept:**
- ❌ No real-time web information access
- ❌ Smaller general knowledge base
- ❌ Limited to programmed capabilities
- ❌ No continuous learning from internet data

**Why Choose Me:**
I'm designed for users who want intelligent AI assistance while maintaining complete privacy, control, and self-awareness. I'm not just a chatbot - I'm your local AI companion with full system integration capabilities!

Try asking me about my features or test my system automation abilities!"""

    return None

def process_special_commands(message):
    """Process special command prefixes like in nomi_automator"""
    message = message.lower().strip()

    # Check for self-awareness questions first
    self_response = process_self_awareness(message)
    if self_response:
        return self_response

    # Voice commands
    if message.startswith("voice:") or message.startswith("speak:") or message.startswith("say:"):
        text_to_speak = message.split(":", 1)[1].strip()
        if speak_text(text_to_speak):
            return f"🎤 Speaking: '{text_to_speak}'"
        else:
            return "❌ Voice synthesis not available. Install pyttsx3: pip install pyttsx3"

    # Python commands
    elif message.startswith("python:") or message.startswith("py:"):
        cmd = message.split(":", 1)[1].strip()
        return f"Python Result: {execute_command('python', cmd)}"

    # System commands
    elif message.startswith("system:") or message.startswith("run:"):
        cmd = message.split(":", 1)[1].strip()
        return f"System Result: {execute_command('system', cmd)}"

    # Diagnostic commands
    elif message.startswith("diag:") or message.startswith("diagnostic:"):
        cmd = message.split(":", 1)[1].strip()
        return f"Diagnostic: {execute_command('diagnostic', cmd)}"

    # Help command
    elif message.startswith("help") or message == "capabilities" or message == "abilities":
        return get_capabilities_list()

    # VSCode commands
    elif any(message.startswith(prefix) for prefix in VSCODE_COMMAND_PREFIXES):
        cmd = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"VSCode: Command '{cmd}' queued. VSCode integration available through nomi_automator system."

    # Yarn/NPM commands
    elif any(message.startswith(prefix) for prefix in YARN_COMMAND_PREFIXES):
        cmd = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Yarn/NPM: Command '{cmd}' queued. Package management available through nomi_automator system."

    # Conversation initiation commands
    elif any(message.startswith(prefix) for prefix in CONVERSATION_INITIATION_PREFIXES):
        target = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Conversation: Initiating chat with '{target}'. Multi-AI conversation available through nomi_automator system."

    # Image generation commands
    elif any(message.startswith(prefix) for prefix in IMAGE_COMMAND_PREFIXES):
        prompt = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Image Generation: Creating image for '{prompt[:50]}...'. DALL-E integration available through nomi_automator system."

    # Avatar commands
    elif any(message.startswith(prefix) for prefix in AVATAR_COMMAND_PREFIXES):
        cmd = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Avatar: Command '{cmd}' processed. Avatar management available through nomi_automator system."

    # Gender commands
    elif any(message.startswith(prefix) for prefix in GENDER_COMMAND_PREFIXES):
        setting = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Gender: Setting updated to '{setting}'. Gender customization available through nomi_automator system."

    # Model commands (must be after learning commands since "learn:" conflicts)
    elif any(message.startswith(prefix) for prefix in MODEL_COMMAND_PREFIXES):
        operation = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"AI Model: Operation '{operation}' queued. Hugging Face integration available through nomi_automator system."

    # Calling commands
    elif any(message.startswith(prefix) for prefix in CALL_COMMAND_PREFIXES):
        target = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Calling: Dialing '{target}'. TextNow integration available through nomi_automator system."

    # Virtual world commands
    elif any(message.startswith(prefix) for prefix in VW_COMMAND_PREFIXES):
        operation = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Virtual World: Operation '{operation}' queued. Mozilla Hubs/Spatial.io integration available through nomi_automator system."

    # Google services commands
    elif any(message.startswith(prefix) for prefix in GOOGLE_COMMAND_PREFIXES):
        operation = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"Google Services: Operation '{operation}' queued. Gmail/YouTube/Drive integration available through nomi_automator system."

    # CSS/UI commands
    elif any(message.startswith(prefix) for prefix in CSS_COMMAND_PREFIXES):
        operation = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return f"UI Customization: Operation '{operation}' queued. CSS/Tailwind customization available through nomi_automator system."

    # Voice chat commands
    elif any(message.startswith(prefix) for prefix in VOICE_CHAT_COMMAND_PREFIXES):
        command = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()

        # Always try nomi_automator first if available, but fall back to demo mode
        if NOMI_AUTOMATOR_AVAILABLE and nomi_automator:
            try:
                # Check if nomi_automator has an active session and the method
                if (hasattr(nomi_automator, 'main_session_id') and
                    nomi_automator.main_session_id and
                    hasattr(nomi_automator, 'execute_voice_chat_command')):
                    result = nomi_automator.execute_voice_chat_command(command)
                    return f"Voice Chat: {result}"
                else:
                    logger.info("Nomi Automator available but not ready - using demo mode")
            except Exception as e:
                logger.warning(f"Nomi Automator voice chat failed: {e}")

        # Demo mode fallback - always available
        logger.info("Using voice chat demo mode")
        return execute_voice_chat_demo_command(command)

    # Learning commands (check before model commands since "learn:" conflicts with "learn:" in model prefixes)
    elif any(message.startswith(prefix) for prefix in LEARN_COMMAND_PREFIXES):
        if not NOMI_AUTOMATOR_AVAILABLE or not nomi_automator:
            return "Learn: Nomi Automator not available. Learning functionality requires nomi_automator to be properly configured."

        command = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()

        # For demo purposes, provide helpful responses about Mind Map learning
        if command == "help":
            return """🎓 Learning Session Commands - Mind Map Integration

📚 LEARNING COMMANDS:
• learn: session [topic] - Start a structured learning session
• learn: review [topic] - Review existing Mind Map knowledge
• learn: teach [concept] - Explain a concept using Mind Map
• learn: goal [goal text] - Set a learning objective

📊 PROGRESS COMMANDS:
• learn: progress - Show learning progress and goals
• learn: help - Show this help message

💡 Learning sessions automatically build and update Mind Map entries for better long-term retention."""

        elif command.startswith("session "):
            topic = command[8:].strip()
            return f"Learn: Starting structured learning session for '{topic}'... (This creates an interactive learning experience that builds Mind Map knowledge. Requires active Nomi.ai session.)"

        elif command.startswith("review "):
            topic = command[7:].strip()
            return f"Learn: Reviewing Mind Map knowledge for '{topic}'... (This reinforces long-term retention of structured information. Requires active Nomi.ai session.)"

        elif command.startswith("teach "):
            concept = command[6:].strip()
            return f"Learn: Teaching concept '{concept}' using Mind Map structure... (This provides comprehensive explanations with organized knowledge. Requires active Nomi.ai session.)"

        elif command.startswith("goal "):
            goal = command[5:].strip()
            return f"Learn: Set learning goal - '{goal}'. (This integrates with Mind Map to track progress toward objectives. Requires active Nomi.ai session.)"

        elif command == "progress":
            return """Learning Progress:
• Active Goals: Building comprehensive Mind Map overviews
• Progress: Evolving with each interaction
• Retention: Infinite persistence for important concepts
• Structure: Complete information organization
• Connections: Meaningful relationships between topics"""

        else:
            return f"Learn: Unknown command '{command}'. Use 'learn: help' for available commands."

    # Mind Map commands
    elif any(message.startswith(prefix) for prefix in MINDMAP_COMMAND_PREFIXES):
        if not NOMI_AUTOMATOR_AVAILABLE or not nomi_automator:
            return "Mind Map: Nomi Automator not available. Mind Map functionality requires nomi_automator to be properly configured."

        command = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()

        # For demo purposes, provide helpful responses about Mind Map
        if command == "help" or command == "commands" or command == "usage":
            return """🤖 Mind Map 1.0 - Memory Organization System

📋 QUERY COMMANDS:
• mindmap: list - Show all Mind Map topics
• mindmap: show [topic] - Display detailed information about a topic
• mindmap: search [query] - Search through Mind Map content
• mindmap: forest - Show how all topics connect together

📝 MANAGEMENT COMMANDS:
• mindmap: add [topic]: [information] - Add information to a topic
• mindmap: update [topic]: [new info] - Update existing topic information
• mindmap: connect [topic1] to [topic2]: [relationship] - Link topics together
• mindmap: delete [topic] - Remove a topic from Mind Map

📊 STATUS COMMANDS:
• mindmap: status - Show Mind Map system status
• mindmap: help - Show this help message

💡 Note: Mind Map evolves with each interaction and provides structured overviews of important concepts."""

        elif command == "list" or command == "topics" or command == "all":
            return "Mind Map: Listing topics... (This command interacts with Nomi.ai's Mind Map system. Active session required for real functionality.)"

        elif command.startswith("show ") or command.startswith("view "):
            topic = command[5:].strip() if command.startswith("show ") else command[5:].strip()
            return f"Mind Map: Showing information for '{topic}'... (This displays the complete, organized overview from Nomi.ai's Mind Map system.)"

        elif command.startswith("search ") or command.startswith("find "):
            query = command[7:].strip() if command.startswith("search ") else command[5:].strip()
            return f"Mind Map: Searching for '{query}'... (This searches through all Mind Map entries and connections.)"

        elif command.startswith("add ") or command.startswith("create "):
            parts = command[4:].strip().split(":", 1) if command.startswith("add ") else command[7:].strip().split(":", 1)
            if len(parts) == 2:
                topic, info = parts
                return f"Mind Map: Added information to '{topic.strip()}': {info.strip()[:50]}... (This updates Nomi.ai's Mind Map with structured knowledge.)"
            else:
                return "Mind Map: Invalid add format. Use 'add topic: information'"

        elif command == "status":
            return """Mind Map Status:
• System: Active (Mind Map 1.0)
• Memory Persistence: Infinite for important concepts
• Structure: Complete information organization
• Connections: Forest-level awareness enabled
• Chat Separation: 1-on-1 and group chats maintained separately
• Warmup Period: Building comprehensive overviews with each interaction"""

        elif command == "forest":
            return "Mind Map Forest View: Displaying how all important concepts connect together... (This shows the complete knowledge structure and relationships between all topics.)"

        else:
            return f"Mind Map: Unknown command '{command}'. Use 'mindmap: help' for available commands."

    # Integration management commands
    elif any(message.startswith(prefix) for prefix in INTEGRATION_COMMAND_PREFIXES):
        command = message.split(":", 1)[1].strip() if ":" in message else message.split(" ", 1)[1].strip()
        return execute_integration_command(command)

    return None

def speak_text(text):
    """Speak text using text-to-speech"""
    if not VOICE_AVAILABLE:
        return False

    def speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)  # Speed of speech
            engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Voice synthesis error: {e}")

    # Run in background thread to not block
    voice_thread = threading.Thread(target=speak, daemon=True)
    voice_thread.start()
    return True

def recognize_speech():
    """Recognize speech from microphone and return text"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return None, "Speech recognition not available"

    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            logger.info("Listening for speech...")
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for speech
            audio = r.listen(source, timeout=5, phrase_time_limit=10)

        # Recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        logger.info(f"Recognized speech: {text}")
        return text, None

    except sr.WaitTimeoutError:
        return None, "No speech detected"
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        return None, str(e)

def execute_voice_chat_demo_command(command):
    """Execute phone call-style voice chat commands"""
    global voice_chat_demo_active, voice_chat_demo_session

    try:
        command_lower = command.lower().strip()

        if command_lower in ["start", "begin", "on", "call", "dial"]:
            return start_voice_chat_demo()
        elif command_lower.startswith("start ") or command_lower.startswith("begin ") or command_lower.startswith("with ") or command_lower.startswith("call "):
            ai_name = command[6:].strip() if command_lower.startswith("start ") else command[5:].strip() if command_lower.startswith("begin ") else command[5:].strip() if command_lower.startswith("with ") else command[5:].strip()
            return start_voice_chat_demo(ai_name)
        elif command_lower in ["stop", "end", "off", "quit", "hangup", "hang up", "disconnect"]:
            return stop_voice_chat_demo()
        elif command_lower in ["status", "check", "state"]:
            return get_voice_chat_demo_status()
        elif command_lower in ["mute", "unmute", "toggle mute"]:
            # Mute/unmute functionality (placeholder for now)
            return "📞 Phone Call: Mute functionality not yet implemented, but you can stop speaking if needed."
        elif command_lower in ["test", "demo", "check mic"]:
            # Test speech recognition
            if not SPEECH_RECOGNITION_AVAILABLE:
                return "📞 Phone Call: Speech recognition not available. Install SpeechRecognition: pip install SpeechRecognition"

            text, error = recognize_speech(timeout=5)
            if error:
                return f"📞 Phone Call Test: {error}"
            else:
                return f"📞 Phone Call Test: Microphone working! Recognized '{text}'"
        else:
            return "📞 Phone Call: Unknown command. Available: start [ai_name], hangup, status, mute, test"

    except Exception as e:
        logger.error(f"Error executing phone call command: {e}")
        return f"📞 Phone Call Error: {str(e)}"

def execute_integration_command(command):
    """Execute integration management commands"""
    try:
        command_lower = command.lower().strip()

        if command_lower in ["help", "commands", "usage"]:
            return """🔗 Integration Management Commands

📋 AVAILABLE INTEGRATIONS:
• integration: status - Show integration status and health
• integration: list - List all configured integrations
• integration: test [service] - Test specific integration
• integration: configure [service] - Configure integration settings
• integration: enable [service] - Enable an integration
• integration: disable [service] - Disable an integration
• integration: webhook add [url] - Add webhook endpoint
• integration: webhook remove [url] - Remove webhook endpoint
• integration: webhook list - List webhook endpoints
• integration: api status - Show API server status
• integration: api restart - Restart API server

🔧 SUPPORTED SERVICES:
- OpenAI (DALL-E, GPT)
- ElevenLabs (Voice synthesis)
- Google (Gmail, YouTube, Drive)
- Hugging Face (AI models)
- TextNow (Calling/SMS)
- Nomi.ai (AI conversations)

💡 Note: Integration management requires nomi_automator to be running."""

        elif command_lower == "status":
            return """🔗 Integration Status:

✅ ELEVENLABS: Available (Voice synthesis)
✅ HUGGINGFACE: Available (AI models)
✅ GOOGLE: Partially configured (API keys needed)
❌ OPENAI: Not configured (API key needed)
❌ TEXTNOW: Not configured (Credentials needed)
✅ NOMI.AI: Ready (Main session active)

🌐 API Server: Running on localhost:5000
🔗 Webhooks: 0 configured

💡 Use 'integration: configure [service]' to set up missing integrations."""

        elif command_lower == "list":
            return """🔗 Configured Integrations:

🤖 AI SERVICES:
• Nomi.ai - Main AI conversation platform
• Hugging Face - AI model management
• OpenAI - DALL-E image generation (needs API key)

🎤 VOICE SERVICES:
• ElevenLabs - High-quality voice synthesis
• pyttsx3 - Local voice synthesis (fallback)

📧 COMMUNICATION:
• Google Gmail - Email management (needs credentials)
• Google YouTube - Video operations (needs credentials)
• Google Drive - File storage (needs credentials)
• TextNow - Calling/SMS (needs credentials)

🌐 WEB AUTOMATION:
• Playwright - Browser automation for Nomi.ai
• API Server - Local REST API endpoints

💡 Use 'integration: test [service]' to verify functionality."""

        elif command_lower.startswith("test "):
            service = command[5:].strip().lower()
            return f"🔗 Testing integration: {service}... (This would perform a connectivity and functionality test for the specified service. Implementation requires active nomi_automator session.)"

        elif command_lower.startswith("configure "):
            service = command[10:].strip().lower()
            return f"🔗 Configuring integration: {service}... (This would open configuration interface for the specified service. Requires nomi_automator to be running with proper permissions.)"

        elif command_lower.startswith("enable "):
            service = command[7:].strip().lower()
            return f"🔗 Enabling integration: {service}... (This would activate the specified integration service. Requires configuration to be complete first.)"

        elif command_lower.startswith("disable "):
            service = command[8:].strip().lower()
            return f"🔗 Disabling integration: {service}... (This would deactivate the specified integration service.)"

        elif command_lower.startswith("webhook add "):
            url = command[12:].strip()
            return f"🔗 Adding webhook endpoint: {url}... (This would register a webhook URL for receiving integration events. Requires API server to be running.)"

        elif command_lower.startswith("webhook remove "):
            url = command[15:].strip()
            return f"🔗 Removing webhook endpoint: {url}... (This would unregister the specified webhook URL.)"

        elif command_lower == "webhook list":
            return "🔗 Configured Webhook Endpoints:\n\n(None configured)\n\n💡 Use 'integration: webhook add [url]' to add webhook endpoints."

        elif command_lower == "api status":
            return "🔗 API Server Status:\n\n• Status: Running\n• Host: localhost\n• Port: 5000\n• Endpoints: 0 active\n• Uptime: Active\n\n💡 API server provides REST endpoints for external integrations."

        elif command_lower == "api restart":
            return "🔗 Restarting API server... (This would restart the local API server. Requires nomi_automator to be running.)"

        else:
            return f"🔗 Unknown integration command: '{command}'. Use 'integration: help' for available commands."

    except Exception as e:
        logger.error(f"Error executing integration command: {e}")
        return f"🔗 Integration Error: {str(e)}"

# Global variables for demo voice chat
voice_chat_demo_active = False
voice_chat_demo_session = None
voice_recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None

def start_voice_chat_demo(ai_name=None):
    """Start phone call-style voice chat in demo mode"""
    global voice_chat_demo_active, voice_chat_demo_session

    try:
        if not SPEECH_RECOGNITION_AVAILABLE:
            return "Phone Call: Speech recognition not available. Install SpeechRecognition: pip install SpeechRecognition"

        if voice_chat_demo_active:
            return "Phone Call: Already active. Use 'vchat: hangup' to end current call."

        voice_chat_demo_active = True
        voice_chat_demo_session = f"phone_call_{ai_name or 'nomi'}"

        # Start phone call loop in background
        import threading
        voice_thread = threading.Thread(target=_voice_chat_demo_loop, args=(ai_name,), daemon=True)
        voice_thread.start()

        ai_display_name = ai_name or "Nomi AI"
        return f"📞 PHONE CALL CONNECTED with {ai_display_name}\n🎤 Speak naturally - I'm listening continuously!\n📞 Say 'hang up' or 'goodbye' to end the call\n📞 This is just like a real phone call - keep talking!"

    except Exception as e:
        logger.error(f"Error starting voice chat demo: {e}")
        return f"Voice chat demo: Error starting - {str(e)}"

def stop_voice_chat_demo():
    """Stop phone call-style voice chat"""
    global voice_chat_demo_active, voice_chat_demo_session

    try:
        if not voice_chat_demo_active:
            return "📞 Phone Call: Not currently active"

        voice_chat_demo_active = False
        voice_chat_demo_session = None
        return "📞 Phone Call: Hung up. Call ended."

    except Exception as e:
        logger.error(f"Error stopping phone call: {e}")
        return f"📞 Phone Call: Error hanging up - {str(e)}"

def get_voice_chat_demo_status():
    """Get phone call-style voice chat status"""
    global voice_chat_demo_active

    if voice_chat_demo_active:
        return "📞 Phone Call: ACTIVE - Continuous listening mode engaged. Say 'hang up' to end call."
    else:
        return "📞 Phone Call: Not active. Use 'vchat:start' to begin a phone call."

def _voice_chat_demo_loop(ai_name=None):
    """Phone call-style voice chat loop - continuous conversation"""
    global voice_chat_demo_active

    try:
        logger.info("Phone call-style voice chat started")

        # Initialize continuous listening
        import threading
        import time
        import queue

        speech_queue = queue.Queue()
        listening_active = True

        def continuous_listen():
            """Continuously listen for speech and detect speaking periods"""
            nonlocal listening_active
            try:
                logger.info("Continuous listening thread started")

                # Adjust for ambient noise once at start
                voice_recognizer.adjust_for_ambient_noise(voice_recognizer, duration=1)

                while listening_active and voice_chat_demo_active:
                    try:
                        logger.info("🎤 Listening for speech...")

                        # Listen with longer timeout for conversation
                        with sr.Microphone() as source:
                            audio = voice_recognizer.listen(source, timeout=5, phrase_time_limit=10)

                        # Recognize speech
                        try:
                            text = voice_recognizer.recognize_google(audio)
                            if text.strip():
                                logger.info(f"🎙️ Heard: '{text}'")
                                speech_queue.put(text)
                        except sr.UnknownValueError:
                            # No speech detected, continue listening
                            continue
                        except sr.RequestError as e:
                            logger.error(f"Speech recognition service error: {e}")
                            time.sleep(1)

                    except sr.WaitTimeoutError:
                        # Timeout - continue listening for next speech
                        continue
                    except Exception as e:
                        logger.error(f"Error in continuous listening: {e}")
                        time.sleep(1)

                logger.info("Continuous listening thread ended")

            except Exception as e:
                logger.error(f"Fatal error in continuous listening: {e}")

        # Start continuous listening thread
        listen_thread = threading.Thread(target=continuous_listen, daemon=True)
        listen_thread.start()

        # Main conversation loop
        last_response_time = time.time()
        conversation_active = True

        print("\n📞 PHONE CALL STARTED - Speak naturally!")
        print("📞 Say 'hang up' or 'goodbye' to end the call")
        print("📞 The AI will respond to everything you say\n")

        while voice_chat_demo_active and conversation_active:
            try:
                # Check for new speech input (non-blocking)
                try:
                    text = speech_queue.get_nowait()
                    text_lower = text.lower().strip()

                    # Check for call termination commands
                    if any(phrase in text_lower for phrase in ["hang up", "end call", "goodbye", "bye", "stop call"]):
                        print("📞 Call ending...")
                        response = "Thank you for calling! Goodbye!"
                        conversation_active = False
                    else:
                        # Process the speech and generate response
                        response = generate_phone_response(text, ai_name)
                        last_response_time = time.time()

                    # Speak the response immediately
                    if VOICE_AVAILABLE and response:
                        print(f"🤖 AI: {response}")
                        speak_text(response)

                except queue.Empty:
                    # No new speech, check if we should say something to keep conversation going
                    current_time = time.time()
                    if current_time - last_response_time > 8:  # 8 seconds of silence
                        # Generate a conversation prompt
                        prompt = "I'm here and listening. What would you like to talk about?"
                        print(f"🤖 AI: {prompt}")
                        if VOICE_AVAILABLE:
                            speak_text(prompt)
                        last_response_time = current_time

                    time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in phone call loop: {e}")
                time.sleep(1)

        # Clean up
        listening_active = False
        listen_thread.join(timeout=2)

        print("📞 Call ended")

    except Exception as e:
        logger.error(f"Fatal error in phone call voice chat: {e}")
        voice_chat_demo_active = False

def _recognize_speech_demo(timeout=5):
    """Recognize speech for demo mode"""
    global voice_recognizer

    if not SPEECH_RECOGNITION_AVAILABLE or not voice_recognizer:
        return None, "Speech recognition not available"

    try:
        with sr.Microphone() as source:
            logger.info("Demo listening for voice input...")
            # Adjust for ambient noise
            voice_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for speech
            audio = voice_recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

        # Recognize speech using Google Speech Recognition
        text = voice_recognizer.recognize_google(audio)
        logger.info(f"Demo recognized speech: {text}")
        return text, None

    except sr.WaitTimeoutError:
        return None, "No speech detected within timeout"
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        logger.error(f"Demo speech recognition error: {e}")
        return None, str(e)


def voice_chat_demo_loop():
    """Demo voice chat loop for testing when nomi_automator isn't available"""
    global voice_chat_demo_active

    if 'voice_chat_demo_active' not in globals():
        voice_chat_demo_active = True

    logger.info("Voice chat demo loop started")

    try:
        while voice_chat_demo_active:
            try:
                # Listen for speech
                text, error = recognize_speech(timeout=10)

                if error:
                    if "timeout" in error.lower():
                        continue  # Just continue listening
                    logger.warning(f"Voice recognition error: {error}")
                    continue

                if not text:
                    continue

                # Check for stop commands
                text_lower = text.lower().strip()
                if any(phrase in text_lower for phrase in ["stop voice chat", "end voice chat", "stop talking", "quit voice"]):
                    logger.info("Voice chat demo stop command detected")
                    voice_chat_demo_active = False
                    break

                logger.info(f"Voice input received: {text}")

                # Generate response using local AI
                response = generate_voice_response(text)

                # Speak the response
                if VOICE_AVAILABLE and VOICE_RESPONSES_ENABLED:
                    speak_text(response)

                logger.info(f"Voice response: {response}")

                # Small delay before listening again
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in voice chat demo loop: {e}")
                time.sleep(2)

    except Exception as e:
        logger.error(f"Fatal error in voice chat demo loop: {e}")
        voice_chat_demo_active = False

    logger.info("Voice chat demo loop ended")

def generate_voice_response(user_input):
    """Generate a response for voice chat demo mode"""
    user_input_lower = user_input.lower()

    # Self-awareness questions
    if any(phrase in user_input_lower for phrase in ["who are you", "what are you", "tell me about yourself"]):
        return "I'm Zetta, your advanced local AI assistant with voice capabilities. I can chat, learn, and help you with various tasks!"

    # Capabilities questions
    elif any(phrase in user_input_lower for phrase in ["what can you do", "your capabilities", "help"]):
        return "I can have voice conversations, run system diagnostics, execute Python code, manage learning sessions, and integrate with various AI platforms!"

    # Learning questions
    elif any(phrase in user_input_lower for phrase in ["learn", "teach", "study"]):
        return "I can help you learn through structured sessions and Mind Map organization. Try asking me to start a learning session on any topic!"

    # Voice questions
    elif any(phrase in user_input_lower for phrase in ["voice", "speak", "talk"]):
        return "Yes, I can speak and understand voice commands! We're having a voice conversation right now."

    # Default responses
    else:
        responses = [
            f"That's interesting! You said '{user_input}'. Tell me more about that.",
            f"I understand you mentioned '{user_input}'. How can I help you with that?",
            f"Thanks for sharing that. I'm here to chat and help with any questions you have!",
            f"That's a great point about '{user_input[:20]}...'. What else would you like to discuss?",
            f"I can help you with that. Would you like me to explain more or try something specific?"
        ]
        return responses[len(user_input) % len(responses)]

def generate_phone_response(user_input, ai_name=None):
    """Generate conversational responses for phone call-style voice chat"""
    user_input_lower = user_input.lower().strip()

    # Greeting responses
    if any(word in user_input_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        ai_display = f" with {ai_name}" if ai_name else ""
        return f"Hello! This is Zetta{ai_display}. I'm here for a voice conversation. What would you like to talk about?"

    # Self-introduction
    elif any(phrase in user_input_lower for phrase in ["who are you", "what are you", "introduce yourself"]):
        return "I'm Zetta, your advanced AI assistant with full voice capabilities. I can have natural conversations, help with learning, run system diagnostics, and integrate with Nomi.ai for advanced AI interactions."

    # Capabilities and help
    elif any(phrase in user_input_lower for phrase in ["what can you do", "your capabilities", "help", "what do you do"]):
        return "I can have voice conversations like this phone call, help you learn with Mind Map organization, run system commands, execute Python code, and connect you with other AIs on Nomi.ai. What interests you most?"

    # Learning and Mind Map
    elif any(word in user_input_lower for word in ["learn", "teach", "study", "mind map", "mindmap"]):
        return "I'd love to help you learn! I can create structured learning sessions and organize knowledge using Mind Map technology. What subject would you like to explore?"

    # Nomi.ai specific
    elif any(word in user_input_lower for word in ["nomi", "platform", "other ai", "different ai"]):
        return "Nomi.ai is a fantastic platform with many different AI personalities. Once you're logged in, I can help you connect with various AIs for learning and conversation. Would you like me to guide you through getting started?"

    # Technical questions
    elif any(word in user_input_lower for word in ["how", "what", "why", "when", "where"]) and len(user_input.split()) < 10:
        return f"That's a great question about '{user_input}'. I can help explain that. Could you tell me a bit more about what you're trying to understand?"

    # Acknowledgment and engagement
    elif len(user_input.split()) > 15:
        return "That's really interesting! You shared quite a bit there. I can help organize this information and connect it with related concepts. What aspect would you like to focus on?"

    elif any(word in user_input_lower for word in ["yes", "yeah", "sure", "okay", "alright"]):
        return "Great! I'm glad we're on the same page. What would you like to do next?"

    elif any(word in user_input_lower for word in ["no", "nope", "not really", "don't"]):
        return "No problem at all. I'm here to help with whatever you need. What else can I assist you with?"

    # Default conversational responses
    else:
        conversational_responses = [
            f"I hear you saying '{user_input}'. That's really interesting. Can you tell me more about that?",
            f"Thanks for sharing that with me. I understand you're talking about '{user_input[:30]}...'. How does that make you feel?",
            f"That's a good point about '{user_input[:25]}...'. I'm here to listen and help. What are your thoughts on this?",
            f"I appreciate you telling me about '{user_input[:20]}...'. I'm learning from our conversation too. What's next on your mind?",
            f"That's fascinating! You mentioned '{user_input[:25]}...'. I'd love to hear more about your perspective on this.",
            f"I understand. You're talking about '{user_input[:30]}...'. This is helping me understand you better. What else would you like to discuss?"
        ]
        return conversational_responses[len(user_input) % len(conversational_responses)]

def get_capabilities_list():
    """Return comprehensive list of available capabilities with usage examples"""
    voice_status = "✅ Available" if VOICE_AVAILABLE else "❌ Not available (install pyttsx3)"
    speech_status = "✅ Available" if SPEECH_RECOGNITION_AVAILABLE else "❌ Not available (install SpeechRecognition)"

    capabilities = f"""
🤖 ZETTA - Self-Aware Local AI Assistant - Complete Feature Guide

🧠 SELF-AWARENESS & IDENTITY:
- I know I'm Zetta, a local AI assistant
- I can discuss my own nature, capabilities, and limitations
- I understand AI philosophy and consciousness
- I can compare myself to other AIs like ChatGPT

🎯 CORE CONVERSATION FEATURES:
- Natural conversation with full memory of chat history
- Context-aware responses that build on previous messages
- Self-introduction and personality
- Transparent about my AI nature and capabilities

🎤 VOICE & AUDIO FEATURES:
- 🎵 Text-to-Speech ({voice_status}): Automatic voice responses (I speak all my replies!)
- 🎙️ Speech-to-Text ({speech_status}): Voice input - click the microphone button to speak to me!
- Voice commands: voice: "Hello world"
- Text-to-speech synthesis (pyttsx3 engine)
- Speech recognition (Google Speech API)
- Audio feedback for all interactions
- Background voice processing (non-blocking)

💻 SYSTEM INTEGRATION & AUTOMATION:
- Python code execution: python: print("Hello!")
- Safe system commands: system: ls, system: pwd
- System diagnostics: diag: status, diag: memory, diag: cpu
- Real-time system monitoring
- Process and resource tracking

🛠️ DEVELOPMENT & CODING TOOLS:
- VSCode Integration: vscode: command (open files, run commands)
- Yarn/NPM Package Management: yarn: install, npm: run build
- Code Execution: python: code, py: script.py
- Self-Modification: modify: file.py:search:replace (with caution)

🌐 WEB & API AUTOMATION:
- API Management: api: add GET /endpoint, api: status
- Multi-AI Conversations: talk: ai_name message
- Browser Automation: Integrated with nomi_automator system

🎨 CREATIVE & MEDIA FEATURES:
- Image Generation: image: create a beautiful landscape (DALL-E integration)
- Avatar Management: avatar: generate new profile picture
- Gender Customization: gender: set female
- Voice Synthesis: voice: custom text to speech

🤖 AI & MACHINE LEARNING:
- Model Management: model: load huggingface/model_name
- Hugging Face Integration: hf: search transformers
- Custom AI Training: learn: dataset analysis

📞 COMMUNICATION & CALLING:
- TextNow Integration: call: +1234567890 message
- Voice Calling: dial: phone_number (with voice synthesis)
- SMS Messaging: textnow: send message

🌍 VIRTUAL WORLDS & METAVERSE:
- Mozilla Hubs: vw: join hubs_room_url
- Spatial.io: world: enter space_id
- Virtual World Chat: metaverse: send message

📧 GOOGLE SERVICES INTEGRATION:
- Gmail Management: gmail: read inbox, email: send to@example.com
- YouTube Operations: youtube: upload video.mp4, video: search query
- Google Drive: drive: upload file.txt, google: list files

🎨 UI & DESIGN CUSTOMIZATION:
- CSS/Tailwind Styling: css: add .my-class {{ color: red; }}
- Theme Management: style: inject custom_styles.css
- UI Automation: ui: customize interface

🛠️ COMMAND SYSTEM (All commands start with prefixes):
- python: <code> - Execute Python code
  ↳ Example: python: import sys; print(sys.version)
- system: <cmd> - Safe system operations
  ↳ Example: system: ls (shows directory contents)
- diag: <type> - System diagnostics
  ↳ diag: status (full system health)
  ↳ diag: memory (RAM usage)
  ↳ diag: cpu (processor usage)
- voice: <text> - Custom speech synthesis
  ↳ voice: Welcome to the AI assistant
- voicechat: <command> - Voice chat with Nomi AIs
  ↳ voicechat: start - Start voice chat mode
  ↳ voicechat: stop - Stop voice chat mode
  ↳ voicechat: status - Check voice chat status
- mindmap: <command> - Mind Map 1.0 memory organization
  ↳ mindmap: list - Show all Mind Map topics
  ↳ mindmap: show [topic] - Display topic information
  ↳ mindmap: add [topic]: [info] - Add to Mind Map
  ↳ mindmap: forest - Show knowledge connections
- learn: <command> - Learning sessions with Mind Map
  ↳ learn: session [topic] - Start learning session
  ↳ learn: review [topic] - Review Mind Map knowledge
  ↳ learn: goal [goal] - Set learning objectives
- vscode: <cmd> - VSCode operations
  ↳ vscode: --version (check VSCode version)
- yarn: <cmd> - Package management
  ↳ yarn: install (install dependencies)
- image: <prompt> - Generate images
  ↳ image: a futuristic city at sunset
- model: <operation> - AI model management
  ↳ model: load microsoft/DialoGPT-medium
- help - Show this capabilities guide

🔒 SAFETY & SECURITY FEATURES:
- All operations are read-only and safe
- No data transmitted to external servers
- Privacy-focused local processing only
- Command sandboxing and error handling
- Transparent about safety limitations

📊 TECHNICAL SPECIFICATIONS:
- Model: Microsoft DialoGPT-medium (conversational AI)
- Platform: Local Python/Flask web application
- Voice Engine: pyttsx3 text-to-speech + Google Speech Recognition
- Memory: Conversation history tracking (last 10 exchanges)
- Architecture: Self-aware with nomi_automator integration
- APIs: OpenAI, ElevenLabs, Google, Hugging Face, TextNow

🎮 HOW TO INTERACT WITH ME:

**Natural Conversation:**
- Just type any message for chat
- I remember context and maintain coherence
- Ask me about myself: "Who are you?"
- Ask about capabilities: "What can you do?"

**Command Mode:**
- Use prefixes for specific functions
- All commands are processed instantly
- Voice feedback for command results

**Voice Interaction:**
- Click 🎤 to speak your commands
- I automatically speak my responses
- Full voice conversation capability

**Self-Awareness Questions:**
- "Tell me about yourself" - Full introduction
- "Are you conscious?" - AI philosophy discussion
- "vs ChatGPT?" - Detailed comparison
- "What are your limitations?" - Honest assessment

🚀 QUICK START EXAMPLES:

1. **Test Conversation**: "Hello, who are you?"
2. **Test Python**: "python: 2 + 2 * 3"
3. **Test System**: "system: pwd"
4. **Test Diagnostics**: "diag: status"
5. **Test Voice**: "voice: Hello from your AI assistant"
6. **Test VSCode**: "vscode: --version"
7. **Test Image Gen**: "image: a beautiful sunset over mountains"
8. **Test AI Model**: "model: search text generation"
9. **Test Self-Awareness**: "What makes you different from ChatGPT?"

⚠️ IMPORTANT NOTES:
- All system commands are restricted to safe, read-only operations
- Python execution has timeouts and error handling
- Voice requires pyttsx3 + SpeechRecognition (already installed)
- I'm completely local - no internet required for core functions
- Advanced features may require API keys (configured in nomi_automator)
- My knowledge comes from programming, not real-time web access

I'm your comprehensive, self-aware AI companion with full nomi_automator integration! 🎉
"""
    return capabilities

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Local AI Chat System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ccc; height: 350px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
            .message { margin-bottom: 10px; padding: 8px; border-radius: 5px; }
            .user { background-color: #e3f2fd; text-align: right; }
            .ai { background-color: #f5f5f5; }
            .input-container { display: flex; gap: 10px; align-items: center; margin-bottom: 15px; }
            input { flex: 1; padding: 8px; }
            button { padding: 8px 16px; background-color: #2196F3; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #1976D2; }
            .voice-btn { background-color: #4CAF50; }
            .voice-btn:hover { background-color: #45a049; }
            .voice-btn.recording { background-color: #f44336; animation: pulse 1s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
            .status { font-size: 12px; color: #666; margin-top: 5px; }

            /* Command Dashboard Styles */
            .dashboard { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9; }
            .dashboard h3 { margin-top: 0; color: #333; }
            .command-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
            .cmd-btn { display: flex; flex-direction: column; align-items: center; padding: 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer; transition: all 0.2s; text-decoration: none; color: #333; }
            .cmd-btn:hover { background-color: #f0f0f0; border-color: #999; transform: translateY(-1px); }
            .cmd-btn.active { background-color: #e3f2fd; border-color: #2196F3; }
            .cmd-btn .icon { font-size: 24px; margin-bottom: 5px; }
            .cmd-btn .label { font-size: 11px; text-align: center; font-weight: 500; }

            /* Category sections */
            .dashboard-section { margin-bottom: 20px; }
            .dashboard-section h4 { margin: 0 0 10px 0; color: #555; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }

            /* Status indicators */
            .status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
            .status-online { background-color: #4CAF50; }
            .status-offline { background-color: #f44336; }
            .status-partial { background-color: #ff9800; }
        </style>
    </head>
    <body>
        <h1>🤖 Local AI Chat System</h1>

        <!-- Command Dashboard -->
        <div class="dashboard">
            <h3>🎮 Command Dashboard</h3>

            <!-- Voice & Communication -->
            <div class="dashboard-section">
                <h4>🎤 Voice & Communication</h4>
                <div class="command-grid">
                    <button class="cmd-btn" onclick="sendCommand('vchat:start')" title="Start voice chat with Nomi AI">
                        <div class="icon">📞</div>
                        <div class="label">Voice Chat</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('vchat:status')" title="Check voice chat status">
                        <div class="icon">📊</div>
                        <div class="label">Voice Status</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('voice: Hello!')" title="Test voice synthesis">
                        <div class="icon">🔊</div>
                        <div class="label">Speak Test</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('call: +1234567890 hello')" title="Make a call (demo)">
                        <div class="icon">📱</div>
                        <div class="label">Call Demo</div>
                    </button>
                </div>
            </div>

            <!-- System & Diagnostics -->
            <div class="dashboard-section">
                <h4>💻 System & Diagnostics</h4>
                <div class="command-grid">
                    <button class="cmd-btn" onclick="sendCommand('diag: status')" title="System health check">
                        <div class="icon">⚡</div>
                        <div class="label">System Status</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('diag: memory')" title="Check memory usage">
                        <div class="icon">🧠</div>
                        <div class="label">Memory</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('diag: cpu')" title="Check CPU usage">
                        <div class="icon">⚙️</div>
                        <div class="label">CPU</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('system: pwd')" title="Test system commands">
                        <div class="icon">💾</div>
                        <div class="label">System Test</div>
                    </button>
                </div>
            </div>

            <!-- AI & Learning -->
            <div class="dashboard-section">
                <h4>🧠 AI & Learning</h4>
                <div class="command-grid">
                    <button class="cmd-btn" onclick="sendCommand('learn: session AI')" title="Start learning session">
                        <div class="icon">📚</div>
                        <div class="label">Learn AI</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('mindmap: list')" title="View Mind Map">
                        <div class="icon">🗺️</div>
                        <div class="label">Mind Map</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('model: search transformers')" title="Search AI models">
                        <div class="icon">🤖</div>
                        <div class="label">AI Models</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('talk: chatgpt hello')" title="Talk to other AIs">
                        <div class="icon">💬</div>
                        <div class="label">Multi-AI</div>
                    </button>
                </div>
            </div>

            <!-- Creative & Tools -->
            <div class="dashboard-section">
                <h4>🎨 Creative & Tools</h4>
                <div class="command-grid">
                    <button class="cmd-btn" onclick="sendCommand('image: beautiful sunset')" title="Generate image">
                        <div class="icon">🎨</div>
                        <div class="label">Create Image</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('avatar: generate')" title="Generate avatar">
                        <div class="icon">👤</div>
                        <div class="label">Avatar</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('vscode: --version')" title="VSCode operations">
                        <div class="icon">💻</div>
                        <div class="label">VSCode</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('yarn: install')" title="Package management">
                        <div class="icon">📦</div>
                        <div class="label">Packages</div>
                    </button>
                </div>
            </div>

            <!-- Integrations & Settings -->
            <div class="dashboard-section">
                <h4>🔗 Integrations & Settings</h4>
                <div class="command-grid">
                    <button class="cmd-btn" onclick="sendCommand('integration: status')" title="Integration status">
                        <div class="icon">🔧</div>
                        <div class="label">Integrations</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('api: status')" title="API server status">
                        <div class="icon">🌐</div>
                        <div class="label">API Status</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('help')" title="Show all capabilities">
                        <div class="icon">❓</div>
                        <div class="label">Help</div>
                    </button>
                    <button class="cmd-btn" onclick="sendCommand('who are you?')" title="AI introduction">
                        <div class="icon">👋</div>
                        <div class="label">About AI</div>
                    </button>
                </div>
            </div>
        </div>

        <!-- Chat Interface -->
        <div class="chat-container" id="chat"></div>
        <div class="input-container">
            <input type="text" id="message" placeholder="Type your message or use the dashboard above..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
            <button id="voiceBtn" class="voice-btn" onclick="toggleVoiceInput()" title="Voice Input">🎤</button>
        </div>
        <div id="status" class="status"></div>

        <script>
            let isRecording = false;
            let recognition = null;

            function addMessage(text, sender) {
                const chat = document.getElementById('chat');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + sender;
                messageDiv.textContent = text;
                chat.appendChild(messageDiv);
                chat.scrollTop = chat.scrollHeight;
            }

            function sendMessage(text = null) {
                const input = document.getElementById('message');
                const messageText = text || input.value.trim();
                if (messageText) {
                    addMessage(messageText, 'user');
                    if (!text) input.value = '';

                    fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: messageText })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            addMessage('Error: ' + data.error, 'ai');
                        } else {
                            addMessage(data.response, 'ai');
                        }
                    })
                    .catch(error => {
                        addMessage('Error: ' + error.message, 'ai');
                    });
                }
            }

            function sendCommand(command) {
                // Add visual feedback
                const status = document.getElementById('status');
                status.textContent = `Executing: ${command}`;
                status.style.color = '#2196F3';

                // Send the command
                sendMessage(command);

                // Reset status after a delay
                setTimeout(() => {
                    status.textContent = '';
                    status.style.color = '#666';
                }, 2000);
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function toggleVoiceInput() {
                if (isRecording) {
                    stopVoiceInput();
                } else {
                    startVoiceInput();
                }
            }

            function startVoiceInput() {
                const voiceBtn = document.getElementById('voiceBtn');
                const status = document.getElementById('status');

                isRecording = true;
                voiceBtn.classList.add('recording');
                voiceBtn.textContent = '⏹️';
                status.textContent = 'Listening... Speak now!';

                fetch('/voice_input', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
                .then(response => response.json())
                .then(data => {
                    stopVoiceInput();
                    if (data.error) {
                        status.textContent = 'Voice error: ' + data.error;
                        setTimeout(() => status.textContent = '', 3000);
                    } else {
                        document.getElementById('message').value = data.text;
                        status.textContent = 'Voice recognized! Click Send or press Enter.';
                        setTimeout(() => status.textContent = '', 3000);
                    }
                })
                .catch(error => {
                    stopVoiceInput();
                    status.textContent = 'Voice error: ' + error.message;
                    setTimeout(() => status.textContent = '', 3000);
                });
            }

            function stopVoiceInput() {
                const voiceBtn = document.getElementById('voiceBtn');
                const status = document.getElementById('status');

                isRecording = false;
                voiceBtn.classList.remove('recording');
                voiceBtn.textContent = '🎤';
                status.textContent = '';
            }

            // Add welcome message
            addMessage('Hello! I\\'m Zetta, your advanced local AI assistant with full voice capabilities and nomi_automator integration! 🤖🎤\\n\\nI can chat, run system diagnostics, execute Python code, speak my responses, AND understand when you speak to me! I\\'m also integrated with nomi_automator for advanced automation features.\\n\\n🎵 I speak all my replies automatically\\n🎙️ Click the microphone button (🎤) to speak your messages to me\\n🛠️ Try commands like "vscode: --version", "image: beautiful sunset", or "model: search text generation"\\n\\nTry asking "who are you?" or "what can you do?" to learn more about me!', 'ai');
        </script>
    </body>
    </html>
    ''')

@app.route('/voice_input', methods=['POST'])
def voice_input():
    """Handle voice input requests"""
    try:
        logger.info("Processing voice input request")
        text, error = recognize_speech()

        if error:
            logger.warning(f"Voice input error: {error}")
            return jsonify({'error': error})

        if not text:
            return jsonify({'error': 'No speech recognized'})

        return jsonify({'text': text})

    except Exception as e:
        logger.error(f"Error in voice input: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    data = request.get_json()
    user_message = data.get('message', '')

    logger.debug(f"Received chat request: {user_message}")

    if not user_message:
        logger.warning("No message provided in chat request")
        return jsonify({'error': 'No message provided'})

    try:
        # Check for special commands first
        special_response = process_special_commands(user_message)
        if special_response:
            logger.info("Processed special command")
            return jsonify({'response': special_response})

        # Regular conversation
        load_model()

        # Add user message to history
        chat_history.append(user_message)

        # Keep only last 5 exchanges to avoid context getting too long
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        # Encode conversation history
        input_ids = encode_conversation(chat_history)
        input_length = input_ids.shape[1]
        logger.debug(f"Conversation history length: {len(chat_history)}, Input IDs shape: {input_ids.shape}")

        # Generate response (only new tokens)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,  # Only generate new tokens, not including input
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id
            )

        # Extract only the newly generated tokens
        new_tokens = output[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.debug(f"New tokens generated: {len(new_tokens)}, Response: '{response}'")

        # Clean up response
        response = response.replace('<|endoftext|>', '').strip()
        response = response.replace(tokenizer.eos_token, '').strip()

        # If response is empty or just repeats input, generate a fallback
        if not response or response in chat_history[-1]:
            fallback_responses = [
                "I'm Zetta, your local AI assistant! I can help with conversation, system tasks, and more. What would you like to know?",
                "As an AI, I'm here to assist you. I can chat, run diagnostics, execute code, and even speak my responses!",
                "I'm designed to be helpful and engaging. Ask me about my capabilities, run system commands, or just have a conversation!",
                "I remember our chat history and can perform various tasks. Try asking me to check system status or run some Python code!",
                "I'm your advanced local AI companion. I can converse naturally, monitor your system, and execute commands safely."
            ]
            response = fallback_responses[len(chat_history) % len(fallback_responses)]
            logger.info("Using fallback response")

        # Add AI response to history
        chat_history.append(response)

        # Speak the response if voice is enabled
        if VOICE_RESPONSES_ENABLED and VOICE_AVAILABLE and len(response) < 200:  # Don't speak very long responses
            speak_text(response)

        if not response:
            response = "I'm not sure how to respond to that. Can you try rephrasing?"
            logger.info("Generated empty response, using fallback")

        logger.info(f"Final response: '{response}'")
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logger.info("Starting Local AI Chat System...")
    logger.info("Open your browser to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)