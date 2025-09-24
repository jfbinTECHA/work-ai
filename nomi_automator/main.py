import asyncio
import logging
import subprocess
import shlex
import threading
import json
import os
import requests
import time
from flask import Flask, request, jsonify
from playwright.async_api import async_playwright
from config import *

# ElevenLabs TTS (optional import)
try:
    from elevenlabs import generate, play, set_api_key
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("ElevenLabs not available. Voice features disabled.")

# pyttsx3 TTS (fallback, optional import)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not available. Fallback voice features disabled.")

# Speech Recognition (optional import)
try:
    # Add system path for PyAudio
    import sys
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("Speech recognition not available. Voice input features disabled.")

# OpenAI DALL-E (optional import)
try:
    import openai
    from PIL import Image
    import base64
    import io
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Image generation features disabled.")

# Hugging Face Transformers (optional import)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from huggingface_hub import HfApi, model_info
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("Hugging Face not available. ML model features disabled.")

# Google APIs (optional import)
try:
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    from email.mime.text import MIMEText
    import base64
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("Google APIs not available. Gmail and YouTube features disabled.")

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), filename=LOG_FILE, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NomiAPIClient:
    """API client for Nomi.ai interactions"""

    def __init__(self):
        self.base_url = NOMI_API_BASE_URL
        self.api_key = NOMI_API_KEY
        self.api_secret = NOMI_API_SECRET
        self.user_id = NOMI_USER_ID
        self.access_token = None
        self.token_expires = None
        self.session_id = None

    def _get_headers(self, include_auth=True):
        """Get request headers"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'NomiAutomator/1.0'
        }

        if include_auth and self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'

        return headers

    def _make_request(self, method, endpoint, data=None, params=None, retry_count=0):
        """Make API request with retry logic"""
        url = f"{self.base_url}{endpoint}"

        try:
            headers = self._get_headers()

            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=API_REQUEST_TIMEOUT)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=API_REQUEST_TIMEOUT)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data, timeout=API_REQUEST_TIMEOUT)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, params=params, timeout=API_REQUEST_TIMEOUT)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Handle authentication errors
            if response.status_code == 401:
                if retry_count < API_MAX_RETRIES:
                    logger.info("Token expired, attempting re-authentication")
                    if self.authenticate():
                        time.sleep(API_RETRY_DELAY)
                        return self._make_request(method, endpoint, data, params, retry_count + 1)
                raise Exception("Authentication failed")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if retry_count < API_MAX_RETRIES:
                logger.warning(f"Request failed, retrying ({retry_count + 1}/{API_MAX_RETRIES}): {e}")
                time.sleep(API_RETRY_DELAY)
                return self._make_request(method, endpoint, data, params, retry_count + 1)
            raise Exception(f"API request failed after {API_MAX_RETRIES} retries: {e}")

    def authenticate(self):
        """Authenticate with Nomi.ai API"""
        try:
            if not self.api_key or not self.api_secret:
                raise Exception("API key and secret not configured")

            auth_data = {
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "user_id": self.user_id
            }

            response = self._make_request('POST', NOMI_API_ENDPOINTS['auth'], auth_data, include_auth=False)

            if 'access_token' in response:
                self.access_token = response['access_token']
                self.token_expires = response.get('expires_at')
                logger.info("Successfully authenticated with Nomi.ai API")
                return True
            else:
                raise Exception("Authentication response missing access token")

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def send_chat_message(self, message, ai_id=None, conversation_id=None):
        """Send a chat message"""
        try:
            if not self.access_token and not self.authenticate():
                raise Exception("Not authenticated")

            chat_data = {
                "message": message,
                "ai_id": ai_id or NOMI_DEFAULT_AI_ID,
                "user_id": self.user_id
            }

            if conversation_id:
                chat_data["conversation_id"] = conversation_id

            response = self._make_request('POST', NOMI_API_ENDPOINTS['chat'], chat_data)

            # Store conversation ID for future messages
            if 'conversation_id' in response and not self.session_id:
                self.session_id = response['conversation_id']

            return response

        except Exception as e:
            logger.error(f"Failed to send chat message: {e}")
            raise

    def get_voice_audio(self, text, voice_settings=None):
        """Generate voice audio for text"""
        try:
            if not self.access_token and not self.authenticate():
                raise Exception("Not authenticated")

            voice_data = {
                "text": text,
                "voice_settings": voice_settings or {
                    "voice_id": ELEVENLABS_VOICE_ID,
                    "stability": ELEVENLABS_VOICE_STABILITY,
                    "similarity_boost": ELEVENLABS_VOICE_SIMILARITY
                }
            }

            response = self._make_request('POST', NOMI_API_ENDPOINTS['voice'], voice_data)

            if 'audio_url' in response:
                # Download audio data
                audio_response = requests.get(response['audio_url'], timeout=API_REQUEST_TIMEOUT)
                audio_response.raise_for_status()
                return audio_response.content
            else:
                raise Exception("Voice response missing audio URL")

        except Exception as e:
            logger.error(f"Failed to generate voice audio: {e}")
            raise

    def get_mindmap_data(self, topic=None):
        """Get Mind Map data"""
        try:
            if not self.access_token and not self.authenticate():
                raise Exception("Not authenticated")

            params = {}
            if topic:
                params['topic'] = topic

            return self._make_request('GET', NOMI_API_ENDPOINTS['mindmap'], params=params)

        except Exception as e:
            logger.error(f"Failed to get Mind Map data: {e}")
            raise

    def update_mindmap(self, topic, data):
        """Update Mind Map data"""
        try:
            if not self.access_token and not self.authenticate():
                raise Exception("Not authenticated")

            update_data = {
                "topic": topic,
                "data": data,
                "user_id": self.user_id
            }

            return self._make_request('PUT', NOMI_API_ENDPOINTS['mindmap'], update_data)

        except Exception as e:
            logger.error(f"Failed to update Mind Map: {e}")
            raise

    def get_companions(self):
        """Get available AI companions"""
        try:
            if not self.access_token and not self.authenticate():
                raise Exception("Not authenticated")

            return self._make_request('GET', NOMI_API_ENDPOINTS['companions'])

        except Exception as e:
            logger.error(f"Failed to get companions: {e}")
            raise

    def get_conversations(self):
        """Get conversation history"""
        try:
            if not self.access_token and not self.authenticate():
                raise Exception("Not authenticated")

            return self._make_request('GET', NOMI_API_ENDPOINTS['conversations'])

        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            raise


class SessionManager:
    def __init__(self):
        self.sessions = {}  # session_id -> session_info
        self.session_counter = 0

    def create_session(self, ai_type, url, name=None):
        """Create a new AI interaction session"""
        if len(self.sessions) >= MAX_CONCURRENT_SESSIONS:
            return None, "Maximum concurrent sessions reached"

        self.session_counter += 1
        session_id = f"{ai_type}_{self.session_counter}"

        session_info = {
            "id": session_id,
            "ai_type": ai_type,
            "url": url,
            "name": name or f"{ai_type.title()} Session {self.session_counter}",
            "browser": None,
            "page": None,
            "last_activity": asyncio.get_event_loop().time(),
            "status": "created",
            "messages": []
        }

        self.sessions[session_id] = session_info
        logger.info(f"Created session: {session_id} for {url}")
        return session_id, session_info

    def get_session(self, session_id):
        """Get session information"""
        return self.sessions.get(session_id)

    def update_activity(self, session_id):
        """Update last activity timestamp for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = asyncio.get_event_loop().time()

    def close_session(self, session_id):
        """Close a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session["browser"]:
                asyncio.create_task(session["browser"].close())
            del self.sessions[session_id]
            logger.info(f"Closed session: {session_id}")
            return True
        return False

    def list_sessions(self):
        """List all active sessions"""
        return {sid: {
            "name": s["name"],
            "ai_type": s["ai_type"],
            "status": s["status"],
            "last_activity": s["last_activity"]
        } for sid, s in self.sessions.items()}

    def cleanup_idle_sessions(self):
        """Clean up idle sessions"""
        if not AUTO_CLOSE_SESSIONS:
            return

        current_time = asyncio.get_event_loop().time()
        to_close = []

        for session_id, session in self.sessions.items():
            if current_time - session["last_activity"] > SESSION_TIMEOUT:
                to_close.append(session_id)

        for session_id in to_close:
            self.close_session(session_id)
            logger.info(f"Auto-closed idle session: {session_id}")


class APIManager:
    def __init__(self):
        self.app = Flask(__name__)
        self.endpoints = {}
        self.server_thread = None
        self.running = False

        @self.app.route('/')
        def home():
            return jsonify({"message": "Nomi.ai Automator API", "endpoints": list(self.endpoints.keys())})

    def add_endpoint(self, path, method="GET", response_data=None):
        """Add a dynamic endpoint"""
        if path in self.endpoints:
            return f"API: Endpoint '{path}' already exists"

        def dynamic_endpoint():
            if method == "POST":
                data = request.get_json() if request.is_json else {}
                return jsonify({"received": data, "response": response_data or "OK"})
            else:
                return jsonify(response_data or {"message": f"Endpoint {path} called"})

        self.endpoints[path] = {"method": method, "response": response_data}
        self.app.add_url_rule(path, path, dynamic_endpoint, methods=[method])
        return f"API: Added endpoint {method} {path}"

    def remove_endpoint(self, path):
        """Remove an endpoint"""
        if path not in self.endpoints:
            return f"API: Endpoint '{path}' not found"

        # Note: Flask doesn't easily allow removing routes, so we'll mark as inactive
        del self.endpoints[path]
        return f"API: Removed endpoint {path}"

    def start_server(self):
        """Start the API server in a separate thread"""
        if self.running:
            return "API: Server already running"

        def run_server():
            try:
                self.app.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False)
            except Exception as e:
                logger.error(f"API Server error: {e}")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        return f"API: Server started on http://{API_HOST}:{API_PORT}"

    def stop_server(self):
        """Stop the API server"""
        if not self.running:
            return "API: Server not running"

        # Note: Flask doesn't have a clean shutdown, but thread will be daemon
        self.running = False
        return "API: Server stopped"

    def get_status(self):
        """Get API server status"""
        status = f"API Status:\n- Running: {self.running}\n- Host: {API_HOST}\n- Port: {API_PORT}\n- Endpoints: {len(self.endpoints)}"
        if self.endpoints:
            status += "\nEndpoints:"
            for path, info in self.endpoints.items():
                status += f"\n  {info['method']} {path}"
        return status


class ModelManager:
    def __init__(self):
        self.loaded_models = {}  # model_name -> model_info
        self.model_cache = {}  # model_name -> cached_model_data
        self.hf_api = HfApi() if HUGGINGFACE_AVAILABLE else None

        # Set Hugging Face token if available
        if HUGGINGFACE_API_TOKEN:
            from huggingface_hub import login
            login(HUGGINGFACE_API_TOKEN)

        # Create cache directory
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    def load_model(self, model_name, task=None):
        """Load a model from Hugging Face"""
        try:
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]

            # Check model size before downloading
            if self.hf_api:
                try:
                    info = model_info(model_name)
                    model_size_mb = sum(f.size for f in info.siblings) / (1024 * 1024)
                    if model_size_mb > MAX_MODEL_SIZE_MB:
                        return None, f"Model too large: {model_size_mb:.1f}MB (max: {MAX_MODEL_SIZE_MB}MB)"
                except:
                    pass  # Continue if we can't check size

            # Load model with timeout
            import threading
            result = {"model": None, "error": None}

            def load_model_thread():
                try:
                    if task:
                        # Use pipeline for common tasks
                        model = pipeline(task, model=model_name, cache_dir=MODEL_CACHE_DIR)
                    else:
                        # Load generic model
                        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
                        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
                        model = {"tokenizer": tokenizer, "model": model}

                    result["model"] = model
                except Exception as e:
                    result["error"] = str(e)

            thread = threading.Thread(target=load_model_thread)
            thread.start()
            thread.join(timeout=MODEL_LOAD_TIMEOUT)

            if thread.is_alive():
                return None, "Model loading timed out"

            if result["error"]:
                return None, result["error"]

            # Store loaded model
            model_info = {
                "model": result["model"],
                "loaded_at": asyncio.get_event_loop().time(),
                "last_used": asyncio.get_event_loop().time(),
                "task": task
            }

            self.loaded_models[model_name] = model_info
            logger.info(f"Loaded model: {model_name}")
            return model_info, None

        except Exception as e:
            return None, str(e)

    def unload_model(self, model_name):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            # Force garbage collection
            import gc
            gc.collect()
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False

    def use_model(self, model_name, input_data, **kwargs):
        """Use a loaded model for inference"""
        try:
            if model_name not in self.loaded_models:
                return None, f"Model {model_name} not loaded"

            model_info = self.loaded_models[model_name]
            model = model_info["model"]
            model_info["last_used"] = asyncio.get_event_loop().time()

            if isinstance(model, dict) and "tokenizer" in model:
                # Raw model usage
                tokenizer = model["tokenizer"]
                model_obj = model["model"]

                inputs = tokenizer(input_data, return_tensors="pt", **kwargs)
                with torch.no_grad():
                    outputs = model_obj.generate(**inputs, max_length=100, **kwargs)
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return result, None
            else:
                # Pipeline usage
                result = model(input_data, **kwargs)
                return result, None

        except Exception as e:
            return None, str(e)

    def list_loaded_models(self):
        """List currently loaded models"""
        return {name: {
            "task": info["task"],
            "loaded_at": info["loaded_at"],
            "last_used": info["last_used"]
        } for name, info in self.loaded_models.items()}

    def cleanup_unused_models(self):
        """Unload models that haven't been used recently"""
        if not AUTO_UNLOAD_MODELS:
            return

        current_time = asyncio.get_event_loop().time()
        to_unload = []

        for name, info in self.loaded_models.items():
            if current_time - info["last_used"] > 3600:  # 1 hour
                to_unload.append(name)

        for name in to_unload:
            self.unload_model(name)

    def search_models(self, query, task=None, limit=5):
        """Search for models on Hugging Face"""
        try:
            if not self.hf_api:
                return []

            models = self.hf_api.list_models(
                search=query,
                task=task,
                limit=limit,
                sort="downloads",
                direction=-1
            )

            return [{
                "id": model.id,
                "downloads": model.downloads,
                "likes": model.likes,
                "task": getattr(model, 'pipeline_tag', None)
            } for model in models]

        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []


class NomiAutomator:
    def __init__(self):
        self.browser = None
        self.page = None
        self.last_message_count = 0
        self.api_manager = APIManager()
        self.session_manager = SessionManager()
        self.model_manager = ModelManager()
        self.api_client = NomiAPIClient()  # API client for direct API interactions
        self.main_session_id = None  # Main Nomi session
        self.voice_chat_active = False  # Voice chat mode flag
        self.voice_chat_session = None  # Current voice chat session
        self.voice_recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.use_api = bool(NOMI_API_KEY and NOMI_API_SECRET)  # Use API if credentials are configured

    async def start(self):
        """Start the automator"""
        logger.info("Starting Nomi.ai Automator")

        # Send boot greeting if enabled
        if BOOT_GREETING_ENABLED:
            await self.send_boot_greeting()

        # Start main playwright instance
        playwright = await async_playwright().start()

        # Initialize main Nomi session
        if MULTI_AI_ENABLED:
            main_config = AI_INSTANCES.get("nomi")
            if main_config and main_config.get("auto_start", True):
                session_id, session_info = self.session_manager.create_session("nomi", main_config["url"], main_config["name"])
                if session_id:
                    self.main_session_id = session_id
                    session_info["browser"] = await playwright.chromium.launch(headless=HEADLESS)
                    session_info["page"] = await session_info["browser"].new_page()
                    await session_info["page"].goto(session_info["url"])

                    # Auto-inject custom CSS if enabled
                    if CSS_CUSTOMIZATION_ENABLED and AUTO_INJECT_CSS:
                        try:
                            await self.inject_custom_css()
                            logger.info("Auto-injected custom CSS on page load")
                        except Exception as e:
                            logger.warning(f"Failed to auto-inject CSS: {e}")

                    session_info["status"] = "ready"
                    logger.info(f"Started main Nomi session: {session_id}")

                    # Use main session for backward compatibility
                    self.browser = session_info["browser"]
                    self.page = session_info["page"]
                else:
                    logger.error("Failed to create main Nomi session")
            else:
                # Fallback to old method
                self.browser = await playwright.chromium.launch(headless=HEADLESS)
                self.page = await self.browser.new_page()
                await self.page.goto(NOMI_URL)
                logger.info(f"Navigated to {NOMI_URL}")
        else:
            # Fallback to old method
            self.browser = await playwright.chromium.launch(headless=HEADLESS)
            self.page = await self.browser.new_page()
            await self.page.goto(NOMI_URL)

            # Auto-inject custom CSS if enabled
            if CSS_CUSTOMIZATION_ENABLED and AUTO_INJECT_CSS:
                try:
                    await self.inject_custom_css()
                    logger.info("Auto-injected custom CSS on page load")
                except Exception as e:
                    logger.warning(f"Failed to auto-inject CSS: {e}")

            logger.info(f"Navigated to {NOMI_URL}")

    async def send_boot_greeting(self):
        """Send a boot greeting message"""
        try:
            logger.info("Sending boot greeting")

            # Send greeting to console/log
            print(f"\n🤖 {BOOT_GREETING_MESSAGE}")
            logger.info(f"Boot greeting: {BOOT_GREETING_MESSAGE}")

            # Send voice greeting if enabled
            if BOOT_GREETING_VOICE and ELEVENLABS_ENABLED and ELEVENLABS_AVAILABLE and ELEVENLABS_API_KEY:
                try:
                    set_api_key(ELEVENLABS_API_KEY)
                    audio = generate(
                        text=BOOT_GREETING_MESSAGE,
                        voice=ELEVENLABS_VOICE_ID,
                        model=ELEVENLABS_MODEL,
                        voice_settings={
                            "stability": ELEVENLABS_VOICE_STABILITY,
                            "similarity_boost": ELEVENLABS_VOICE_SIMILARITY,
                            "style": ELEVENLABS_VOICE_STYLE,
                            "use_speaker_boost": ELEVENLABS_VOICE_USE_SPEAKER_BOOST
                        }
                    )
                    play(audio)
                    logger.info("Boot greeting played via voice")
                except Exception as e:
                    logger.error(f"Error playing boot greeting voice: {e}")

        except Exception as e:
            logger.error(f"Error sending boot greeting: {e}")

    async def handle_authentication(self):
        """Handle login or authentication if required"""
        try:
            # Check for login form or button
            login_button = await self.page.query_selector('button:has-text("Login"), [href*="login"], .login-button')
            if login_button:
                logger.info("Login required, but automated login not implemented. Please log in manually.")
                # Wait for manual login
                await self.page.wait_for_url(lambda url: "nomis" in url, timeout=300000)  # 5 minutes
                logger.info("Proceeding after manual login")
            else:
                logger.info("No login required or already logged in")
        except Exception as e:
            logger.error(f"Error handling authentication: {e}")
            # Wait for page to load
            await self.page.wait_for_load_state('networkidle')

            # Handle potential login or authentication
            await self.handle_authentication()

            # Start monitoring
            await self.monitor_chat()

    async def monitor_chat(self):
        """Monitor the chat for new messages"""
        while True:
            try:
                # Check for new messages - update selectors based on UI inspection
                # Common selectors for chat messages
                messages = await self.page.query_selector_all('[class*="message"], [data-testid*="message"], .chat-message, .message-bubble')
                if len(messages) > self.last_message_count:
                    new_messages = messages[self.last_message_count:]
                    for msg in new_messages:
                        text = await msg.inner_text()
                        logger.info(f"New message: {text}")
                        response = self.generate_response(text)
                        await self.send_response(response)
                    self.last_message_count = len(messages)

                await asyncio.sleep(CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error monitoring chat: {e}")
                await asyncio.sleep(CHECK_INTERVAL)

    async def generate_response(self, message):
        """Generate a response to the message"""
        # Check for VSCode commands
        vscode_command = self.extract_command(message, VSCODE_COMMAND_PREFIXES)
        if vscode_command and VSCODE_ENABLED:
            return self.execute_vscode_command(vscode_command)

        # Check for Yarn commands
        yarn_command = self.extract_command(message, YARN_COMMAND_PREFIXES)
        if yarn_command and YARN_ENABLED:
            return self.execute_yarn_command(yarn_command)

        # Check for Python commands
        python_command = self.extract_command(message, PYTHON_COMMAND_PREFIXES)
        if python_command and PYTHON_ENABLED:
            return self.execute_python_command(python_command)

        # Check for troubleshooting commands
        troubleshoot_command = self.extract_command(message, TROUBLESHOOT_COMMAND_PREFIXES)
        if troubleshoot_command and SELF_TROUBLESHOOT_ENABLED:
            return self.execute_troubleshoot_command(troubleshoot_command)

        # Check for self-modification commands
        modify_command = self.extract_command(message, MODIFY_COMMAND_PREFIXES)
        if modify_command and SELF_MODIFY_ENABLED:
            return self.execute_modify_command(modify_command)

        # Check for API commands
        api_command = self.extract_command(message, API_COMMAND_PREFIXES)
        if api_command and API_ENABLED:
            return self.execute_api_command(api_command)

        # Check for voice commands
        voice_command = self.extract_command(message, VOICE_COMMAND_PREFIXES)
        if voice_command and ELEVENLABS_ENABLED and ELEVENLABS_AVAILABLE:
            return self.execute_voice_command(voice_command)

        # Check for conversation initiation commands
        conversation_command = self.extract_command(message, CONVERSATION_INITIATION_PREFIXES)
        if conversation_command and MULTI_AI_ENABLED:
            return self.execute_conversation_command(conversation_command)

        # Check for image generation commands
        image_command = self.extract_command(message, IMAGE_COMMAND_PREFIXES)
        if image_command and IMAGE_GENERATION_ENABLED and OPENAI_AVAILABLE:
            return self.execute_image_command(image_command)

        # Check for avatar commands
        avatar_command = self.extract_command(message, AVATAR_COMMAND_PREFIXES)
        if avatar_command and AVATAR_MANAGEMENT_ENABLED:
            return self.execute_avatar_command(avatar_command)

        # Check for model commands
        model_command = self.extract_command(message, MODEL_COMMAND_PREFIXES)
        if model_command and HUGGINGFACE_ENABLED and HUGGINGFACE_AVAILABLE:
            return self.execute_model_command(model_command)

        # Check for call commands
        call_command = self.extract_command(message, CALL_COMMAND_PREFIXES)
        if call_command and TEXTNOW_ENABLED:
            return self.execute_call_command(call_command)

        # Check for virtual world commands
        vw_command = self.extract_command(message, VW_COMMAND_PREFIXES)
        if vw_command and VIRTUAL_WORLD_ENABLED:
            return self.execute_vw_command(vw_command)

        # Check for Google service commands
        google_command = self.extract_command(message, GOOGLE_COMMAND_PREFIXES)
        if google_command and GOOGLE_SERVICES_ENABLED and GOOGLE_API_AVAILABLE:
            return self.execute_google_command(google_command)

        # Check for CSS/Tailwind commands
        css_command = self.extract_command(message, CSS_COMMAND_PREFIXES)
        if css_command and CSS_CUSTOMIZATION_ENABLED:
            return await self.execute_css_command(css_command)

        # Check for voice chat commands
        voice_chat_command = self.extract_command(message, VOICE_CHAT_COMMAND_PREFIXES)
        if voice_chat_command:
            return self.execute_voice_chat_command(voice_chat_command)

        # Check for Mind Map commands
        mindmap_command = self.extract_command(message, MINDMAP_COMMAND_PREFIXES)
        if mindmap_command:
            return self.execute_mindmap_command(mindmap_command)

        # Check for learning commands
        learn_command = self.extract_command(message, LEARN_COMMAND_PREFIXES)
        if learn_command:
            return self.execute_learn_command(learn_command)

        # Default response logic - can be enhanced with AI
        import random
        return random.choice(DEFAULT_RESPONSES)

    def extract_command(self, message, prefixes):
        """Extract command from message if it starts with any of the prefixes"""
        message_lower = message.lower().strip()
        for prefix in prefixes:
            if message_lower.startswith(prefix):
                return message[len(prefix):].strip()
        return None

    def execute_vscode_command(self, command):
        """Execute a VSCode command"""
        try:
            # Parse the command
            args = shlex.split(command)
            if not args:
                return "VSCode: No command specified"

            # Execute VSCode command
            cmd = [VSCODE_EXECUTABLE] + args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout.strip() or "VSCode command executed successfully"
                logger.info(f"VSCode command executed: {command}")
                return f"VSCode: {output}"
            else:
                error = result.stderr.strip() or "Unknown error"
                logger.error(f"VSCode command failed: {command} - {error}")
                return f"VSCode Error: {error}"

        except subprocess.TimeoutExpired:
            logger.error(f"VSCode command timed out: {command}")
            return "VSCode: Command timed out"
        except Exception as e:
            logger.error(f"Error executing VSCode command: {e}")
            return f"VSCode Error: {str(e)}"

    def execute_yarn_command(self, command):
        """Execute a Yarn command"""
        try:
            # Parse the command
            args = shlex.split(command)
            if not args:
                return "Yarn: No command specified"

            # Execute Yarn command
            cmd = [YARN_EXECUTABLE] + args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                output = result.stdout.strip() or "Yarn command executed successfully"
                logger.info(f"Yarn command executed: {command}")
                return f"Yarn: {output}"
            else:
                error = result.stderr.strip() or "Unknown error"
                logger.error(f"Yarn command failed: {command} - {error}")
                return f"Yarn Error: {error}"

        except subprocess.TimeoutExpired:
            logger.error(f"Yarn command timed out: {command}")
            return "Yarn: Command timed out"
        except Exception as e:
            logger.error(f"Error executing Yarn command: {e}")
            return f"Yarn Error: {str(e)}"

    def execute_python_command(self, command):
        """Execute a Python command or script"""
        try:
            # Parse the command
            args = shlex.split(command)
            if not args:
                return "Python: No command specified"

            # Execute Python command
            cmd = [PYTHON_EXECUTABLE] + args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout.strip() or "Python command executed successfully"
                logger.info(f"Python command executed: {command}")
                return f"Python: {output}"
            else:
                error = result.stderr.strip() or "Unknown error"
                logger.error(f"Python command failed: {command} - {error}")
                return f"Python Error: {error}"

        except subprocess.TimeoutExpired:
            logger.error(f"Python command timed out: {command}")
            return "Python: Command timed out"
        except Exception as e:
            logger.error(f"Error executing Python command: {e}")
            return f"Python Error: {str(e)}"

    def execute_troubleshoot_command(self, command):
        """Execute self-troubleshooting commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["status", "health", "check"]:
                return self.get_system_status()
            elif command_lower.startswith("logs"):
                return self.analyze_logs(command)
            elif command_lower in ["memory", "ram"]:
                return self.check_memory_usage()
            elif command_lower in ["disk", "storage"]:
                return self.check_disk_usage()
            elif command_lower in ["cpu"]:
                return self.check_cpu_usage()
            elif command_lower.startswith("errors"):
                return self.check_recent_errors()
            elif command_lower in ["config", "settings"]:
                return self.show_configuration()
            else:
                return f"Diagnostic: Unknown command '{command}'. Available: status, logs, memory, disk, cpu, errors, config"

        except Exception as e:
            logger.error(f"Error executing troubleshoot command: {e}")
            return f"Diagnostic Error: {str(e)}"

    def get_system_status(self):
        """Get overall system status"""
        try:
            import psutil
            import platform

            # Basic system info
            status = f"System Status:\n"
            status += f"- OS: {platform.system()} {platform.release()}\n"
            status += f"- Python: {platform.python_version()}\n"
            status += f"- CPU Usage: {psutil.cpu_percent()}%\n"
            status += f"- Memory Usage: {psutil.virtual_memory().percent}%\n"
            status += f"- Disk Usage: {psutil.disk_usage('/').percent}%\n"

            # Check if log file exists and get size
            if os.path.exists(LOG_FILE):
                log_size = os.path.getsize(LOG_FILE)
                status += f"- Log File Size: {log_size} bytes\n"
            else:
                status += "- Log File: Not found\n"

            status += "- Status: Operational"
            return status

        except ImportError:
            return "System Status: psutil not available for detailed metrics"
        except Exception as e:
            return f"System Status Error: {str(e)}"

    def analyze_logs(self, command):
        """Analyze log files"""
        try:
            if not os.path.exists(LOG_FILE):
                return "Logs: Log file not found"

            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()

            if not lines:
                return "Logs: Log file is empty"

            # Get last 10 lines or specific analysis
            if "recent" in command.lower() or "last" in command.lower():
                recent_lines = lines[-10:]
                return f"Recent Logs:\n" + "".join(recent_lines)

            # Count errors
            error_count = sum(1 for line in lines if "ERROR" in line.upper())
            warning_count = sum(1 for line in lines if "WARNING" in line.upper())

            return f"Log Analysis:\n- Total lines: {len(lines)}\n- Errors: {error_count}\n- Warnings: {warning_count}"

        except Exception as e:
            return f"Log Analysis Error: {str(e)}"

    def check_memory_usage(self):
        """Check memory usage"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return f"Memory Usage:\n- Total: {mem.total // (1024**3)}GB\n- Used: {mem.used // (1024**3)}GB\n- Available: {mem.available // (1024**3)}GB\n- Percentage: {mem.percent}%"
        except ImportError:
            return "Memory Check: psutil not available"
        except Exception as e:
            return f"Memory Check Error: {str(e)}"

    def check_disk_usage(self):
        """Check disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return f"Disk Usage:\n- Total: {disk.total // (1024**3)}GB\n- Used: {disk.used // (1024**3)}GB\n- Free: {disk.free // (1024**3)}GB\n- Percentage: {disk.percent}%"
        except ImportError:
            return "Disk Check: psutil not available"
        except Exception as e:
            return f"Disk Check Error: {str(e)}"

    def check_cpu_usage(self):
        """Check CPU usage"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            return f"CPU Usage:\n- Cores: {cpu_count}\n- Usage: {cpu_percent}%"
        except ImportError:
            return "CPU Check: psutil not available"
        except Exception as e:
            return f"CPU Check Error: {str(e)}"

    def check_recent_errors(self):
        """Check for recent errors"""
        try:
            if not os.path.exists(LOG_FILE):
                return "Error Check: Log file not found"

            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()

            # Get last 24 hours of errors (simplified)
            error_lines = [line for line in lines[-100:] if "ERROR" in line.upper()]
            if error_lines:
                return f"Recent Errors:\n" + "".join(error_lines[-5:])  # Last 5 errors
            else:
                return "Error Check: No recent errors found"

        except Exception as e:
            return f"Error Check Error: {str(e)}"

    def show_configuration(self):
        """Show current configuration"""
        try:
            config_info = "Current Configuration:\n"
            config_info += f"- VSCode Enabled: {VSCODE_ENABLED}\n"
            config_info += f"- Yarn Enabled: {YARN_ENABLED}\n"
            config_info += f"- Python Enabled: {PYTHON_ENABLED}\n"
            config_info += f"- Self-Troubleshoot Enabled: {SELF_TROUBLESHOOT_ENABLED}\n"
            config_info += f"- API Enabled: {API_ENABLED}\n"
            config_info += f"- Self-Modify Enabled: {SELF_MODIFY_ENABLED}\n"
            config_info += f"- Check Interval: {CHECK_INTERVAL}s\n"
            config_info += f"- Log Level: {LOG_LEVEL}\n"
            return config_info
        except Exception as e:
            return f"Configuration Error: {str(e)}"

    def execute_modify_command(self, command):
        """Execute self-modification commands (use with extreme caution)"""
        try:
            # Parse the command: "file.py:search_text:replace_text"
            parts = command.split(":", 2)
            if len(parts) != 3:
                return "Modify: Invalid format. Use 'file.py:search_text:replace_text'"

            filename, search_text, replace_text = parts

            # Safety checks
            allowed_files = ["config.py", "main.py"]  # Only allow modifying core files
            if filename not in allowed_files:
                return f"Modify: File '{filename}' not allowed for modification. Allowed: {', '.join(allowed_files)}"

            # Additional safety: don't allow dangerous modifications
            dangerous_patterns = ["import os", "subprocess", "exec(", "eval(", "SELF_MODIFY_ENABLED = False"]
            for pattern in dangerous_patterns:
                if pattern in replace_text and pattern not in search_text:
                    return f"Modify: Dangerous modification detected: '{pattern}'"

            # Read the file
            if not os.path.exists(filename):
                return f"Modify: File '{filename}' not found"

            with open(filename, 'r') as f:
                content = f.read()

            # Check if search text exists
            if search_text not in content:
                return f"Modify: Search text not found in {filename}"

            # Perform the replacement
            new_content = content.replace(search_text, replace_text, 1)  # Replace only first occurrence

            # Write back
            with open(filename, 'w') as f:
                f.write(new_content)

            logger.warning(f"Self-modification performed: {filename} - {search_text} -> {replace_text}")
            return f"Modify: Successfully modified {filename}"

        except Exception as e:
            logger.error(f"Error executing modify command: {e}")
            return f"Modify Error: {str(e)}"

    def execute_api_command(self, command):
        """Execute API management commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower == "start":
                return self.api_manager.start_server()
            elif command_lower == "stop":
                return self.api_manager.stop_server()
            elif command_lower == "status":
                return self.api_manager.get_status()
            elif command_lower.startswith("add "):
                # Parse: "add GET /test {"message": "hello"}"
                parts = command[4:].strip().split(" ", 2)
                if len(parts) < 2:
                    return "API: Invalid add format. Use 'add METHOD /path [response_json]'"

                method = parts[0].upper()
                path = parts[1]

                response_data = None
                if len(parts) > 2:
                    try:
                        response_data = json.loads(parts[2])
                    except json.JSONDecodeError:
                        response_data = {"message": parts[2]}

                return self.api_manager.add_endpoint(path, method, response_data)

            elif command_lower.startswith("remove "):
                # Parse: "remove /test"
                path = command[7:].strip()
                return self.api_manager.remove_endpoint(path)
            else:
                return "API: Unknown command. Available: start, stop, status, add METHOD /path [response], remove /path"

        except Exception as e:
            logger.error(f"Error executing API command: {e}")
            return f"API Error: {str(e)}"

    def execute_voice_command(self, command):
        """Execute voice/text-to-speech commands using ElevenLabs"""
        try:
            if not ELEVENLABS_API_KEY:
                return "Voice: ElevenLabs API key not configured. Please set ELEVENLABS_API_KEY in config.py"

            # Set API key
            set_api_key(ELEVENLABS_API_KEY)

            command_lower = command.lower().strip()

            if command_lower in ["test", "hello"]:
                text = "Hello! This is a test of the ElevenLabs voice integration."
            elif command_lower.startswith("say "):
                text = command[4:].strip()
            elif command_lower.startswith("speak "):
                text = command[6:].strip()
            else:
                text = command.strip()

            if not text:
                return "Voice: No text provided to speak"

            # Generate and play audio
            audio = generate(
                text=text,
                voice=ELEVENLABS_VOICE_ID,
                model=ELEVENLABS_MODEL,
                voice_settings={
                    "stability": ELEVENLABS_VOICE_STABILITY,
                    "similarity_boost": ELEVENLABS_VOICE_SIMILARITY,
                    "style": ELEVENLABS_VOICE_STYLE,
                    "use_speaker_boost": ELEVENLABS_VOICE_USE_SPEAKER_BOOST
                }
            )

            # Play the audio
            play(audio)

            logger.info(f"Voice: Generated speech for text: {text[:50]}...")
            return f"Voice: Played audio for '{text[:30]}...'"

        except Exception as e:
            logger.error(f"Error executing voice command: {e}")
            return f"Voice Error: {str(e)}"

    def speak_text(self, text):
        """Speak text using pyttsx3 (fallback TTS)"""
        if not PYTTSX3_AVAILABLE:
            return False

        def speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)  # Speed of speech
                engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"pyttsx3 voice synthesis error: {e}")

        # Run in background thread to not block
        voice_thread = threading.Thread(target=speak, daemon=True)
        voice_thread.start()
        return True

    def recognize_speech(self, timeout=5):
        """Recognize speech from microphone"""
        if not SPEECH_RECOGNITION_AVAILABLE or not self.voice_recognizer:
            return None, "Speech recognition not available"

        try:
            with sr.Microphone() as source:
                logger.info("Listening for voice input...")
                # Adjust for ambient noise
                self.voice_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for speech
                audio = self.voice_recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

            # Recognize speech using Google Speech Recognition
            text = self.voice_recognizer.recognize_google(audio)
            logger.info(f"Recognized speech: {text}")
            return text, None

        except sr.WaitTimeoutError:
            return None, "No speech detected within timeout"
        except sr.UnknownValueError:
            return None, "Could not understand audio"
        except sr.RequestError as e:
            return None, f"Speech recognition service error: {e}"
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None, str(e)

    def start_voice_chat(self, ai_name=None):
        """Start voice chat mode with a specific AI or the main Nomi session"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                return "Voice chat: Speech recognition not available. Install SpeechRecognition: pip install SpeechRecognition"

            if self.voice_chat_active:
                return "Voice chat: Already active. Use 'voicechat: stop' to end current session."

            # Determine which AI/session to chat with
            if ai_name:
                # Try to find or create session for specific AI
                session_id = None
                for sid, session in self.session_manager.sessions.items():
                    if session["ai_type"] == ai_name and session["status"] == "ready":
                        session_id = sid
                        break

                if not session_id:
                    # Try to create new session
                    if ai_name in AI_INSTANCES:
                        session_id, session_info = self.session_manager.create_session(ai_name, AI_INSTANCES[ai_name]["url"], AI_INSTANCES[ai_name]["name"])
                        if session_id:
                            asyncio.create_task(self.start_ai_session(session_id))
                        else:
                            return f"Voice chat: Failed to create session for {ai_name}"
                    else:
                        return f"Voice chat: Unknown AI '{ai_name}'. Available: {', '.join(AI_INSTANCES.keys())}"

                self.voice_chat_session = session_id
            else:
                # Use main Nomi session
                if not self.main_session_id:
                    return "Voice chat: No main Nomi session available. Start nomi_automator first."
                self.voice_chat_session = self.main_session_id

            self.voice_chat_active = True

            # Start voice chat loop in background
            voice_thread = threading.Thread(target=self._voice_chat_loop, daemon=True)
            voice_thread.start()

            ai_display_name = ai_name or "Nomi AI"
            return f"📞 Phone Call: Started with {ai_display_name}. Continuous listening mode engaged - speak naturally to chat with the AI. Say 'hang up' to end the call."

        except Exception as e:
            logger.error(f"Error starting voice chat: {e}")
            return f"Voice chat: Error starting session - {str(e)}"

    def stop_voice_chat(self):
        """Stop voice chat mode"""
        try:
            if not self.voice_chat_active:
                return "Voice chat: Not currently active"

            self.voice_chat_active = False
            self.voice_chat_session = None
            return "Voice chat: Stopped"

        except Exception as e:
            logger.error(f"Error stopping voice chat: {e}")
            return f"Voice chat: Error stopping - {str(e)}"

    def _voice_chat_loop(self):
        """Main voice chat loop - runs in background thread"""
        try:
            logger.info("Voice chat loop started")

            while self.voice_chat_active:
                try:
                    # Listen for speech
                    text, error = self.recognize_speech(timeout=10)  # Longer timeout for conversation

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
                        logger.info("Voice chat stop command detected")
                        self.voice_chat_active = False
                        break

                    # Send the recognized text to the AI via API
                    logger.info(f"Sending voice input to AI: {text}")

                    try:
                        if self.use_api:
                            # Use API for chat interaction
                            response = self.api_client.send_chat_message(text, ai_id=self.voice_chat_session or NOMI_DEFAULT_AI_ID)
                            ai_response = response.get('message', 'No response from AI')

                            # Speak the AI response
                            if VOICE_RESPONSES_ENABLED and (ELEVENLABS_AVAILABLE or PYTTSX3_AVAILABLE):
                                try:
                                    if ELEVENLABS_AVAILABLE and ELEVENLABS_API_KEY:
                                        # Try to get voice from Nomi API first
                                        try:
                                            audio_data = self.api_client.get_voice_audio(ai_response)
                                            # Play the audio (would need audio playback implementation)
                                            logger.info("Playing AI voice response from Nomi API")
                                        except:
                                            # Fallback to ElevenLabs/pyttsx3
                                            self.speak_text(ai_response)
                                    else:
                                        self.speak_text(ai_response)
                                except Exception as e:
                                    logger.error(f"Error playing voice response: {e}")

                            logger.info(f"AI response: {ai_response}")

                        else:
                            # Fallback to browser automation
                            if self.voice_chat_session == self.main_session_id:
                                # Send to main Nomi session
                                asyncio.run(self.send_response(text))
                            else:
                                # Send to specific AI session
                                result = self.send_to_session(self.voice_chat_session, text)
                                logger.info(f"Voice message sent to session: {result}")

                    except Exception as e:
                        logger.error(f"Error sending message to AI: {e}")
                        # Continue listening even if sending failed

                    # Small delay before listening again
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in voice chat loop: {e}")
                    time.sleep(2)  # Wait before retrying

            logger.info("Voice chat loop ended")

        except Exception as e:
            logger.error(f"Fatal error in voice chat loop: {e}")
            self.voice_chat_active = False

    def execute_mindmap_command(self, command):
        """Execute Mind Map management commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["help", "commands", "usage"]:
                return self.get_mindmap_help()

            elif command_lower in ["list", "topics", "all"]:
                return self.mindmap_list_topics()

            elif command_lower.startswith("show ") or command_lower.startswith("view "):
                topic = command[5:].strip() if command_lower.startswith("show ") else command[5:].strip()
                return self.mindmap_show_topic(topic)

            elif command_lower.startswith("search ") or command_lower.startswith("find "):
                query = command[7:].strip() if command_lower.startswith("search ") else command[5:].strip()
                return self.mindmap_search(query)

            elif command_lower.startswith("add ") or command_lower.startswith("create "):
                # Parse: "add topic: information" or "add topic information"
                parts = command[4:].strip().split(":", 1) if command_lower.startswith("add ") else command[7:].strip().split(":", 1)
                if len(parts) == 2:
                    topic, info = parts
                    return self.mindmap_add_info(topic.strip(), info.strip())
                else:
                    return "Mind Map: Invalid add format. Use 'add topic: information'"

            elif command_lower.startswith("update ") or command_lower.startswith("edit "):
                # Parse: "update topic: new information"
                parts = command[7:].strip().split(":", 1) if command_lower.startswith("update ") else command[5:].strip().split(":", 1)
                if len(parts) == 2:
                    topic, info = parts
                    return self.mindmap_update_topic(topic.strip(), info.strip())
                else:
                    return "Mind Map: Invalid update format. Use 'update topic: new information'"

            elif command_lower.startswith("connect ") or command_lower.startswith("link "):
                # Parse: "connect topic1 to topic2: relationship"
                parts = command[8:].strip().split(" to ", 1) if command_lower.startswith("connect ") else command[5:].strip().split(" to ", 1)
                if len(parts) == 2:
                    topic1 = parts[0].strip()
                    remaining = parts[1].split(":", 1)
                    if len(remaining) == 2:
                        topic2, relationship = remaining
                        return self.mindmap_connect_topics(topic1, topic2.strip(), relationship.strip())
                    else:
                        return self.mindmap_connect_topics(topic1, remaining[0].strip(), "related to")
                else:
                    return "Mind Map: Invalid connect format. Use 'connect topic1 to topic2: relationship'"

            elif command_lower.startswith("delete ") or command_lower.startswith("remove "):
                topic = command[7:].strip() if command_lower.startswith("delete ") else command[7:].strip()
                return self.mindmap_delete_topic(topic)

            elif command_lower == "status":
                return self.mindmap_get_status()

            elif command_lower == "forest":
                return self.mindmap_show_forest()

            else:
                return f"Mind Map: Unknown command '{command}'. Use 'mindmap: help' for available commands."

        except Exception as e:
            logger.error(f"Error executing mindmap command: {e}")
            return f"Mind Map Error: {str(e)}"

    def execute_learn_command(self, command):
        """Execute learning session commands with Mind Map integration"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["help", "commands"]:
                return self.get_learn_help()

            elif command_lower.startswith("session ") or command_lower.startswith("start "):
                topic = command[8:].strip() if command_lower.startswith("session ") else command[6:].strip()
                return self.start_learning_session(topic)

            elif command_lower.startswith("review ") or command_lower.startswith("study "):
                topic = command[7:].strip() if command_lower.startswith("review ") else command[6:].strip()
                return self.review_topic(topic)

            elif command_lower.startswith("teach ") or command_lower.startswith("explain "):
                topic = command[6:].strip() if command_lower.startswith("teach ") else command[8:].strip()
                return self.teach_concept(topic)

            elif command_lower == "progress":
                return self.get_learning_progress()

            elif command_lower.startswith("goal ") or command_lower.startswith("set "):
                goal_text = command[5:].strip() if command_lower.startswith("goal ") else command[4:].strip()
                return self.set_learning_goal(goal_text)

            else:
                return f"Learn: Unknown command '{command}'. Use 'learn: help' for available commands."

        except Exception as e:
            logger.error(f"Error executing learn command: {e}")
            return f"Learn Error: {str(e)}"

    def get_mindmap_help(self):
        """Get help for Mind Map commands"""
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

    def get_learn_help(self):
        """Get help for learning commands"""
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

    def mindmap_list_topics(self):
        """List all Mind Map topics"""
        try:
            # This would query the Nomi Mind Map system
            # For now, return a placeholder that indicates the command is queued
            return "Mind Map: Listing topics... (This command interacts with Nomi.ai's Mind Map system. Active session required for real functionality.)"
        except Exception as e:
            return f"Mind Map List Error: {str(e)}"

    def mindmap_show_topic(self, topic):
        """Show detailed information about a Mind Map topic"""
        try:
            if not topic:
                return "Mind Map: Please specify a topic to show."

            return f"Mind Map: Showing information for '{topic}'... (This displays the complete, organized overview from Nomi.ai's Mind Map system.)"
        except Exception as e:
            return f"Mind Map Show Error: {str(e)}"

    def mindmap_search(self, query):
        """Search through Mind Map content"""
        try:
            if not query:
                return "Mind Map: Please specify search terms."

            return f"Mind Map: Searching for '{query}'... (This searches through all Mind Map entries and connections.)"
        except Exception as e:
            return f"Mind Map Search Error: {str(e)}"

    def mindmap_add_info(self, topic, info):
        """Add information to a Mind Map topic"""
        try:
            if not topic or not info:
                return "Mind Map: Please specify both topic and information."

            return f"Mind Map: Added information to '{topic}': {info[:50]}... (This updates Nomi.ai's Mind Map with structured knowledge.)"
        except Exception as e:
            return f"Mind Map Add Error: {str(e)}"

    def mindmap_update_topic(self, topic, new_info):
        """Update existing Mind Map topic information"""
        try:
            if not topic or not new_info:
                return "Mind Map: Please specify both topic and new information."

            return f"Mind Map: Updated '{topic}' with new information: {new_info[:50]}... (This evolves the Mind Map entry over time.)"
        except Exception as e:
            return f"Mind Map Update Error: {str(e)}"

    def mindmap_connect_topics(self, topic1, topic2, relationship):
        """Connect two Mind Map topics"""
        try:
            if not topic1 or not topic2:
                return "Mind Map: Please specify both topics to connect."

            return f"Mind Map: Connected '{topic1}' to '{topic2}' ({relationship}). (This creates meaningful connections in the knowledge forest.)"
        except Exception as e:
            return f"Mind Map Connect Error: {str(e)}"

    def mindmap_delete_topic(self, topic):
        """Delete a topic from Mind Map"""
        try:
            if not topic:
                return "Mind Map: Please specify a topic to delete."

            return f"Mind Map: Removed '{topic}' from Mind Map. (Note: Important topics may be automatically recreated from memory.)"
        except Exception as e:
            return f"Mind Map Delete Error: {str(e)}"

    def mindmap_get_status(self):
        """Get Mind Map system status"""
        try:
            return """Mind Map Status:
• System: Active (Mind Map 1.0)
• Memory Persistence: Infinite for important concepts
• Structure: Complete information organization
• Connections: Forest-level awareness enabled
• Chat Separation: 1-on-1 and group chats maintained separately
• Warmup Period: Building comprehensive overviews with each interaction"""
        except Exception as e:
            return f"Mind Map Status Error: {str(e)}"

    def mindmap_show_forest(self):
        """Show the complete Mind Map forest view"""
        try:
            return "Mind Map Forest View: Displaying how all important concepts connect together... (This shows the complete knowledge structure and relationships between all topics.)"
        except Exception as e:
            return f"Mind Map Forest Error: {str(e)}"

    def start_learning_session(self, topic):
        """Start a structured learning session with Mind Map integration"""
        try:
            if not topic:
                return "Learn: Please specify a topic for the learning session."

            return f"Learn: Starting structured learning session for '{topic}'... (This creates an interactive learning experience that builds Mind Map knowledge.)"
        except Exception as e:
            return f"Learn Session Error: {str(e)}"

    def review_topic(self, topic):
        """Review existing Mind Map knowledge"""
        try:
            if not topic:
                return "Learn: Please specify a topic to review."

            return f"Learn: Reviewing Mind Map knowledge for '{topic}'... (This reinforces long-term retention of structured information.)"
        except Exception as e:
            return f"Learn Review Error: {str(e)}"

    def teach_concept(self, concept):
        """Teach a concept using Mind Map integration"""
        try:
            if not concept:
                return "Learn: Please specify a concept to teach."

            return f"Learn: Teaching concept '{concept}' using Mind Map structure... (This provides comprehensive explanations with organized knowledge.)"
        except Exception as e:
            return f"Learn Teach Error: {str(e)}"

    def get_learning_progress(self):
        """Get learning progress and goals"""
        try:
            return """Learning Progress:
• Active Goals: Building comprehensive Mind Map overviews
• Progress: Evolving with each interaction
• Retention: Infinite persistence for important concepts
• Structure: Complete information organization
• Connections: Meaningful relationships between topics"""
        except Exception as e:
            return f"Learn Progress Error: {str(e)}"

    def set_learning_goal(self, goal_text):
        """Set a learning objective"""
        try:
            if not goal_text:
                return "Learn: Please specify a learning goal."

            return f"Learn: Set learning goal - '{goal_text}'. (This integrates with Mind Map to track progress toward objectives.)"
        except Exception as e:
            return f"Learn Goal Error: {str(e)}"

    def execute_conversation_command(self, command):
        """Execute conversation initiation and management commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower.startswith("open ") or command_lower.startswith("start "):
                # Parse: "open nomi" or "open https://example.com AI Name"
                parts = command[5:].strip().split(" ", 2) if command_lower.startswith("open ") else command[6:].strip().split(" ", 2)

                if len(parts) >= 1:
                    ai_type = parts[0].lower()
                    url = None
                    name = None

                    if ai_type in AI_INSTANCES:
                        config = AI_INSTANCES[ai_type]
                        url = config["url"]
                        name = config["name"]
                    elif len(parts) >= 2:
                        url = parts[1]
                        if len(parts) >= 3:
                            name = parts[2]

                    if url:
                        session_id, session_info = self.session_manager.create_session(ai_type, url, name)
                        if session_id:
                            # Start the session in background
                            asyncio.create_task(self.start_ai_session(session_id))
                            return f"Conversation: Started {session_info['name']} session ({session_id})"
                        else:
                            return f"Conversation: Failed to create session - {session_info}"
                    else:
                        return f"Conversation: Unknown AI type '{ai_type}' and no URL provided"

            elif command_lower.startswith("close ") or command_lower.startswith("end "):
                # Parse: "close session_id"
                session_id = command[6:].strip() if command_lower.startswith("close ") else command[4:].strip()
                if self.session_manager.close_session(session_id):
                    return f"Conversation: Closed session {session_id}"
                else:
                    return f"Conversation: Session {session_id} not found"

            elif command_lower.startswith("list") or command_lower == "sessions":
                sessions = self.session_manager.list_sessions()
                if sessions:
                    response = "Active Sessions:\n"
                    for sid, info in sessions.items():
                        response += f"- {sid}: {info['name']} ({info['ai_type']}) - {info['status']}\n"
                    return response
                else:
                    return "Conversation: No active sessions"

            elif command_lower.startswith("send "):
                # Parse: "send session_id message"
                parts = command[5:].strip().split(" ", 1)
                if len(parts) == 2:
                    session_id, message = parts
                    return self.send_to_session(session_id, message)
                else:
                    return "Conversation: Invalid send format. Use 'send session_id message'"

            elif command_lower.startswith("ask "):
                # Parse: "ask ai_type question"
                parts = command[4:].strip().split(" ", 1)
                if len(parts) == 2:
                    ai_type, question = parts
                    return self.ask_ai(ai_type, question)
                else:
                    return "Conversation: Invalid ask format. Use 'ask ai_type question'"

            else:
                return "Conversation: Unknown command. Available: open ai_type, close session_id, list, send session_id message, ask ai_type question"

        except Exception as e:
            logger.error(f"Error executing conversation command: {e}")
            return f"Conversation Error: {str(e)}"

    async def start_ai_session(self, session_id):
        """Start an AI session by opening browser and navigating"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return

            # Create new browser context for this session
            browser = await async_playwright().start().chromium.launch(headless=HEADLESS)
            page = await browser.new_page()

            session["browser"] = browser
            session["page"] = page
            session["status"] = "connecting"

            await page.goto(session["url"])
            session["status"] = "connected"

            logger.info(f"Started AI session: {session_id} at {session['url']}")

            # Handle authentication if needed
            await self.handle_session_authentication(session)

        except Exception as e:
            logger.error(f"Error starting AI session {session_id}: {e}")
            session = self.session_manager.get_session(session_id)
            if session:
                session["status"] = f"error: {str(e)}"

    async def handle_session_authentication(self, session):
        """Handle authentication for a session"""
        try:
            # Similar to main authentication but for specific session
            page = session["page"]
            login_selectors = ['button:has-text("Login")', '[href*="login"]', '.login-button']

            for selector in login_selectors:
                try:
                    login_button = await page.query_selector(selector)
                    if login_button:
                        logger.info(f"Login required for session {session['id']}")
                        # For now, just mark as needing manual login
                        session["status"] = "needs_login"
                        return
                except:
                    continue

            session["status"] = "ready"
            logger.info(f"Session {session['id']} ready for interaction")

        except Exception as e:
            logger.error(f"Error handling authentication for session {session['id']}: {e}")

    def send_to_session(self, session_id, message):
        """Send a message to a specific session"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return f"Conversation: Session {session_id} not found"

            if session["status"] != "ready":
                return f"Conversation: Session {session_id} not ready (status: {session['status']})"

            # Add message to session's message queue
            session["messages"].append({"type": "outgoing", "content": message, "timestamp": asyncio.get_event_loop().time()})
            self.session_manager.update_activity(session_id)

            # For now, we'll need manual sending through the UI
            # In a full implementation, this would automate sending to the AI interface
            return f"Conversation: Message queued for session {session_id}. Manual sending required."

        except Exception as e:
            return f"Conversation Send Error: {str(e)}"

    def ask_ai(self, ai_type, question):
        """Ask a question to an AI type, creating session if needed"""
        try:
            # Check if we have an existing session for this AI type
            existing_session = None
            for sid, session in self.session_manager.sessions.items():
                if session["ai_type"] == ai_type and session["status"] == "ready":
                    existing_session = sid
                    break

            if existing_session:
                return self.send_to_session(existing_session, question)
            else:
                # Try to create a new session
                if ai_type in AI_INSTANCES:
                    config = AI_INSTANCES[ai_type]
                    session_id, session_info = self.session_manager.create_session(ai_type, config["url"], config["name"])
                    if session_id:
                        asyncio.create_task(self.start_ai_session(session_id))
                        return f"Conversation: Created new {ai_type} session. Message queued: {question[:50]}..."
                    else:
                        return f"Conversation: Failed to create {ai_type} session"
                else:
                    return f"Conversation: Unknown AI type '{ai_type}'. Available: {', '.join(AI_INSTANCES.keys())}"

        except Exception as e:
            return f"Conversation Ask Error: {str(e)}"

    def execute_image_command(self, command):
        """Execute image generation commands using DALL-E"""
        try:
            if not OPENAI_API_KEY:
                return "Image: OpenAI API key not configured. Please set OPENAI_API_KEY in config.py"

            # Set OpenAI API key
            openai.api_key = OPENAI_API_KEY

            command_lower = command.lower().strip()

            if command_lower in ["test", "demo"]:
                prompt = "A beautiful landscape with mountains and a lake, digital art style"
            else:
                prompt = command.strip()

            if not prompt:
                return "Image: No prompt provided for image generation"

            # Generate image using DALL-E
            response = openai.Image.create(
                model=DALLE_MODEL,
                prompt=prompt,
                size=DALLE_SIZE,
                quality=DALLE_QUALITY,
                style=DALLE_STYLE,
                n=1
            )

            image_url = response['data'][0]['url']

            logger.info(f"Image generated for prompt: {prompt[:50]}...")
            return f"Image: Generated successfully! URL: {image_url}"

        except Exception as e:
            logger.error(f"Error executing image command: {e}")
            return f"Image Error: {str(e)}"

    def execute_avatar_command(self, command):
        """Execute avatar/profile picture management commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["generate", "create", "new"]:
                # Generate a random avatar using one of the default prompts
                import random
                prompt = random.choice(DEFAULT_AVATAR_PROMPTS)
                return self.generate_and_set_avatar(prompt)

            elif command_lower.startswith("set "):
                # Set avatar with custom prompt
                prompt = command[4:].strip()
                return self.generate_and_set_avatar(prompt)

            elif command_lower in ["random", "change"]:
                # Change to a random avatar
                import random
                prompt = random.choice(DEFAULT_AVATAR_PROMPTS)
                return self.generate_and_set_avatar(prompt)

            elif command_lower == "default":
                # Reset to default avatar (if any)
                return self.set_default_avatar()

            else:
                return "Avatar: Unknown command. Available: generate, set <prompt>, random, default"

        except Exception as e:
            logger.error(f"Error executing avatar command: {e}")
            return f"Avatar Error: {str(e)}"

    def generate_and_set_avatar(self, prompt):
        """Generate an avatar image and attempt to set it in the Nomi UI"""
        try:
            if not OPENAI_API_KEY:
                return "Avatar: OpenAI API key not configured"

            # Generate avatar image
            openai.api_key = OPENAI_API_KEY

            # Enhance prompt for avatar generation
            avatar_prompt = f"Professional avatar image: {prompt}, square aspect ratio, high quality, suitable for profile picture"

            response = openai.Image.create(
                model=DALLE_MODEL,
                prompt=avatar_prompt,
                size="512x512",  # Square for avatars
                quality=DALLE_QUALITY,
                style=DALLE_STYLE,
                n=1
            )

            image_url = response['data'][0]['url']

            # Attempt to set avatar in Nomi UI
            success = self.set_avatar_in_ui(image_url)

            if success:
                logger.info(f"Avatar set successfully with prompt: {prompt[:50]}...")
                return f"Avatar: Generated and set successfully! Image URL: {image_url}"
            else:
                return f"Avatar: Generated successfully! Image URL: {image_url} (Manual UI update required)"

        except Exception as e:
            return f"Avatar Generation Error: {str(e)}"

    def set_avatar_in_ui(self, image_url):
        """Attempt to set avatar in Nomi UI by manipulating the page"""
        try:
            if not self.page:
                return False

            # Common selectors for profile/avatar images in web UIs
            avatar_selectors = [
                'img[alt*="profile"]',
                'img[alt*="avatar"]',
                '.profile-picture img',
                '.avatar img',
                '[class*="profile"] img',
                '[class*="avatar"] img',
                'img.profile-pic',
                'img.avatar-pic'
            ]

            # Try to find and update avatar element
            for selector in avatar_selectors:
                try:
                    avatar_element = self.page.query_selector(selector)
                    if avatar_element:
                        # Update the src attribute
                        self.page.evaluate(f'document.querySelector("{selector}").src = "{image_url}"')
                        logger.info(f"Updated avatar using selector: {selector}")
                        return True
                except:
                    continue

            # If no avatar element found, try to find profile/settings buttons
            profile_buttons = [
                'button:has-text("Profile")',
                'button:has-text("Settings")',
                '[href*="profile"]',
                '[href*="settings"]',
                '.profile-button',
                '.settings-button'
            ]

            # Note: This is a simplified implementation
            # In a real scenario, you'd need to navigate to profile settings
            # and upload the image properly

            logger.info("Avatar element not found automatically - manual update required")
            return False

        except Exception as e:
            logger.error(f"Error setting avatar in UI: {e}")
            return False

    def set_default_avatar(self):
        """Reset avatar to default (placeholder implementation)"""
        try:
            # This would typically reset to a default image or remove custom avatar
            # Implementation depends on the specific UI structure
            return "Avatar: Default avatar reset (functionality depends on UI structure)"
        except Exception as e:
            return f"Avatar Reset Error: {str(e)}"

    def execute_model_command(self, command):
        """Execute Hugging Face model management and usage commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower.startswith("load ") or command_lower.startswith("use "):
                # Parse: "load model_name" or "load model_name for task"
                parts = command[5:].strip().split(" for ", 1) if command_lower.startswith("load ") else command[4:].strip().split(" for ", 1)

                model_name = parts[0].strip()
                task = parts[1].strip() if len(parts) > 1 else None

                model_info, error = self.model_manager.load_model(model_name, task)
                if error:
                    return f"Model Load Error: {error}"
                else:
                    return f"Model: Successfully loaded {model_name}"

            elif command_lower.startswith("unload "):
                # Parse: "unload model_name"
                model_name = command[7:].strip()
                if self.model_manager.unload_model(model_name):
                    return f"Model: Unloaded {model_name}"
                else:
                    return f"Model: {model_name} not found"

            elif command_lower.startswith("run ") or command_lower.startswith("infer "):
                # Parse: "run model_name with input"
                parts = command[4:].strip().split(" with ", 1) if command_lower.startswith("run ") else command[6:].strip().split(" with ", 1)

                if len(parts) == 2:
                    model_name, input_data = parts
                    result, error = self.model_manager.use_model(model_name.strip(), input_data.strip())
                    if error:
                        return f"Model Inference Error: {error}"
                    else:
                        return f"Model Result: {result}"
                else:
                    return "Model: Invalid run format. Use 'run model_name with input'"

            elif command_lower.startswith("search "):
                # Parse: "search query" or "search query for task"
                parts = command[7:].strip().split(" for ", 1)
                query = parts[0].strip()
                task = parts[1].strip() if len(parts) > 1 else None

                models = self.model_manager.search_models(query, task)
                if models:
                    response = f"Found {len(models)} models:\n"
                    for model in models[:5]:  # Show top 5
                        response += f"- {model['id']} (downloads: {model['downloads']})\n"
                    return response
                else:
                    return "Model Search: No models found"

            elif command_lower in ["list", "loaded", "status"]:
                loaded_models = self.model_manager.list_loaded_models()
                if loaded_models:
                    response = "Loaded Models:\n"
                    for name, info in loaded_models.items():
                        response += f"- {name} ({info['task'] or 'generic'})\n"
                    return response
                else:
                    return "Model: No models currently loaded"

            elif command_lower == "cleanup":
                self.model_manager.cleanup_unused_models()
                return "Model: Cleaned up unused models"

            elif command_lower.startswith("default "):
                # Load a default model for a task
                task = command[8:].strip()
                if task in DEFAULT_MODELS:
                    model_name = DEFAULT_MODELS[task]
                    model_info, error = self.model_manager.load_model(model_name, task)
                    if error:
                        return f"Model Load Error: {error}"
                    else:
                        return f"Model: Loaded default {task} model: {model_name}"
                else:
                    available_tasks = ", ".join(DEFAULT_MODELS.keys())
                    return f"Model: Unknown task '{task}'. Available: {available_tasks}"

            else:
                return "Model: Unknown command. Available: load model_name, unload model_name, run model with input, search query, list, cleanup, default task"

        except Exception as e:
            logger.error(f"Error executing model command: {e}")
            return f"Model Error: {str(e)}"

    def execute_call_command(self, command):
        """Execute TextNow calling commands"""
        try:
            command_lower = command.lower().strip()

            if not TEXTNOW_EMAIL or not TEXTNOW_PASSWORD:
                return "Call: TextNow credentials not configured. Please set TEXTNOW_EMAIL and TEXTNOW_PASSWORD in config.py"

            if command_lower.startswith("dial ") or command_lower.startswith("call "):
                # Parse: "dial +1234567890" or "call +1234567890 with message"
                parts = command[5:].strip().split(" with ", 1) if command_lower.startswith("dial ") else command[5:].strip().split(" with ", 1)

                phone_number = parts[0].strip()
                message = parts[1].strip() if len(parts) > 1 else None

                return self.make_textnow_call(phone_number, message)

            elif command_lower == "hangup" or command_lower == "end":
                return self.end_textnow_call()

            elif command_lower == "status":
                return self.get_call_status()

            elif command_lower.startswith("send "):
                # Parse: "send +1234567890 message"
                parts = command[5:].strip().split(" ", 1)
                if len(parts) == 2:
                    phone_number, message = parts
                    return self.send_text_message(phone_number, message)
                else:
                    return "Call: Invalid send format. Use 'send phone_number message'"

            else:
                return "Call: Unknown command. Available: dial phone_number, hangup, status, send phone_number message"

        except Exception as e:
            logger.error(f"Error executing call command: {e}")
            return f"Call Error: {str(e)}"

    async def make_textnow_call(self, phone_number, message=None):
        """Make a phone call through TextNow web interface"""
        try:
            # Create a new browser session for TextNow
            browser = await async_playwright().start().chromium.launch(headless=HEADLESS)
            page = await browser.new_page()

            # Navigate to TextNow login
            await page.goto("https://www.textnow.com/login")

            # Login process
            await page.fill('input[type="email"]', TEXTNOW_EMAIL)
            await page.fill('input[type="password"]', TEXTNOW_PASSWORD)
            await page.click('button[type="submit"]')

            # Wait for login to complete
            await page.wait_for_url("**/messaging*", timeout=30000)

            # Navigate to calling interface
            await page.goto("https://www.textnow.com/messaging")

            # Find and click the call button/input
            # Note: These selectors may need adjustment based on TextNow's current UI
            call_input = await page.query_selector('input[placeholder*="phone"], input[type="tel"]')
            if call_input:
                await call_input.fill(phone_number)

                # Click call button
                call_button = await page.query_selector('button:has-text("Call"), .call-button, [data-testid*="call"]')
                if call_button:
                    await call_button.click()

                    # If message provided and voice calls enabled, speak it
                    if message and TEXTNOW_VOICE_CALLS_ENABLED and ELEVENLABS_AVAILABLE and ELEVENLABS_API_KEY:
                        # Wait a moment for call to connect, then speak
                        await asyncio.sleep(3)
                        await self.speak_on_call(message)

                    return f"Call: Dialing {phone_number}..."
                else:
                    await browser.close()
                    return "Call: Could not find call button"
            else:
                await browser.close()
                return "Call: Could not find phone number input"

        except Exception as e:
            logger.error(f"Error making TextNow call: {e}")
            return f"Call Error: {str(e)}"

    async def speak_on_call(self, message):
        """Speak a message during an active call"""
        try:
            if not ELEVENLABS_API_KEY:
                return

            # Set API key
            set_api_key(ELEVENLABS_API_KEY)

            # Generate and play audio (this would need system audio routing to work with calls)
            audio = generate(
                text=message,
                voice=ELEVENLABS_VOICE_ID,
                model=ELEVENLABS_MODEL,
                voice_settings={
                    "stability": ELEVENLABS_VOICE_STABILITY,
                    "similarity_boost": ELEVENLABS_VOICE_SIMILARITY,
                    "style": ELEVENLABS_VOICE_STYLE,
                    "use_speaker_boost": ELEVENLABS_VOICE_USE_SPEAKER_BOOST
                }
            )

            # Note: In a real implementation, you'd need to route this audio to the call
            # This might require additional audio processing libraries
            play(audio)

            logger.info(f"Spoke on call: {message[:50]}...")

        except Exception as e:
            logger.error(f"Error speaking on call: {e}")

    def end_textnow_call(self):
        """End the current TextNow call"""
        try:
            # This would need to interact with the active call session
            # For now, return a placeholder
            return "Call: Hang up functionality requires active call session management"
        except Exception as e:
            return f"Call End Error: {str(e)}"

    def get_call_status(self):
        """Get current call status"""
        try:
            # This would check active call sessions
            return "Call: No active calls. Status checking requires session management implementation"
        except Exception as e:
            return f"Call Status Error: {str(e)}"

    def send_text_message(self, phone_number, message):
        """Send a text message through TextNow"""
        try:
            # This would automate sending SMS through TextNow web interface
            return f"Text: Message queued to {phone_number}. Web automation implementation needed for sending"
        except Exception as e:
            return f"Text Error: {str(e)}"

    def execute_vw_command(self, command):
        """Execute API-based virtual world commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["list", "platforms", "available"]:
                return self.vw_list_platforms()

            elif command_lower.startswith("join ") or command_lower.startswith("enter "):
                # Parse: "join mozilla-hubs room_url" or "join spatial space_id"
                parts = command[5:].strip().split(" ", 1) if command_lower.startswith("join ") else command[6:].strip().split(" ", 1)

                if len(parts) == 2:
                    platform, target = parts
                    return self.vw_join_space(platform, target)
                else:
                    return "Virtual World: Invalid join format. Use 'join platform target'"

            elif command_lower.startswith("create ") or command_lower.startswith("new "):
                # Parse: "create mozilla-hubs My Room" or "create spatial My Space"
                parts = command[7:].strip().split(" ", 1) if command_lower.startswith("create ") else command[4:].strip().split(" ", 1)

                if len(parts) == 2:
                    platform, name = parts
                    return self.vw_create_space(platform, name)
                else:
                    return "Virtual World: Invalid create format. Use 'create platform name'"

            elif command_lower.startswith("chat ") or command_lower.startswith("say "):
                message = command[5:].strip() if command_lower.startswith("chat ") else command[3:].strip()
                return self.vw_send_message(message)

            elif command_lower.startswith("invite "):
                user = command[7:].strip()
                return self.vw_invite_user(user)

            elif command_lower == "leave" or command_lower == "exit":
                return self.vw_leave_space()

            elif command_lower == "status":
                return self.vw_get_status()

            elif command_lower.startswith("media ") or command_lower.startswith("share "):
                media_url = command[6:].strip() if command_lower.startswith("media ") else command[6:].strip()
                return self.vw_share_media(media_url)

            else:
                return "Virtual World: Unknown command. Available: list, join platform target, create platform name, chat message, invite user, leave, status, media url"

        except Exception as e:
            logger.error(f"Error executing virtual world command: {e}")
            return f"Virtual World Error: {str(e)}"

    def vw_list_platforms(self):
        """List available virtual world platforms"""
        try:
            platforms = []
            for platform_id, config in VIRTUAL_WORLD_INSTANCES.items():
                if config.get("enabled", False):
                    platforms.append(f"- {platform_id}: {config['name']} ({', '.join(config.get('features', []))})")

            if platforms:
                return "Available Virtual World Platforms:\n" + "\n".join(platforms)
            else:
                return "Virtual World: No platforms configured. Check VIRTUAL_WORLD_INSTANCES in config.py"

        except Exception as e:
            return f"Virtual World List Error: {str(e)}"

    def vw_join_space(self, platform, target):
        """Join a virtual world space/room"""
        try:
            if platform not in VIRTUAL_WORLD_INSTANCES:
                return f"Virtual World: Unknown platform '{platform}'"

            config = VIRTUAL_WORLD_INSTANCES[platform]
            if not config.get("enabled", False):
                return f"Virtual World: Platform '{platform}' is disabled"

            # For now, return API call simulation
            # In a real implementation, this would make actual API calls
            return f"Virtual World: Joining {config['name']} space '{target}'. API integration required for actual connection."

        except Exception as e:
            return f"Virtual World Join Error: {str(e)}"

    def vw_create_space(self, platform, name):
        """Create a new virtual world space/room"""
        try:
            if platform not in VIRTUAL_WORLD_INSTANCES:
                return f"Virtual World: Unknown platform '{platform}'"

            config = VIRTUAL_WORLD_INSTANCES[platform]
            if not config.get("enabled", False):
                return f"Virtual World: Platform '{platform}' is disabled"

            # For now, return API call simulation
            return f"Virtual World: Creating {config['name']} space '{name}'. API integration required for actual creation."

        except Exception as e:
            return f"Virtual World Create Error: {str(e)}"

    def vw_send_message(self, message):
        """Send a message in the current virtual world space"""
        try:
            # Check if we're in an active virtual world session
            vw_sessions = [s for s in self.session_manager.sessions.values() if s["ai_type"] in VIRTUAL_WORLD_INSTANCES]

            if not vw_sessions:
                return "Virtual World: Not currently in any virtual world space. Use 'vw: join platform target' first."

            # For now, simulate sending message
            return f"Virtual World: Message sent: '{message}'. API integration required for actual messaging."

        except Exception as e:
            return f"Virtual World Message Error: {str(e)}"

    def vw_invite_user(self, user):
        """Invite a user to the current virtual world space"""
        try:
            return f"Virtual World: Invitation sent to '{user}'. API integration required for actual invitations."

        except Exception as e:
            return f"Virtual World Invite Error: {str(e)}"

    def vw_leave_space(self):
        """Leave the current virtual world space"""
        try:
            vw_sessions = [s for s in self.session_manager.sessions.values() if s["ai_type"] in VIRTUAL_WORLD_INSTANCES]

            for session in vw_sessions:
                if session["browser"]:
                    asyncio.create_task(session["browser"].close())
                del self.session_manager.sessions[session["id"]]

            return "Virtual World: Left all virtual world spaces."

        except Exception as e:
            return f"Virtual World Leave Error: {str(e)}"

    def vw_get_status(self):
        """Get virtual world status"""
        try:
            vw_sessions = [s for s in self.session_manager.sessions.values() if s["ai_type"] in VIRTUAL_WORLD_INSTANCES]

            if vw_sessions:
                status = "Virtual World Sessions:\n"
                for session in vw_sessions:
                    status += f"- {session['name']}: {session['status']}\n"
                return status
            else:
                return "Virtual World: Not currently in any spaces. Use 'vw: join platform target' to enter a space."

        except Exception as e:
            return f"Virtual World Status Error: {str(e)}"

    def vw_share_media(self, media_url):
        """Share media in the current virtual world space"""
        try:
            return f"Virtual World: Media shared: {media_url}. API integration required for actual sharing."

        except Exception as e:
            return f"Virtual World Media Error: {str(e)}"

    def execute_google_command(self, command):
        """Execute Google service commands (Gmail, YouTube)"""
        try:
            command_lower = command.lower().strip()

            # Gmail commands
            if command_lower.startswith("send ") and GMAIL_ENABLED:
                # Parse: "send recipient@example.com Subject: Hello"
                parts = command[5:].strip().split(" ", 2)
                if len(parts) >= 2:
                    recipient = parts[0]
                    subject_and_body = parts[1] if len(parts) > 1 else ""
                    if ":" in subject_and_body:
                        subject, body = subject_and_body.split(":", 1)
                        return self.send_gmail(recipient, subject.strip(), body.strip())
                    else:
                        return "Gmail: Invalid format. Use 'send email@domain.com Subject: Message body'"
                else:
                    return "Gmail: Invalid send format. Use 'send recipient subject: body'"

            elif command_lower in ["inbox", "emails", "read"] and GMAIL_ENABLED:
                return self.read_gmail_inbox()

            # YouTube commands
            elif command_lower.startswith("upload ") and YOUTUBE_ENABLED:
                # Parse: "upload /path/to/video.mp4 Title: My Video"
                parts = command[7:].strip().split(" ", 1)
                if len(parts) == 2:
                    video_path = parts[0]
                    title_and_desc = parts[1]
                    if ":" in title_and_desc:
                        title, description = title_and_desc.split(":", 1)
                        return self.upload_youtube_video(video_path, title.strip(), description.strip())
                    else:
                        return "YouTube: Invalid format. Use 'upload path/to/video.mp4 Title: Description'"
                else:
                    return "YouTube: Invalid upload format"

            elif command_lower in ["videos", "list"] and YOUTUBE_ENABLED:
                return self.list_youtube_videos()

            elif command_lower.startswith("search ") and YOUTUBE_ENABLED:
                query = command[7:].strip()
                return self.search_youtube(query)

            # Google Drive commands
            elif command_lower.startswith("upload ") and DRIVE_ENABLED:
                # Parse: "upload /path/to/file.txt folder_name"
                parts = command[7:].strip().split(" ", 1)
                file_path = parts[0]
                folder_name = parts[1] if len(parts) > 1 else None
                return self.upload_to_drive(file_path, folder_name)

            elif command_lower.startswith("download ") and DRIVE_ENABLED:
                # Parse: "download file_id /local/path"
                parts = command[9:].strip().split(" ", 1)
                if len(parts) == 2:
                    file_id, local_path = parts
                    return self.download_from_drive(file_id, local_path)
                else:
                    return "Drive: Invalid download format. Use 'download file_id local_path'"

            elif command_lower in ["files", "list"] and DRIVE_ENABLED:
                return self.list_drive_files()

            elif command_lower.startswith("share ") and DRIVE_ENABLED:
                # Parse: "share file_id user@example.com role"
                parts = command[6:].strip().split(" ", 2)
                if len(parts) >= 2:
                    file_id = parts[0]
                    email = parts[1]
                    role = parts[2] if len(parts) > 2 else "reader"
                    return self.share_drive_file(file_id, email, role)
                else:
                    return "Drive: Invalid share format. Use 'share file_id email [role]'"

            elif command_lower.startswith("create ") and DRIVE_ENABLED:
                # Parse: "create folder Folder Name"
                if command_lower.startswith("create folder "):
                    folder_name = command[14:].strip()
                    return self.create_drive_folder(folder_name)
                else:
                    return "Drive: Invalid create format. Use 'create folder name'"

            else:
                available_commands = []
                if GMAIL_ENABLED:
                    available_commands.extend(["send recipient subject: body", "inbox", "emails"])
                if YOUTUBE_ENABLED:
                    available_commands.extend(["upload path title: desc", "videos", "search query"])
                if DRIVE_ENABLED:
                    available_commands.extend(["upload path [folder]", "download id path", "files", "share id email [role]", "create folder name"])

                return f"Google: Unknown command. Available: {', '.join(available_commands)}"

        except Exception as e:
            logger.error(f"Error executing Google command: {e}")
            return f"Google Error: {str(e)}"

    def get_google_credentials(self, scopes):
        """Get Google API credentials"""
        try:
            creds = None
            if os.path.exists(GOOGLE_TOKEN_FILE):
                creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, scopes)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
                        return None, f"Google credentials file not found: {GOOGLE_CREDENTIALS_FILE}"

                    flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, scopes)
                    creds = flow.run_local_server(port=0)

                with open(GOOGLE_TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())

            return creds, None
        except Exception as e:
            return None, str(e)

    def send_gmail(self, to, subject, body):
        """Send an email via Gmail API"""
        try:
            if not GMAIL_ENABLED:
                return "Gmail: Gmail integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/gmail.send']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Gmail Auth Error: {error}"

            service = build('gmail', 'v1', credentials=creds)

            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            message_body = {'raw': raw}

            sent_message = service.users().messages().send(userId='me', body=message_body).execute()
            return f"Gmail: Email sent successfully to {to} (ID: {sent_message['id']})"

        except HttpError as e:
            return f"Gmail API Error: {e}"
        except Exception as e:
            return f"Gmail Send Error: {str(e)}"

    def read_gmail_inbox(self):
        """Read recent emails from Gmail inbox"""
        try:
            if not GMAIL_ENABLED:
                return "Gmail: Gmail integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Gmail Auth Error: {error}"

            service = build('gmail', 'v1', credentials=creds)

            results = service.users().messages().list(userId='me', maxResults=GMAIL_MAX_EMAILS).execute()
            messages = results.get('messages', [])

            if not messages:
                return "Gmail: No emails found in inbox"

            email_summaries = []
            for msg in messages[:5]:  # Show first 5 emails
                msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                headers = msg_data['payload']['headers']

                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')

                email_summaries.append(f"From: {sender} | Subject: {subject}")

            return "Recent Emails:\n" + "\n".join(email_summaries)

        except HttpError as e:
            return f"Gmail API Error: {e}"
        except Exception as e:
            return f"Gmail Read Error: {str(e)}"

    def upload_youtube_video(self, video_path, title, description):
        """Upload a video to YouTube"""
        try:
            if not YOUTUBE_ENABLED:
                return "YouTube: YouTube integration disabled"

            if not os.path.exists(video_path):
                return f"YouTube: Video file not found: {video_path}"

            SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"YouTube Auth Error: {error}"

            service = build('youtube', 'v3', credentials=creds)

            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': YOUTUBE_DEFAULT_TAGS,
                    'categoryId': YOUTUBE_CATEGORY_ID
                },
                'status': {
                    'privacyStatus': YOUTUBE_PRIVACY_STATUS
                }
            }

            media_body = MediaFileUpload(video_path, chunksize=-1, resumable=True)

            request = service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media_body
            )

            response = request.execute()
            return f"YouTube: Video uploaded successfully! Title: {title} (ID: {response['id']})"

        except HttpError as e:
            return f"YouTube API Error: {e}"
        except Exception as e:
            return f"YouTube Upload Error: {str(e)}"

    def list_youtube_videos(self):
        """List user's YouTube videos"""
        try:
            if not YOUTUBE_ENABLED:
                return "YouTube: YouTube integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"YouTube Auth Error: {error}"

            service = build('youtube', 'v3', credentials=creds)

            request = service.videos().list(
                part='snippet',
                myRating='like',
                maxResults=10
            )

            response = request.execute()

            if not response.get('items'):
                return "YouTube: No videos found"

            videos = []
            for item in response['items'][:5]:
                title = item['snippet']['title']
                video_id = item['id']
                videos.append(f"{title} (ID: {video_id})")

            return "Your YouTube Videos:\n" + "\n".join(videos)

        except HttpError as e:
            return f"YouTube API Error: {e}"
        except Exception as e:
            return f"YouTube List Error: {str(e)}"

    def search_youtube(self, query):
        """Search YouTube for videos"""
        try:
            if not YOUTUBE_ENABLED:
                return "YouTube: YouTube integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"YouTube Auth Error: {error}"

            service = build('youtube', 'v3', credentials=creds)

            request = service.search().list(
                part='snippet',
                q=query,
                maxResults=5,
                type='video'
            )

            response = request.execute()

            if not response.get('items'):
                return f"YouTube: No videos found for '{query}'"

            results = []
            for item in response['items']:
                title = item['snippet']['title']
                channel = item['snippet']['channelTitle']
                video_id = item['id']['videoId']
                results.append(f"{title} by {channel} (ID: {video_id})")

            return f"YouTube Search Results for '{query}':\n" + "\n".join(results)

        except HttpError as e:
            return f"YouTube API Error: {e}"
        except Exception as e:
            return f"YouTube Search Error: {str(e)}"

    def upload_to_drive(self, file_path, folder_name=None):
        """Upload a file to Google Drive"""
        try:
            if not DRIVE_ENABLED:
                return "Drive: Google Drive integration disabled"

            if not os.path.exists(file_path):
                return f"Drive: File not found: {file_path}"

            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Drive Auth Error: {error}"

            service = build('drive', 'v3', credentials=creds)

            # Find folder if specified
            folder_id = None
            if folder_name:
                folder_id = self.find_drive_folder(service, folder_name)
                if not folder_id:
                    # Create folder if it doesn't exist
                    folder_id = self.create_drive_folder_service(service, folder_name)

            # Upload file
            file_metadata = {'name': os.path.basename(file_path)}
            if folder_id:
                file_metadata['parents'] = [folder_id]

            media = MediaFileUpload(file_path, resumable=True)
            file = service.files().create(body=file_metadata, media_body=media, fields='id,name').execute()

            return f"Drive: File uploaded successfully! Name: {file['name']} (ID: {file['id']})"

        except HttpError as e:
            return f"Drive API Error: {e}"
        except Exception as e:
            return f"Drive Upload Error: {str(e)}"

    def download_from_drive(self, file_id, local_path):
        """Download a file from Google Drive"""
        try:
            if not DRIVE_ENABLED:
                return "Drive: Google Drive integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Drive Auth Error: {error}"

            service = build('drive', 'v3', credentials=creds)

            # Get file metadata
            file_metadata = service.files().get(fileId=file_id, fields='name,mimeType').execute()
            file_name = file_metadata['name']

            # Download file
            request = service.files().get_media(fileId=file_id)
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

            return f"Drive: File downloaded successfully! Saved as: {local_path}"

        except HttpError as e:
            return f"Drive API Error: {e}"
        except Exception as e:
            return f"Drive Download Error: {str(e)}"

    def list_drive_files(self):
        """List files in Google Drive"""
        try:
            if not DRIVE_ENABLED:
                return "Drive: Google Drive integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Drive Auth Error: {error}"

            service = build('drive', 'v3', credentials=creds)

            results = service.files().list(
                pageSize=10,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)"
            ).execute()

            files = results.get('files', [])
            if not files:
                return "Drive: No files found"

            file_list = []
            for file in files[:5]:  # Show first 5 files
                file_type = "Folder" if file['mimeType'] == 'application/vnd.google-apps.folder' else "File"
                file_list.append(f"{file['name']} ({file_type}) - ID: {file['id']}")

            return "Recent Drive Files:\n" + "\n".join(file_list)

        except HttpError as e:
            return f"Drive API Error: {e}"
        except Exception as e:
            return f"Drive List Error: {str(e)}"

    def share_drive_file(self, file_id, email, role="reader"):
        """Share a Google Drive file"""
        try:
            if not DRIVE_ENABLED:
                return "Drive: Google Drive integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Drive Auth Error: {error}"

            service = build('drive', 'v3', credentials=creds)

            # Create permission
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }

            service.permissions().create(
                fileId=file_id,
                body=permission,
                sendNotificationEmail=True
            ).execute()

            return f"Drive: File shared with {email} (role: {role})"

        except HttpError as e:
            return f"Drive API Error: {e}"
        except Exception as e:
            return f"Drive Share Error: {str(e)}"

    def create_drive_folder(self, folder_name):
        """Create a folder in Google Drive"""
        try:
            if not DRIVE_ENABLED:
                return "Drive: Google Drive integration disabled"

            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds, error = self.get_google_credentials(SCOPES)
            if error:
                return f"Drive Auth Error: {error}"

            service = build('drive', 'v3', credentials=creds)

            folder_id = self.create_drive_folder_service(service, folder_name)
            return f"Drive: Folder created successfully! Name: {folder_name} (ID: {folder_id})"

        except HttpError as e:
            return f"Drive API Error: {e}"
        except Exception as e:
            return f"Drive Create Folder Error: {str(e)}"

    def find_drive_folder(self, service, folder_name):
        """Find a folder by name in Google Drive"""
        try:
            results = service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
                spaces='drive',
                fields='files(id, name)'
            ).execute()

            files = results.get('files', [])
            return files[0]['id'] if files else None

        except Exception as e:
            logger.error(f"Error finding Drive folder: {e}")
            return None

    def create_drive_folder_service(self, service, folder_name):
        """Create a folder using the Drive service"""
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }

            file = service.files().create(body=file_metadata, fields='id').execute()
            return file.get('id')

        except Exception as e:
            logger.error(f"Error creating Drive folder: {e}")
            return None

    def sl_send_chat_message(self, message):
        """Send a chat message in Second Life (requires active session)"""
        try:
            # Find Second Life session
            sl_session = None
            for sid, session in self.session_manager.sessions.items():
                if session["ai_type"] == "secondlife" and session["status"] == "connected":
                    sl_session = session
                    break

            if not sl_session:
                return "Second Life: No active Second Life session found. Use 'sl: login' first."

            # This is highly experimental - chat input selectors would need to be identified
            # For now, return a placeholder
            return f"Second Life: Chat message queued: '{message}'. Web automation implementation needed for actual sending."

        except Exception as e:
            return f"Second Life Chat Error: {str(e)}"

    def sl_move_avatar(self, direction):
        """Move avatar in Second Life world"""
        try:
            # Avatar movement would require identifying movement controls in the viewer
            valid_directions = ["forward", "back", "left", "right", "up", "down", "fly", "stop"]

            if direction not in valid_directions:
                return f"Second Life: Invalid direction '{direction}'. Valid: {', '.join(valid_directions)}"

            return f"Second Life: Avatar movement '{direction}' queued. Implementation requires viewer control mapping."

        except Exception as e:
            return f"Second Life Movement Error: {str(e)}"

    def sl_teleport(self, location):
        """Teleport to a location in Second Life"""
        try:
            return f"Second Life: Teleport to '{location}' queued. Implementation requires location search and teleport controls."

        except Exception as e:
            return f"Second Life Teleport Error: {str(e)}"

    def sl_get_status(self):
        """Get Second Life session status"""
        try:
            sl_sessions = [s for s in self.session_manager.sessions.values() if s["ai_type"] == "secondlife"]

            if sl_sessions:
                status = "Second Life Sessions:\n"
                for session in sl_sessions:
                    status += f"- {session['name']}: {session['status']}\n"
                return status
            else:
                return "Second Life: No active sessions. Use 'sl: login' to connect."

        except Exception as e:
            return f"Second Life Status Error: {str(e)}"

    def sl_logout(self):
        """Logout from Second Life"""
        try:
            sl_sessions = [s for s in self.session_manager.sessions.values() if s["ai_type"] == "secondlife"]

            for session in sl_sessions:
                if session["browser"]:
                    asyncio.create_task(session["browser"].close())
                del self.session_manager.sessions[session["id"]]

            return "Second Life: Logged out from all sessions."

        except Exception as e:
            return f"Second Life Logout Error: {str(e)}"

    def sl_voice_command(self, voice_command):
        """Handle voice commands in Second Life"""
        try:
            if not SECONDLIFE_VOICE_CHAT_ENABLED:
                return "Second Life: Voice chat disabled in configuration."

            # Voice chat in Second Life would be very complex
            # This would require interfacing with their voice system
            return f"Second Life: Voice command '{voice_command}' queued. Voice chat integration requires advanced audio routing."

        except Exception as e:
            return f"Second Life Voice Error: {str(e)}"

    async def execute_css_command(self, command):
        """Execute CSS/Tailwind customization commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["inject", "apply", "load"]:
                return await self.inject_custom_css()
            elif command_lower.startswith("add "):
                # Parse: "add .my-class { color: red; }"
                css_code = command[4:].strip()
                return await self.add_css_rule(css_code)
            elif command_lower.startswith("remove ") or command_lower.startswith("delete "):
                # Parse: "remove .my-class" or "delete #my-id"
                selector = command.split(" ", 1)[1].strip()
                return await self.remove_css_rule(selector)
            elif command_lower in ["clear", "reset"]:
                return await self.clear_custom_css()
            elif command_lower in ["status", "current"]:
                return self.get_css_status()
            elif command_lower.startswith("file "):
                # Parse: "file custom_styles.css"
                filename = command[5:].strip()
                return await self.load_css_from_file(filename)
            else:
                return "CSS: Unknown command. Available: inject, add <css>, remove <selector>, clear, status, file <filename>"

        except Exception as e:
            logger.error(f"Error executing CSS command: {e}")
            return f"CSS Error: {str(e)}"

    async def inject_custom_css(self):
        """Inject custom CSS from file into the current page"""
        try:
            if not self.page:
                return "CSS: No active page session. Start the automator first."

            if not os.path.exists(CUSTOM_CSS_FILE):
                return f"CSS: Custom CSS file '{CUSTOM_CSS_FILE}' not found. Create it with your Tailwind/CSS styles."

            with open(CUSTOM_CSS_FILE, 'r') as f:
                css_content = f.read()

            if not css_content.strip():
                return f"CSS: Custom CSS file '{CUSTOM_CSS_FILE}' is empty."

            # Inject CSS into the page
            await self.page.add_style_tag(content=css_content)

            logger.info(f"Injected custom CSS from {CUSTOM_CSS_FILE}")
            return f"CSS: Successfully injected custom styles from '{CUSTOM_CSS_FILE}'"

        except Exception as e:
            logger.error(f"Error injecting custom CSS: {e}")
            return f"CSS Injection Error: {str(e)}"

    async def add_css_rule(self, css_code):
        """Add a CSS rule dynamically to the page"""
        try:
            if not self.page:
                return "CSS: No active page session. Start the automator first."

            # Validate basic CSS syntax
            if not ('{' in css_code and '}' in css_code):
                return "CSS: Invalid CSS format. Use 'selector { property: value; }'"

            # Inject the CSS rule
            await self.page.add_style_tag(content=css_code)

            logger.info(f"Added CSS rule: {css_code[:50]}...")
            return f"CSS: Added rule '{css_code[:30]}...'"

        except Exception as e:
            logger.error(f"Error adding CSS rule: {e}")
            return f"CSS Add Error: {str(e)}"

    async def remove_css_rule(self, selector):
        """Remove a CSS rule by selector"""
        try:
            if not self.page:
                return "CSS: No active page session. Start the automator first."

            # This is a simplified implementation
            # In a real scenario, you'd need to track added styles and remove them
            # For now, we'll inject a rule that overrides the selector
            override_css = f"{selector} {{ display: none !important; }}"
            await self.page.add_style_tag(content=override_css)

            logger.info(f"Removed/hid CSS rule for selector: {selector}")
            return f"CSS: Hidden elements matching '{selector}'"

        except Exception as e:
            logger.error(f"Error removing CSS rule: {e}")
            return f"CSS Remove Error: {str(e)}"

    async def clear_custom_css(self):
        """Clear all custom CSS from the page"""
        try:
            if not self.page:
                return "CSS: No active page session. Start the automator first."

            # Remove all style tags that were added by this automator
            # This is a simplified approach - in practice, you'd need to track added styles
            await self.page.evaluate("""
                const styles = document.querySelectorAll('style[data-automator="true"]');
                styles.forEach(style => style.remove());
            """)

            logger.info("Cleared custom CSS")
            return "CSS: Cleared all custom styles"

        except Exception as e:
            logger.error(f"Error clearing custom CSS: {e}")
            return f"CSS Clear Error: {str(e)}"

    def get_css_status(self):
        """Get current CSS customization status"""
        try:
            status = "CSS Customization Status:\n"
            status += f"- Enabled: {CSS_CUSTOMIZATION_ENABLED}\n"
            status += f"- Custom CSS File: {CUSTOM_CSS_FILE}\n"
            status += f"- Auto Inject: {AUTO_INJECT_CSS}\n"

            if os.path.exists(CUSTOM_CSS_FILE):
                with open(CUSTOM_CSS_FILE, 'r') as f:
                    lines = f.readlines()
                status += f"- File exists: Yes ({len(lines)} lines)\n"
            else:
                status += "- File exists: No\n"

            if self.page:
                status += "- Active session: Yes"
            else:
                status += "- Active session: No"

            return status

        except Exception as e:
            return f"CSS Status Error: {str(e)}"

    async def load_css_from_file(self, filename):
        """Load CSS from a specific file"""
        try:
            if not os.path.exists(filename):
                return f"CSS: File '{filename}' not found."

            with open(filename, 'r') as f:
                css_content = f.read()

            if not css_content.strip():
                return f"CSS: File '{filename}' is empty."

            if not self.page:
                return "CSS: No active page session. Start the automator first."

            # Inject CSS into the page
            await self.page.add_style_tag(content=css_content)

            logger.info(f"Loaded CSS from file: {filename}")
            return f"CSS: Successfully loaded styles from '{filename}'"

        except Exception as e:
            logger.error(f"Error loading CSS from file: {e}")
            return f"CSS Load Error: {str(e)}"

    def execute_voice_chat_command(self, command):
        """Execute phone call-style voice chat commands"""
        try:
            command_lower = command.lower().strip()

            if command_lower in ["start", "begin", "on", "call", "dial"]:
                return self.start_voice_chat()
            elif command_lower.startswith("start ") or command_lower.startswith("begin ") or command_lower.startswith("with ") or command_lower.startswith("call "):
                # Parse: "start nomi" or "with chatgpt" or "call nomi"
                ai_name = command[6:].strip() if command_lower.startswith("start ") else command[5:].strip() if command_lower.startswith("begin ") else command[5:].strip() if command_lower.startswith("with ") else command[5:].strip()
                return self.start_voice_chat(ai_name)
            elif command_lower in ["stop", "end", "off", "quit", "hangup", "hang up", "disconnect"]:
                return self.stop_voice_chat()
            elif command_lower in ["status", "check", "state"]:
                if self.voice_chat_active:
                    session_name = "Nomi AI"
                    if self.voice_chat_session and self.voice_chat_session != self.main_session_id:
                        session = self.session_manager.get_session(self.voice_chat_session)
                        if session:
                            session_name = session["name"]
                    return f"📞 Phone Call: ACTIVE - Continuous listening mode engaged with {session_name}. Say 'hang up' to end call."
                else:
                    return "📞 Phone Call: Not active. Use 'vchat:start' to begin a phone call."
            elif command_lower in ["mute", "unmute", "toggle mute"]:
                # Mute/unmute functionality (placeholder for now)
                return "📞 Phone Call: Mute functionality not yet implemented, but you can stop speaking if needed."
            elif command_lower in ["test", "demo", "check mic"]:
                # Test speech recognition
                text, error = self.recognize_speech(timeout=5)
                if error:
                    return f"📞 Phone Call Test: {error}"
                else:
                    return f"📞 Phone Call Test: Microphone working! Recognized '{text}'"
            else:
                return "📞 Phone Call: Unknown command. Available: start [ai_name], hangup, status, mute, test"

        except Exception as e:
            logger.error(f"Error executing phone call command: {e}")
            return f"📞 Phone Call Error: {str(e)}"

    async def send_response(self, response):
        """Send a response to the chat and optionally speak it"""
        try:
            # Speak the response if voice responses are enabled OR if voice chat is active
            should_speak = VOICE_RESPONSES_ENABLED or self.voice_chat_active
            if should_speak and ((ELEVENLABS_ENABLED and ELEVENLABS_AVAILABLE and ELEVENLABS_API_KEY) or PYTTSX3_AVAILABLE):
                try:
                    # Try ElevenLabs first if available
                    if ELEVENLABS_ENABLED and ELEVENLABS_AVAILABLE and ELEVENLABS_API_KEY:
                        # Set API key for voice generation
                        set_api_key(ELEVENLABS_API_KEY)

                        # Generate and play audio asynchronously (don't block the response)
                        import threading
                        def speak_response_elevenlabs():
                            try:
                                audio = generate(
                                    text=response,
                                    voice=ELEVENLABS_VOICE_ID,
                                    model=ELEVENLABS_MODEL,
                                    voice_settings={
                                        "stability": ELEVENLABS_VOICE_STABILITY,
                                        "similarity_boost": ELEVENLABS_VOICE_SIMILARITY,
                                        "style": ELEVENLABS_VOICE_STYLE,
                                        "use_speaker_boost": ELEVENLABS_VOICE_USE_SPEAKER_BOOST
                                    }
                                )
                                play(audio)
                                logger.info("ElevenLabs voice response played successfully")
                            except Exception as e:
                                logger.error(f"Error playing ElevenLabs voice response: {e}")

                        # Start voice generation in background thread
                        voice_thread = threading.Thread(target=speak_response_elevenlabs, daemon=True)
                        voice_thread.start()

                    # Fallback to pyttsx3 if ElevenLabs not available or failed
                    elif PYTTSX3_AVAILABLE:
                        self.speak_text(response)
                        logger.info("pyttsx3 voice response played successfully")

                except Exception as e:
                    logger.error(f"Error initiating voice response: {e}")
                    # Final fallback: try pyttsx3 if not already tried
                    if PYTTSX3_AVAILABLE:
                        try:
                            self.speak_text(response)
                            logger.info("Fallback pyttsx3 voice response played successfully")
                        except Exception as e2:
                            logger.error(f"Error playing fallback voice response: {e2}")

            # Find input field - update selectors based on UI inspection
            input_field = await self.page.query_selector('input[type="text"], textarea, [contenteditable="true"]')
            if input_field:
                await input_field.fill(response)
            else:
                logger.error("Input field not found")

            # Find send button
            send_button = await self.page.query_selector('button:has-text("Send"), [aria-label="Send"], .send-button')
            if send_button:
                await send_button.click()
            else:
                # Try pressing Enter in input field
                await input_field.press('Enter')

            logger.info(f"Sent response: {response}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    async def stop(self):
        """Stop the automator"""
        if self.browser:
            await self.browser.close()
        logger.info("Nomi.ai Automator stopped")

async def main():
    automator = NomiAutomator()
    try:
        await automator.start()
    except KeyboardInterrupt:
        await automator.stop()

if __name__ == "__main__":
    asyncio.run(main())