#!/usr/bin/env python3
"""
Test script for voice chat functionality in nomi_automator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import NomiAutomator
import asyncio

async def test_voice_chat():
    """Test voice chat functionality"""
    print("Testing voice chat functionality...")

    # Initialize automator
    automator = NomiAutomator()

    # Test speech recognition
    print("\n1. Testing speech recognition...")
    try:
        text, error = automator.recognize_speech(timeout=3)
        if error:
            print(f"Speech recognition test: {error}")
        else:
            print(f"Speech recognition test: Recognized '{text}'")
    except Exception as e:
        print(f"Speech recognition test failed: {e}")

    # Test voice chat commands
    print("\n2. Testing voice chat commands...")
    try:
        # Test status (should be inactive)
        result = automator.execute_voice_chat_command("status")
        print(f"Voice chat status: {result}")

        # Test invalid command
        result = automator.execute_voice_chat_command("invalid")
        print(f"Invalid command test: {result}")

    except Exception as e:
        print(f"Voice chat command test failed: {e}")

    # Test TTS
    print("\n3. Testing text-to-speech...")
    try:
        test_text = "Hello, this is a test of the voice chat system."
        result = automator.speak_text(test_text)
        if result:
            print("Text-to-speech test: Success")
        else:
            print("Text-to-speech test: Failed (TTS not available)")
    except Exception as e:
        print(f"Text-to-speech test failed: {e}")

    print("\nVoice chat functionality test completed!")

if __name__ == "__main__":
    asyncio.run(test_voice_chat())