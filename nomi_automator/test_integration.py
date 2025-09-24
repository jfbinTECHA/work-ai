#!/usr/bin/env python3
"""
Test script for VSCode and Yarn integration functionality
"""

import shlex

# Mock configuration values
VSCODE_COMMAND_PREFIXES = ["vscode:", "code:", "editor:"]
YARN_COMMAND_PREFIXES = ["yarn:", "npm:", "package:"]

def extract_command(message, prefixes):
    """Extract command from message if it starts with any of the prefixes"""
    message_lower = message.lower().strip()
    for prefix in prefixes:
        if message_lower.startswith(prefix):
            return message[len(prefix):].strip()
    return None

def test_command_extraction():
    """Test command extraction from messages"""

    # Test VSCode commands
    assert extract_command("vscode: --version", VSCODE_COMMAND_PREFIXES) == "--version"
    assert extract_command("code: .", VSCODE_COMMAND_PREFIXES) == "."
    assert extract_command("editor: --new-window", VSCODE_COMMAND_PREFIXES) == "--new-window"

    # Test Yarn commands
    assert extract_command("yarn: install", YARN_COMMAND_PREFIXES) == "install"
    assert extract_command("npm: add lodash", YARN_COMMAND_PREFIXES) == "add lodash"
    assert extract_command("package: run build", YARN_COMMAND_PREFIXES) == "run build"

    # Test non-matching messages
    assert extract_command("hello world", VSCODE_COMMAND_PREFIXES) is None
    assert extract_command("how are you", YARN_COMMAND_PREFIXES) is None

    print("✓ Command extraction tests passed")

def test_shlex_parsing():
    """Test command parsing with shlex"""
    # Test various command formats
    commands = [
        "--version",
        ".",
        "--new-window",
        "install",
        "add lodash",
        "run build",
        "add package-name --dev"
    ]

    for cmd in commands:
        try:
            args = shlex.split(cmd)
            assert isinstance(args, list)
        except Exception as e:
            print(f"Failed to parse command '{cmd}': {e}")
            raise

    print("✓ Command parsing tests passed")

def test_prefix_matching():
    """Test prefix matching logic"""
    test_cases = [
        ("vscode: --version", True),
        ("code: .", True),
        ("editor: --new-window", True),
        ("yarn: install", True),
        ("npm: add lodash", True),
        ("package: run build", True),
        ("hello world", False),
        ("how are you", False),
        ("vscode hello", False),  # No colon
    ]

    for message, should_match in test_cases:
        vscode_match = extract_command(message, VSCODE_COMMAND_PREFIXES) is not None
        yarn_match = extract_command(message, YARN_COMMAND_PREFIXES) is not None
        has_match = vscode_match or yarn_match

        if has_match != should_match:
            print(f"Failed: '{message}' - expected {should_match}, got {has_match}")
            raise AssertionError(f"Prefix matching failed for '{message}'")

    print("✓ Prefix matching tests passed")

if __name__ == "__main__":
    print("Testing VSCode and Yarn integration...")
    test_command_extraction()
    test_shlex_parsing()
    test_prefix_matching()
    print("All tests passed! ✓")