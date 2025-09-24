#!/bin/bash

# Nomi Automator Desktop Launcher
# This script provides a simple way to start/stop/check the Nomi Automator service

SERVICE_NAME="nomi_automator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
MAIN_SCRIPT="$SCRIPT_DIR/main.py"

echo "🤖 Nomi Automator Launcher (with CSS Customization)"
echo "=================================================="

# Check if service is running
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✅ Service is running"
    echo ""
    echo "Choose an action:"
    echo "1) Stop service"
    echo "2) Restart service"
    echo "3) View logs"
    echo "4) Check status"
    echo "5) Exit"
    read -p "Enter choice (1-5): " choice

    case $choice in
        1)
            echo "Stopping service..."
            sudo systemctl stop "$SERVICE_NAME"
            echo "✅ Service stopped"
            ;;
        2)
            echo "Restarting service..."
            sudo systemctl restart "$SERVICE_NAME"
            echo "✅ Service restarted"
            ;;
        3)
            echo "Viewing logs (Ctrl+C to exit)..."
            sudo journalctl -u "$SERVICE_NAME" -f
            ;;
        4)
            sudo systemctl status "$SERVICE_NAME" --no-pager -l
            ;;
        5)
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
else
    echo "❌ Service is not running"
    echo ""
    echo "Choose an action:"
    echo "1) Start service"
    echo "2) Run directly (for testing)"
    echo "3) Install/setup service"
    echo "4) Exit"
    read -p "Enter choice (1-4): " choice

    case $choice in
        1)
            echo "Starting service..."
            sudo systemctl start "$SERVICE_NAME"
            sleep 2
            if systemctl is-active --quiet "$SERVICE_NAME"; then
                echo "✅ Service started successfully"
            else
                echo "❌ Failed to start service"
                echo "Check logs: sudo journalctl -u $SERVICE_NAME -n 20"
            fi
            ;;
        2)
            echo "Running directly..."
            if [ -f "$VENV_PYTHON" ]; then
                "$VENV_PYTHON" "$MAIN_SCRIPT"
            else
                echo "❌ Virtual environment not found. Run setup first."
            fi
            ;;
        3)
            echo "Installing/setup service..."
            if [ -f "$SCRIPT_DIR/setup_service.sh" ]; then
                bash "$SCRIPT_DIR/setup_service.sh"
            else
                echo "❌ Setup script not found"
            fi
            ;;
        4)
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
fi