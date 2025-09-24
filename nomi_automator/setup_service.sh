#!/bin/bash

# Nomi Automator Systemd Service Setup Script
# This script sets up the nomi_automator as a systemd service

set -e

echo "🤖 Setting up Nomi Automator systemd service..."

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    echo "❌ This script should not be run as root. Please run as your regular user."
    exit 1
fi

# Get the current user
CURRENT_USER=$(whoami)
CURRENT_HOME=$HOME

echo "👤 Setting up service for user: $CURRENT_USER"
echo "🏠 Home directory: $CURRENT_HOME"

# Update the service file with correct paths
SERVICE_FILE="nomi_automator.service"
BACKUP_FILE="${SERVICE_FILE}.backup"

if [ -f "$SERVICE_FILE" ]; then
    cp "$SERVICE_FILE" "$BACKUP_FILE"
    echo "📋 Backing up original service file to $BACKUP_FILE"
fi

# Replace placeholders in service file
sed -i "s/User=nomi/User=$CURRENT_USER/g" "$SERVICE_FILE"
sed -i "s/Group=nomi/Group=$CURRENT_USER/g" "$SERVICE_FILE"
sed -i "s|/home/nomi/CodeBERT|$CURRENT_HOME/CodeBERT|g" "$SERVICE_FILE"

echo "📝 Updated service file with user-specific paths"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p "$CURRENT_HOME/.config/nomi_automator"
mkdir -p "$CURRENT_HOME/.local/share/nomi_automator"

# Copy service file to systemd directory
echo "🔧 Installing systemd service..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service
echo "✅ Enabling nomi_automator service..."
sudo systemctl enable nomi_automator.service

# Start the service
echo "🚀 Starting nomi_automator service..."
sudo systemctl start nomi_automator.service

# Check service status
echo "📊 Checking service status..."
sleep 3
sudo systemctl status nomi_automator.service --no-pager -l

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Service Management Commands:"
echo "  Start service:   sudo systemctl start nomi_automator"
echo "  Stop service:    sudo systemctl stop nomi_automator"
echo "  Restart service: sudo systemctl restart nomi_automator"
echo "  Check status:    sudo systemctl status nomi_automator"
echo "  View logs:       sudo journalctl -u nomi_automator -f"
echo ""
echo "🔧 Configuration:"
echo "  Edit config:     nano $CURRENT_HOME/CodeBERT/nomi_automator/config.py"
echo "  Service file:    /etc/systemd/system/nomi_automator.service"
echo ""
echo "⚠️  Important Notes:"
echo "  - The service will start automatically on boot"
echo "  - Check logs if the service fails to start"
echo "  - Ensure all dependencies are installed (pip install -r requirements.txt)"
echo "  - Configure API keys and credentials in config.py"
echo ""
echo "🤖 Your AI assistant will now greet you on every boot!"