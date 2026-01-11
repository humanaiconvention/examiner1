#!/bin/bash
# Setup Aristotle API Key

echo "Enter your Aristotle API key:"
read -s ARISTOTLE_KEY

# Export for current session
export ARISTOTLE_API_KEY=""

# Add to .bashrc for persistence
if ! grep -q "ARISTOTLE_API_KEY" ~/.bashrc; then
    echo "export ARISTOTLE_API_KEY=\"\"" >> ~/.bashrc
    echo "✓ Added to ~/.bashrc"
fi

# Test API connection
echo ""
echo "Testing Aristotle API connection..."
curl -s -H "Authorization: Bearer "   https://api.aristotle.ai/v1/status | head -20

echo ""
echo "✓ API key configured!"
echo "  Run: source ~/.bashrc"
echo "  Or export ARISTOTLE_API_KEY manually in each session"
