#!/bin/sh

# Create the ~/.streamlit/ directory if it doesn't exist
mkdir -p ~/.streamlit/

# Create or overwrite the config.toml file
cat <<EOL > ~/.streamlit/config.toml
[server]
headless = true
port = $PORT
enableCORS = false
EOL
