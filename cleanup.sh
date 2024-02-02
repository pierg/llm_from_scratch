#!/bin/bash

# Directory of your Python project
PROJECT_DIR="."

# Function to delete files or directories
delete_if_exists() {
    if [ -e "$1" ]; then
        echo "Deleting $1"
        rm -rf "$1"
    fi
}

echo "Cleaning up Python project at $PROJECT_DIR"

# Change to the project directory
cd "$PROJECT_DIR" || exit

# Delete __pycache__ directories and .pyc files
find . -name "__pycache__" -exec rm -rf {} + -o -name "*.pyc" -exec rm -f {} +

# Delete the poetry.lock file
delete_if_exists "poetry.lock"
delete_if_exists ".venv" # or the name of your virtual environment directory

echo "Cleanup complete."

