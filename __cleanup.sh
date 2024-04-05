#!/bin/bash

# Confirm with the user before proceeding
read -p "This action will remove all previous commits from the current branch both locally and remotely. Are you sure you want to proceed? (y/n): " choice
if [ "$choice" != "y" ]; then
    echo "Aborted."
    exit 1
fi

# Get the current branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Reset the branch to an empty state
git checkout --orphan temp_branch
git add -A
git commit -m "Initial commit"
git branch -D "$current_branch"
git branch -m "$current_branch"

# Force push the changes to remote
git push origin "$current_branch" --force

echo "All previous commits have been removed from the branch '$current_branch' both locally and remotely."
