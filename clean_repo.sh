#\!/bin/bash

echo "🧹 Cleaning large files from git history..."

# Remove large model directories from git history
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch -r models/ trained_models/ || true' --prune-empty --tag-name-filter cat -- --all

# Clean up refs
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "✅ Repository cleaned\!"
echo "Before push, check size with: du -sh .git/"
