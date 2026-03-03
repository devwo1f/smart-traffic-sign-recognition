#!/bin/bash
# GitHub Repository Initialization Script

echo "🚀 Initializing Traffic Sign Recognition System repository..."

# Initialize git
git init
git add .
git commit -m "feat: initial project structure

- ML pipeline (EfficientNet + YOLOv8)
- FastAPI backend with PostgreSQL
- TypeScript dashboard with video/webcam support
- Docker Compose deployment
- GitHub Actions CI pipeline
- Documentation and diagrams"

# Set main branch
git branch -M main

# Prompt for remote
echo ""
echo "📌 Add your remote origin:"
echo "   git remote add origin https://github.com/<your-username>/traffic-sign-system.git"
echo "   git push -u origin main"
echo ""
echo "📌 Create dev branch:"
echo "   git checkout -b dev"
echo "   git push -u origin dev"
echo ""
echo "✅ Repository initialized!"
