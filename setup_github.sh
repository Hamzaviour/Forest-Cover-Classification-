#!/bin/bash

# GitHub Setup Script for Forest Cover Classification
# Created by Hamza Younas | Elevvo Pathways

echo "🌲 Forest Cover Classification - GitHub Setup"
echo "============================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

echo "📁 Initializing Git repository..."
git init

echo "📝 Adding files to Git..."
git add .

echo "💾 Creating initial commit..."
git commit -m "🎉 Initial commit: Forest Cover Classification Dashboard

✨ Features:
- Professional Streamlit GUI with card-based layout
- Random Forest, XGBoost, and Decision Tree models
- Interactive data exploration and visualization
- Model comparison and performance analysis
- Real-time prediction interface
- Created by Hamza Younas | Elevvo Pathways"

echo "🔗 Setting up remote repository..."
echo "Please create a new repository on GitHub first, then run:"
echo ""
echo "git remote add origin https://github.com/YOUR_USERNAME/forest-cover-classification.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "🚀 After that, you can deploy to Streamlit Cloud at: https://share.streamlit.io"
echo ""
echo "✅ Setup complete! Your project is ready for GitHub."
