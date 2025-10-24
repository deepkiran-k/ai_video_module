#!/usr/bin/env python3
"""
Setup script for AI Video Guide MVP.

This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["uploads", "outputs", "temp"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_api_keys():
    """Check for API keys and provide setup instructions."""
    print("\n🔑 Checking API keys...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY not found")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or add it to your .bashrc/.zshrc file")
    
    if elevenlabs_key:
        print("✅ ELEVENLABS_API_KEY is set")
    else:
        print("ℹ️  ELEVENLABS_API_KEY not found (optional)")
        print("   Set it with: export ELEVENLABS_API_KEY='your-key-here'")
        print("   System will use gTTS fallback if not provided")


def create_env_template():
    """Create a .env template file."""
    env_template = """# AI Video Guide MVP - Environment Variables
# Copy this file to .env and fill in your API keys

# Required for AI features
OPENAI_API_KEY=your_openai_api_key_here

# Optional for high-quality voiceover (falls back to gTTS if not provided)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Optional: Custom directories
# UPLOADS_DIR=uploads
# OUTPUTS_DIR=outputs
"""
    
    env_file = Path(".env.template")
    with open(env_file, "w") as f:
        f.write(env_template)
    
    print(f"✅ Created environment template: {env_file}")


def test_installation():
    """Test if the installation is working."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        print("✅ OpenCV imported successfully")
        
        import moviepy
        print("✅ MoviePy imported successfully")
        
        import easyocr
        print("✅ EasyOCR imported successfully")
        
        import openai
        print("✅ OpenAI imported successfully")
        
        print("✅ All core dependencies are working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function."""
    print("🎬 AI Video Guide MVP - Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check API keys
    check_api_keys()
    
    # Create environment template
    create_env_template()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Set your API keys (see .env.template)")
        print("2. Run: python example_usage.py")
        print("3. Or run: python app.py your_video.mp4 'your prompt'")
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("Some dependencies may not be working correctly.")


if __name__ == "__main__":
    main()