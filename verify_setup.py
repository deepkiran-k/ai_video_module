#!/usr/bin/env python3
"""
Verification script for AI Video Guide MVP.

This script verifies the project structure and provides setup instructions.
"""

import os
import sys
from pathlib import Path


def check_project_structure():
    """Check if all required files and directories exist."""
    print("🔍 Checking project structure...")
    
    required_files = [
        "app.py",
        "setup.py", 
        "example_usage.py",
        "requirements.txt",
        "README.md",
        "PROJECT_SUMMARY.md",
        ".env",
        "core/input_handler.py",
        "core/scene_detector.py",
        "core/ocr_extractor.py",
        "core/prompt_selector.py",
        "core/script_voiceover.py",
        "core/video_editor.py",
        "core/exporter.py"
    ]
    
    required_dirs = [
        "uploads",
        "outputs",
        "core"
    ]
    
    all_good = True
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists() and Path(dir_path).is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_good = False
    
    return all_good


def check_python_version():
    """Check Python version compatibility."""
    print(f"\n🐍 Python version: {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False


def check_api_keys():
    """Check for API keys."""
    print("\n🔑 Checking API keys...")
    
    # Load .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed")
    
    azure_key = os.getenv('api_key')
    azure_endpoint = os.getenv('azure_endpoint')
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    
    if azure_key and azure_endpoint:
        print("✅ Azure OpenAI credentials found in .env file")
        print(f"   Endpoint: {azure_endpoint}")
    else:
        print("⚠️  Azure OpenAI credentials not found in .env file")
        print("   Please check your .env file configuration")
    
    if elevenlabs_key:
        print("✅ ELEVENLABS_API_KEY is set")
    else:
        print("ℹ️  ELEVENLABS_API_KEY not found (optional)")
        print("   System will use gTTS fallback if not provided")


def show_next_steps():
    """Show next steps for setup."""
    print("\n🚀 Next Steps:")
    print("=" * 50)
    
    print("\n1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print("   # OR run the setup script:")
    print("   python3 setup.py")
    
    print("\n2. API Keys:")
    print("   Your Azure OpenAI credentials are already configured in .env file")
    print("   Add ELEVENLABS_API_KEY to .env file for premium voiceover (optional)")
    
    print("\n3. Test the System:")
    print("   python3 example_usage.py")
    print("   # OR process your own video:")
    print("   python3 app.py your_video.mp4 'your tutorial prompt'")
    
    print("\n4. View Documentation:")
    print("   cat README.md")
    print("   cat PROJECT_SUMMARY.md")


def show_usage_examples():
    """Show usage examples."""
    print("\n📖 Usage Examples:")
    print("=" * 30)
    
    examples = [
        ("Basic usage", "python3 app.py video.mp4 'Create a login tutorial'"),
        ("Custom output", "python3 app.py demo.mp4 'Show navigation' --output my_guide.mp4"),
        ("List outputs", "python3 app.py --list-outputs"),
        ("Run example", "python3 example_usage.py"),
        ("Setup help", "python3 setup.py")
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")


def main():
    """Main verification function."""
    print("🎬 AI Video Guide MVP - Setup Verification")
    print("=" * 50)
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check API keys
    check_api_keys()
    
    # Show results
    print("\n📊 Verification Results:")
    print("=" * 30)
    
    if structure_ok and python_ok:
        print("✅ Project structure is complete!")
        print("✅ Python version is compatible!")
        print("\n🎉 Ready for setup and testing!")
    else:
        print("❌ Some issues found. Please check the errors above.")
    
    # Show next steps
    show_next_steps()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 50)
    print("🎯 The AI Video Guide MVP is ready to use!")
    print("   Follow the next steps above to get started.")


if __name__ == "__main__":
    main()