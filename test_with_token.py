#!/usr/bin/env python3
"""
Test Archangel with HuggingFace token
"""

import os
from dotenv import load_dotenv

def test_token_setup():
    """Test if HF token is properly configured"""
    print("🔐 Testing HuggingFace Token Setup")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check if token exists
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("❌ No HF_TOKEN found in environment")
        print("💡 Make sure you've added your token to the .env file")
        return False
    
    if hf_token == "your_hugging_face_token_here":
        print("❌ HF_TOKEN is still the placeholder value")
        print("💡 Replace 'your_hugging_face_token_here' with your actual token")
        return False
    
    if not hf_token.startswith('hf_'):
        print("❌ HF_TOKEN doesn't look like a valid HuggingFace token")
        print("💡 HuggingFace tokens should start with 'hf_'")
        return False
    
    print(f"✅ HF_TOKEN found: {hf_token[:10]}...")
    print("🎯 Token looks valid!")
    
    # Test token with HuggingFace
    try:
        import requests
        
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Token authenticated successfully!")
            print(f"👤 Logged in as: {user_info.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Token authentication failed: {response.status_code}")
            print("💡 Check if your token is valid and has the right permissions")
            return False
            
    except Exception as e:
        print(f"❌ Error testing token: {e}")
        return False

def main():
    """Main test function"""
    if test_token_setup():
        print("\n🚀 Ready to test Archangel with AI!")
        print("Try these commands:")
        print("  python3 archangel_lightweight.py interactive")
        print("  python3 archangel_lightweight.py analyze google.com")
    else:
        print("\n🔧 Please fix the token setup first:")
        print("  1. Get a token from: https://huggingface.co/settings/tokens")
        print("  2. Edit .env file and replace the placeholder")
        print("  3. Run this test again")

if __name__ == "__main__":
    main()