#!/usr/bin/env python3
"""Example usage of the TikTok auto-posting tool."""

import os
import sys
from tiktok_client import TikTokUploader
from video_processor import VideoProcessor
from config import config

def example_validate_video():
    """Example: Validate a video file."""
    print("=== Video Validation Example ===")
    
    # This would be a real video file path
    video_path = "example_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Example video not found: {video_path}")
        print("💡 Place a video file named 'example_video.mp4' in this directory to test")
        return False
    
    processor = VideoProcessor()
    
    # Validate video
    is_valid, errors = processor.validate_video(video_path)
    
    if is_valid:
        print("✅ Video validation passed!")
        
        # Get video info
        info = processor.get_video_info(video_path)
        print(f"📊 Video Information:")
        print(f"  • Size: {info['size_mb']:.2f} MB")
        print(f"  • Duration: {info['duration']:.2f} seconds")
        print(f"  • Resolution: {info['width']}x{info['height']}")
        print(f"  • Aspect ratio: {info['aspect_ratio']:.2f}")
        print(f"  • Format: {info['format']}")
        
        return True
    else:
        print("❌ Video validation failed:")
        for error in errors:
            print(f"  • {error}")
        return False

def example_upload_video():
    """Example: Upload a video to TikTok."""
    print("\n=== Video Upload Example ===")
    
    # Check configuration first
    if not config.is_configured():
        print("❌ TikTok API not configured")
        print("💡 Set up your .env file with API credentials")
        return False
    
    video_path = "example_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Example video not found: {video_path}")
        return False
    
    try:
        uploader = TikTokUploader()
        
        # Upload video
        result = uploader.upload_video(
            video_path=video_path,
            caption="Uploaded using TikTok Auto-Posting Tool! 🚀",
            privacy="private",  # Use private for testing
            disable_duet=False,
            disable_comment=False,
            disable_stitch=False
        )
        
        print("🎉 Upload successful!")
        
        if result.get('data', {}).get('video_id'):
            video_id = result['data']['video_id']
            print(f"📹 Video ID: {video_id}")
        
        if result.get('data', {}).get('share_url'):
            share_url = result['data']['share_url']
            print(f"🔗 Share URL: {share_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

def example_get_user_info():
    """Example: Get user information."""
    print("\n=== User Information Example ===")
    
    if not config.is_configured():
        print("❌ TikTok API not configured")
        return False
    
    try:
        uploader = TikTokUploader()
        user_info = uploader.get_user_info()
        
        if user_info.get('error', {}).get('code') == 'ok':
            user_data = user_info.get('data', {}).get('user', {})
            
            print("👤 User Information:")
            print(f"  • Display Name: {user_data.get('display_name', 'N/A')}")
            print(f"  • Username: {user_data.get('username', 'N/A')}")
            print(f"  • Follower Count: {user_data.get('follower_count', 'N/A')}")
            print(f"  • Following Count: {user_data.get('following_count', 'N/A')}")
            print(f"  • Video Count: {user_data.get('video_count', 'N/A')}")
            
            return True
        else:
            print(f"❌ API Error: {user_info}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to get user info: {str(e)}")
        return False

def main():
    """Run all examples."""
    print("🎬 TikTok Auto-Posting Tool - Examples")
    print("=" * 50)
    
    # Example 1: Validate video
    validation_success = example_validate_video()
    
    # Example 2: Get user info (if configured)
    user_info_success = example_get_user_info()
    
    # Example 3: Upload video (only if validation passed and API is configured)
    if validation_success and config.is_configured():
        upload_success = example_upload_video()
        
        if upload_success:
            print("\n🎉 All examples completed successfully!")
        else:
            print("\n⚠️ Upload example failed (but validation passed)")
    else:
        print("\n💡 Upload example skipped (video validation failed or API not configured)")
    
    print("\n📖 For more examples, check the README.md file")

if __name__ == '__main__':
    main()