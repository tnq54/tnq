"""Configuration management for TikTok auto-posting tool."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TikTokConfig:
    """Configuration class for TikTok API and app settings."""
    
    def __init__(self):
        self.api_key = os.getenv('TIKTOK_API_KEY', '')
        self.api_secret = os.getenv('TIKTOK_API_SECRET', '')
        self.access_token = os.getenv('TIKTOK_ACCESS_TOKEN', '')
        self.user_id = os.getenv('TIKTOK_USER_ID', '')
        
        # Video settings
        self.max_video_size_mb = int(os.getenv('MAX_VIDEO_SIZE_MB', '50'))
        self.supported_formats = ['mp4', 'mov', 'avi', 'webm']
        self.max_duration_seconds = int(os.getenv('MAX_DURATION_SECONDS', '180'))
        
        # Posting settings
        self.default_privacy = os.getenv('DEFAULT_PRIVACY', 'public')  # public, private, friends
        self.auto_caption = os.getenv('AUTO_CAPTION', 'true').lower() == 'true'
        
    def is_configured(self) -> bool:
        """Check if all required configuration is present."""
        return bool(self.api_key and self.api_secret and self.access_token)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.api_key:
            errors.append("TIKTOK_API_KEY is required")
        if not self.api_secret:
            errors.append("TIKTOK_API_SECRET is required")
        if not self.access_token:
            errors.append("TIKTOK_ACCESS_TOKEN is required")
            
        if self.max_video_size_mb <= 0:
            errors.append("MAX_VIDEO_SIZE_MB must be positive")
        if self.max_duration_seconds <= 0:
            errors.append("MAX_DURATION_SECONDS must be positive")
            
        if self.default_privacy not in ['public', 'private', 'friends']:
            errors.append("DEFAULT_PRIVACY must be one of: public, private, friends")
            
        return errors

# Global config instance
config = TikTokConfig()