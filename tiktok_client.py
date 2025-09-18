"""TikTok API client for auto-posting videos."""

import requests
import json
import logging
import os
from typing import Optional, Dict, Any
from config import config
from video_processor import VideoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TikTokUploader:
    """Client for uploading videos to TikTok via API."""
    
    def __init__(self):
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.access_token = config.access_token
        self.user_id = config.user_id
        self.base_url = "https://open-api.tiktok.com"
        self.video_processor = VideoProcessor()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration errors: %s", "; ".join(errors))
            raise ValueError("Invalid configuration")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """
        Make authenticated request to TikTok API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        # Add authentication headers
        headers = kwargs.get('headers', {})
        headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        })
        kwargs['headers'] = headers
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API request failed: %s", str(e))
            raise Exception(f"TikTok API request failed: {str(e)}")
    
    def upload_video(self, video_path: str, caption: str = "", 
                    privacy: str = None, disable_duet: bool = False,
                    disable_comment: bool = False, disable_stitch: bool = False) -> Dict[Any, Any]:
        """
        Upload a video to TikTok.
        
        Args:
            video_path: Path to the video file
            caption: Video caption/description
            privacy: Privacy setting ('public', 'private', 'friends')
            disable_duet: Whether to disable duet
            disable_comment: Whether to disable comments
            disable_stitch: Whether to disable stitch
            
        Returns:
            Upload response data
        """
        # Validate video first
        is_valid, errors = self.video_processor.validate_video(video_path)
        if not is_valid:
            raise ValueError(f"Video validation failed: {'; '.join(errors)}")
        
        # Set default privacy if not provided
        if privacy is None:
            privacy = config.default_privacy
        
        logger.info("Starting video upload: %s", video_path)
        
        # Step 1: Initialize upload session
        init_data = {
            "post_info": {
                "title": caption[:100],  # TikTok has character limits
                "privacy_level": privacy.upper(),
                "disable_duet": disable_duet,
                "disable_comment": disable_comment,
                "disable_stitch": disable_stitch,
                "video_cover_timestamp_ms": 1000
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": self._get_file_size(video_path),
                "chunk_size": 10485760,  # 10MB chunks
                "total_chunk_count": self._calculate_chunks(video_path)
            }
        }
        
        # Initialize upload
        upload_init = self._make_request('POST', '/v2/post/publish/video/init/', json=init_data)
        
        if upload_init.get('error', {}).get('code') != 'ok':
            raise Exception(f"Upload initialization failed: {upload_init}")
        
        upload_id = upload_init['data']['upload_id']
        upload_url = upload_init['data']['upload_url']
        
        logger.info("Upload initialized with ID: %s", upload_id)
        
        # Step 2: Upload video chunks
        self._upload_video_chunks(video_path, upload_url, upload_id)
        
        # Step 3: Commit upload
        commit_data = {
            "upload_id": upload_id
        }
        
        commit_result = self._make_request('POST', '/v2/post/publish/video/commit/', json=commit_data)
        
        if commit_result.get('error', {}).get('code') != 'ok':
            raise Exception(f"Upload commit failed: {commit_result}")
        
        logger.info("Video uploaded successfully!")
        return commit_result
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(file_path)
    
    def _calculate_chunks(self, file_path: str, chunk_size: int = 10485760) -> int:
        """Calculate number of chunks needed for file upload."""
        file_size = self._get_file_size(file_path)
        return (file_size + chunk_size - 1) // chunk_size
    
    def _upload_video_chunks(self, file_path: str, upload_url: str, upload_id: str):
        """Upload video file in chunks."""
        chunk_size = 10485760  # 10MB
        
        with open(file_path, 'rb') as f:
            chunk_number = 0
            while True:
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break
                
                chunk_number += 1
                logger.info("Uploading chunk %d", chunk_number)
                
                # Upload chunk
                files = {'video': (f'chunk_{chunk_number}', chunk_data, 'application/octet-stream')}
                data = {
                    'upload_id': upload_id,
                    'chunk_number': chunk_number
                }
                
                response = requests.post(upload_url, files=files, data=data)
                response.raise_for_status()
                
                chunk_result = response.json()
                if chunk_result.get('error', {}).get('code') != 'ok':
                    raise Exception(f"Chunk upload failed: {chunk_result}")
    
    def get_upload_status(self, upload_id: str) -> Dict[Any, Any]:
        """
        Check the status of an upload.
        
        Args:
            upload_id: Upload ID returned from upload_video
            
        Returns:
            Status information
        """
        return self._make_request('GET', f'/v2/post/publish/status/?upload_id={upload_id}')
    
    def get_user_info(self) -> Dict[Any, Any]:
        """Get current user information."""
        return self._make_request('GET', '/v2/user/info/')
    
    def get_video_list(self, max_count: int = 20) -> Dict[Any, Any]:
        """
        Get list of user's videos.
        
        Args:
            max_count: Maximum number of videos to retrieve
            
        Returns:
            List of videos
        """
        params = {'max_count': min(max_count, 20)}  # API limit
        return self._make_request('GET', '/v2/video/list/', params=params)