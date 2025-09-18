"""Video processing utilities for TikTok auto-posting tool."""

import os
import mimetypes
from typing import Optional, Tuple
from PIL import Image
from config import config

class VideoProcessor:
    """Handles video validation, processing, and optimization for TikTok."""
    
    def __init__(self):
        self.supported_formats = config.supported_formats
        self.max_size_mb = config.max_video_size_mb
        self.max_duration = config.max_duration_seconds
    
    def validate_video(self, video_path: str) -> Tuple[bool, list[str]]:
        """
        Validate video file for TikTok upload.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if file exists
        if not os.path.exists(video_path):
            errors.append(f"Video file not found: {video_path}")
            return False, errors
        
        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            errors.append(f"Video file too large: {file_size_mb:.2f}MB (max: {self.max_size_mb}MB)")
        
        # Check file format by extension (simpler approach)
        file_ext = os.path.splitext(video_path)[1].lower().lstrip('.')
        if file_ext not in self.supported_formats:
            errors.append(f"Unsupported format: {file_ext} (supported: {', '.join(self.supported_formats)})")
        
        # Check mime type if available
        try:
            mime_type, _ = mimetypes.guess_type(video_path)
            if mime_type and not mime_type.startswith('video/'):
                errors.append(f"File does not appear to be a video: {mime_type}")
        except Exception:
            # Ignore mime type errors, file extension check is sufficient
            pass
        
        return len(errors) == 0, errors
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        info = {
            'path': video_path,
            'exists': os.path.exists(video_path),
            'size_mb': 0,
            'duration': 0,  # Will be 0 without moviepy
            'width': 0,     # Will be 0 without moviepy
            'height': 0,    # Will be 0 without moviepy
            'aspect_ratio': 0,
            'format': 'unknown'
        }
        
        if not info['exists']:
            return info
        
        # File size
        info['size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
        
        # File format from extension
        file_ext = os.path.splitext(video_path)[1].lower().lstrip('.')
        info['format'] = file_ext
        
        # Note: For full video analysis, install moviepy or ffmpeg-python
        # For now, we'll use basic file information
        
        return info
    
    def create_thumbnail(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Create a thumbnail from video.
        Note: This is a placeholder implementation. For actual thumbnail generation,
        install moviepy or use ffmpeg.
        
        Args:
            video_path: Path to the video file
            output_path: Optional output path for thumbnail
            
        Returns:
            Path to the created thumbnail
        """
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_thumbnail.jpg"
        
        # Placeholder: Create a simple image as thumbnail
        # In a real implementation, you would extract a frame from the video
        try:
            # Create a simple placeholder thumbnail
            img = Image.new('RGB', (720, 1280), color='gray')
            img.save(output_path, 'JPEG', quality=85)
            
        except Exception as e:
            raise Exception(f"Could not create thumbnail: {str(e)}")
        
        return output_path