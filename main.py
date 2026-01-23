#!/usr/bin/env python3
"""Command-line interface for TikTok auto-posting tool."""

import click
import os
import sys
import json
from typing import Optional

from config import config
from tiktok_client import TikTokUploader
from video_processor import VideoProcessor

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """TikTok Auto-Posting Tool
    
    A tool for automatically uploading videos to TikTok.
    """
    pass

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--caption', '-c', default='', help='Video caption/description')
@click.option('--privacy', '-p', type=click.Choice(['public', 'private', 'friends']), 
              help='Privacy setting (default from config)')
@click.option('--disable-duet', is_flag=True, help='Disable duet feature')
@click.option('--disable-comment', is_flag=True, help='Disable comments')
@click.option('--disable-stitch', is_flag=True, help='Disable stitch feature')
@click.option('--validate-only', is_flag=True, help='Only validate video without uploading')
def upload(video_path: str, caption: str, privacy: Optional[str], 
           disable_duet: bool, disable_comment: bool, disable_stitch: bool, 
           validate_only: bool):
    """Upload a video to TikTok."""
    
    try:
        # Initialize video processor
        processor = VideoProcessor()
        
        # Validate video
        click.echo(f"Validating video: {video_path}")
        is_valid, errors = processor.validate_video(video_path)
        
        if not is_valid:
            click.echo("❌ Video validation failed:", err=True)
            for error in errors:
                click.echo(f"  • {error}", err=True)
            sys.exit(1)
        
        # Show video info
        info = processor.get_video_info(video_path)
        click.echo("✅ Video validation passed!")
        click.echo(f"📊 Video info:")
        click.echo(f"  • Size: {info['size_mb']:.2f} MB")
        click.echo(f"  • Duration: {info['duration']:.2f} seconds")
        click.echo(f"  • Resolution: {info['width']}x{info['height']}")
        click.echo(f"  • Aspect ratio: {info['aspect_ratio']:.2f}")
        click.echo(f"  • Format: {info['format']}")
        
        if validate_only:
            click.echo("✅ Validation complete (upload skipped)")
            return
        
        # Check configuration
        if not config.is_configured():
            click.echo("❌ TikTok API not configured. Please set up your .env file.", err=True)
            click.echo("Copy .env.example to .env and fill in your API credentials.")
            sys.exit(1)
        
        # Initialize uploader
        click.echo("🚀 Starting upload...")
        uploader = TikTokUploader()
        
        # Upload video
        result = uploader.upload_video(
            video_path=video_path,
            caption=caption,
            privacy=privacy,
            disable_duet=disable_duet,
            disable_comment=disable_comment,
            disable_stitch=disable_stitch
        )
        
        click.echo("🎉 Upload completed successfully!")
        
        # Show result
        if result.get('data', {}).get('video_id'):
            video_id = result['data']['video_id']
            click.echo(f"📹 Video ID: {video_id}")
        
        if result.get('data', {}).get('share_url'):
            share_url = result['data']['share_url']
            click.echo(f"🔗 Share URL: {share_url}")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
def validate(video_path: str):
    """Validate a video file for TikTok upload."""
    
    try:
        processor = VideoProcessor()
        
        click.echo(f"Validating video: {video_path}")
        is_valid, errors = processor.validate_video(video_path)
        
        if is_valid:
            click.echo("✅ Video validation passed!")
            
            # Show detailed info
            info = processor.get_video_info(video_path)
            click.echo(f"📊 Video details:")
            click.echo(f"  • File size: {info['size_mb']:.2f} MB")
            click.echo(f"  • Duration: {info['duration']:.2f} seconds")
            click.echo(f"  • Resolution: {info['width']}x{info['height']}")
            click.echo(f"  • Aspect ratio: {info['aspect_ratio']:.2f}")
            click.echo(f"  • Format: {info['format']}")
            
        else:
            click.echo("❌ Video validation failed:")
            for error in errors:
                click.echo(f"  • {error}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
def config_check():
    """Check TikTok API configuration."""
    
    click.echo("🔧 Checking configuration...")
    
    errors = config.validate()
    
    if not errors:
        click.echo("✅ Configuration is valid!")
        click.echo(f"📊 Settings:")
        click.echo(f"  • Max video size: {config.max_video_size_mb} MB")
        click.echo(f"  • Max duration: {config.max_duration_seconds} seconds")
        click.echo(f"  • Default privacy: {config.default_privacy}")
        click.echo(f"  • Auto caption: {config.auto_caption}")
        click.echo(f"  • Supported formats: {', '.join(config.supported_formats)}")
        
        # Test API connection
        try:
            uploader = TikTokUploader()
            user_info = uploader.get_user_info()
            
            if user_info.get('error', {}).get('code') == 'ok':
                click.echo("✅ API connection successful!")
                display_name = user_info.get('data', {}).get('user', {}).get('display_name', 'Unknown')
                click.echo(f"👤 Connected as: {display_name}")
            else:
                click.echo("❌ API connection failed")
                click.echo(f"Error: {user_info}")
                
        except Exception as e:
            click.echo(f"❌ API connection failed: {str(e)}")
    else:
        click.echo("❌ Configuration errors:")
        for error in errors:
            click.echo(f"  • {error}")
        
        click.echo("\n💡 To fix:")
        click.echo("1. Copy .env.example to .env")
        click.echo("2. Fill in your TikTok API credentials")
        click.echo("3. Run this command again to verify")

@cli.command()
def user_info():
    """Get current user information from TikTok API."""
    
    try:
        if not config.is_configured():
            click.echo("❌ TikTok API not configured. Run 'config-check' for help.", err=True)
            sys.exit(1)
        
        uploader = TikTokUploader()
        user_info = uploader.get_user_info()
        
        if user_info.get('error', {}).get('code') == 'ok':
            user_data = user_info.get('data', {}).get('user', {})
            
            click.echo("👤 User Information:")
            click.echo(f"  • Display Name: {user_data.get('display_name', 'N/A')}")
            click.echo(f"  • Username: {user_data.get('username', 'N/A')}")
            click.echo(f"  • Follower Count: {user_data.get('follower_count', 'N/A')}")
            click.echo(f"  • Following Count: {user_data.get('following_count', 'N/A')}")
            click.echo(f"  • Video Count: {user_data.get('video_count', 'N/A')}")
        else:
            click.echo(f"❌ Failed to get user info: {user_info}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--count', '-n', default=10, help='Number of videos to list (max 20)')
def list_videos(count: int):
    """List user's recent videos."""
    
    try:
        if not config.is_configured():
            click.echo("❌ TikTok API not configured. Run 'config-check' for help.", err=True)
            sys.exit(1)
        
        uploader = TikTokUploader()
        videos = uploader.get_video_list(max_count=min(count, 20))
        
        if videos.get('error', {}).get('code') == 'ok':
            video_list = videos.get('data', {}).get('videos', [])
            
            if not video_list:
                click.echo("📹 No videos found")
                return
            
            click.echo(f"📹 Recent videos ({len(video_list)}):")
            for i, video in enumerate(video_list, 1):
                title = video.get('title', 'No title')
                video_id = video.get('id', 'Unknown ID')
                view_count = video.get('view_count', 'N/A')
                create_time = video.get('create_time', 'Unknown')
                
                click.echo(f"  {i}. {title}")
                click.echo(f"     ID: {video_id}")
                click.echo(f"     Views: {view_count}")
                click.echo(f"     Created: {create_time}")
                click.echo()
        else:
            click.echo(f"❌ Failed to get videos: {videos}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()