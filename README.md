# TikTok Auto-Posting Tool

A Python tool for automatically uploading videos to TikTok using the TikTok API.

## Features

- 🎥 **Video Upload**: Upload videos to TikTok with custom captions and settings
- ✅ **Video Validation**: Validate video files before upload (size, format, duration)
- 🔧 **Configuration Management**: Easy setup with environment variables
- 📊 **Video Processing**: Get video information and create thumbnails
- 🎛️ **Privacy Controls**: Set privacy levels, disable duet/comments/stitch
- 💻 **Command-Line Interface**: Easy-to-use CLI with multiple commands
- 📋 **User Management**: Get user info and list uploaded videos

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tnq54/tnq.git
cd tnq
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your TikTok API credentials
```

## Configuration

Create a `.env` file with your TikTok API credentials:

```bash
# TikTok API Configuration
TIKTOK_API_KEY=your_api_key_here
TIKTOK_API_SECRET=your_api_secret_here
TIKTOK_ACCESS_TOKEN=your_access_token_here
TIKTOK_USER_ID=your_user_id_here

# Video Settings
MAX_VIDEO_SIZE_MB=50
MAX_DURATION_SECONDS=180

# Posting Settings
DEFAULT_PRIVACY=public
AUTO_CAPTION=true
```

### Getting TikTok API Credentials

1. Go to [TikTok for Developers](https://developers.tiktok.com/)
2. Create an app and get your API credentials
3. Generate an access token for your account
4. Add the credentials to your `.env` file

## Usage

### Command-Line Interface

The tool provides several commands through the CLI:

#### Upload a Video
```bash
python main.py upload path/to/video.mp4 --caption "My awesome video!" --privacy public
```

Options:
- `--caption, -c`: Video caption/description
- `--privacy, -p`: Privacy setting (public, private, friends)
- `--disable-duet`: Disable duet feature
- `--disable-comment`: Disable comments
- `--disable-stitch`: Disable stitch feature
- `--validate-only`: Only validate video without uploading

#### Validate a Video
```bash
python main.py validate path/to/video.mp4
```

#### Check Configuration
```bash
python main.py config-check
```

#### Get User Information
```bash
python main.py user-info
```

#### List Recent Videos
```bash
python main.py list-videos --count 10
```

### Programmatic Usage

You can also use the tool programmatically:

```python
from tiktok_client import TikTokUploader
from video_processor import VideoProcessor

# Validate video
processor = VideoProcessor()
is_valid, errors = processor.validate_video("video.mp4")

if is_valid:
    # Upload video
    uploader = TikTokUploader()
    result = uploader.upload_video(
        video_path="video.mp4",
        caption="My video caption",
        privacy="public"
    )
    print(f"Upload successful! Video ID: {result['data']['video_id']}")
```

## Video Requirements

- **Formats**: MP4, MOV, AVI, WebM
- **Size**: Maximum 50MB (configurable)
- **Duration**: Maximum 180 seconds (configurable)
- **Aspect Ratio**: Recommended 9:16 (vertical)
- **Resolution**: Any resolution supported by TikTok

## API Endpoints Used

The tool uses the following TikTok API endpoints:

- `/v2/post/publish/video/init/` - Initialize video upload
- `/v2/post/publish/video/commit/` - Commit video upload
- `/v2/post/publish/status/` - Check upload status
- `/v2/user/info/` - Get user information
- `/v2/video/list/` - List user videos

## Error Handling

The tool includes comprehensive error handling for:

- Invalid video files
- Network issues
- API errors
- Configuration problems
- File size/format restrictions

## Development

### Project Structure

```
tnq/
├── main.py              # CLI interface
├── tiktok_client.py     # TikTok API client
├── video_processor.py   # Video validation and processing
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment file
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **"Configuration errors"**: Make sure your `.env` file is properly set up with valid API credentials.

2. **"Video validation failed"**: Check that your video meets the requirements (size, format, duration).

3. **"API request failed"**: Verify your API credentials and internet connection.

4. **"Upload initialization failed"**: This usually indicates API credential issues or quota limits.

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
python main.py upload video.mp4
```

## License

This project is open source. Please check the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is for educational and legitimate use only. Make sure to comply with TikTok's Terms of Service and API usage policies. The authors are not responsible for any misuse of this tool.