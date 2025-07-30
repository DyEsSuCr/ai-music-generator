# AI Music Generator

An AI-powered music generator that uses ACE-Step to create complete electronic rap tracks with custom lyrics, deployed on Modal.

## Features

- 🎵 **AI Music Generation**: Uses ACE-Step model to create high-quality music
- 🎤 **Custom Lyrics**: Support for lyrics with verse, chorus, and bridge structure
- ⚡ **GPU Accelerated**: Runs on L40S GPU for fast generation
- 🔄 **REST API**: FastAPI endpoint for easy integration
- 📦 **Auto Scaling**: Modal auto-scaling for efficient resource management
- 🎨 **Additional Models**: Includes Qwen2-7B for LLM and SDXL-Turbo for images

## Requirements

- Modal account
- Python 3.11+
- `requirements.txt` file with dependencies
- `prompts` folder with configuration files

## Setup

### 1. Modal Secrets

Create a Modal secret named `music-gen-secret`:

```bash
modal secret create music-gen-secret
```

### 2. Volumes

The project uses two Modal volumes:
- `ace-step-models`: For storing ACE-Step models
- `qwen-hf-cache`: For Hugging Face cache

These are created automatically if they don't exist.

### 3. Project Structure

```
.
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── prompts/               # Prompt configuration files
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DyEsSuCr/ai-music-generator
cd music-generator
```

2. Install Modal:
```bash
pip install modal
```

3. Authenticate with Modal:
```bash
modal auth new
```

4. Deploy the application:
```bash
modal deploy main.py
```

## Usage

### API Endpoint

Once deployed, the application exposes a POST endpoint for music generation:

```bash
curl -X POST <endpoint-url>/generate
```

### Local Execution

To test locally:

```bash
modal run main.py
```

This will generate an `ai-music.wav` file in the current directory.

### Customization

To customize the generated music, modify the parameters in the `generate()` function:

```python
self.music_model(
    prompt='electronic rap',           # Musical style
    lyrics="[verse]\n...",            # Custom lyrics
    audio_duration=180,               # Duration in seconds
    infer_step=60,                    # Inference steps
    guidance_scale=15,                # Guidance scale
    save_path=output_path,
)
```

## Configuration Parameters

- **prompt**: Description of the desired musical style
- **lyrics**: Lyrics with structure [verse], [chorus], [bridge]
- **audio_duration**: Audio duration in seconds (default: 180)
- **infer_step**: Number of inference steps (more steps = better quality)
- **guidance_scale**: Prompt adherence control (1-20)

## Lyrics Structure

Lyrics should follow this format:

```
[verse]
Verse content...

[chorus]
Chorus content...

[bridge]
Bridge content...
```

## Models Used

- **ACE-Step**: Main music generation model
- **Qwen2-7B-Instruct**: Large language model for text processing
- **SDXL-Turbo**: Image generation model for thumbnails

## Resources

- **GPU**: L40S for accelerated processing
- **Memory**: Auto-scaling based on demand
- **Storage**: Persistent volumes for models

## Limitations

- Requires GPU for optimal performance
- Models are downloaded on first run (may take time)
- Scale-down occurs after 15 seconds of inactivity

## Troubleshooting

### Insufficient memory error
- Reduce `infer_step` or `audio_duration`
- Verify L40S GPU availability

### Models not found
- Check that volumes are properly mounted
- Review Modal logs for download errors

### Authentication issues
- Verify `music-gen-secret` is configured
- Check Hugging Face permissions if needed

## API Response

The API returns a JSON response with base64-encoded audio data:

```json
{
  "audio_data": "UklGRiQAAABXQVZFZm10..."
}
```

To decode and save the audio:

```python
import base64

# Decode base64 audio data
audio_bytes = base64.b64decode(response_data['audio_data'])

# Save to file
with open('generated_music.wav', 'wb') as f:
    f.write(audio_bytes)
```

## Performance Optimization

- **Cold Start**: First request may take longer due to model loading
- **Warm Instances**: Subsequent requests are faster with cached models
- **Batch Processing**: Consider batching multiple requests for efficiency
