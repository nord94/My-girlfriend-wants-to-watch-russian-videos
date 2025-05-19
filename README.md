# She just doesn't speak Russian you know :/

This tool generates SRT subtitle files from video files using OpenAI's Whisper speech recognition model. It can transcribe audio in the original language or translate it to English.

## Requirements

Make sure you have the following installed:

- Python 3.8+ 
- PyTorch and TorchAudio
- Transformers (Hugging Face)
- ffmpeg (for audio extraction)
- pytube (for downloading videos)

Install dependencies:

```bash
pip install torch torchaudio transformers numpy pytube
```

Make sure ffmpeg is installed and available in your PATH:
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

## Usage

Run the script with a video file:

```bash
python audio_to_srt.py your_video.mp4
```

### Options:

- `--language`, `-l`: Language code for transcription (default: "ru" for Russian)
- `--task`, `-t`: Task to perform - "transcribe" (keep original language) or "translate" (to English)
- `--output`, `-o`: Output SRT file path (default: input_file_name.srt)

### Examples:

Translate Russian audio to English (default):
```bash
python audio_to_srt.py 4-russian-japanese-war.mp4
```

Transcribe without translation:
```bash
python audio_to_srt.py 4-russian-japanese-war.mp4 --task transcribe
```

Specify a different language (e.g., Japanese):
```bash
python audio_to_srt.py 4-russian-japanese-war.mp4 --language ja
```

Specify output file:
```bash
python audio_to_srt.py 4-russian-japanese-war.mp4 -o custom_output.srt
```

## Adding Subtitles to Video

After generating the SRT file, you can embed it into your video using ffmpeg. There are two methods:

### Method 1: Simple Embedding
This method preserves video quality but doesn't allow for subtitle styling:

```bash
ffmpeg -i your_video.mp4 -i your_video.srt -c copy -c:s mov_text your_video_subtitled.mp4
```

### Method 2: Styled Subtitles with Background
This method allows for custom styling but requires video re-encoding:

```bash
ffmpeg -i your_video.mp4 -vf "subtitles=your_video.srt:force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BackColour=&H80000000,Outline=1,Shadow=0,MarginV=25'" -c:a copy your_video_subtitled.mp4
```

#### Subtitle Styling Options
You can customize the appearance by modifying these parameters:
- `FontName`: Font family (e.g., Arial, Times New Roman)
- `FontSize`: Text size (e.g., 24, 28)
- `PrimaryColour`: Text color in hex (e.g., &HFFFFFF for white)
- `OutlineColour`: Outline color in hex (e.g., &H000000 for black)
- `BackColour`: Background color with transparency (e.g., &H80000000 for semi-transparent black)
  - First two digits after &H control transparency (00=transparent, FF=opaque)
- `Outline`: Outline thickness in pixels
- `Shadow`: Shadow effect (0=off, 1=on)
- `MarginV`: Vertical margin in pixels

For higher quality output, add `-crf 18` before the output filename. Lower CRF values mean higher quality (18 is visually lossless).

The script will show this command after it finishes processing.

## Troubleshooting

- If you encounter `Audio length: 0.00 seconds` error, the script will now try multiple methods to load the audio.
- For large video files, the script processes the audio in chunks to avoid memory issues.
- If subtitles aren't appearing correctly, check that the SRT file contains valid entries. 