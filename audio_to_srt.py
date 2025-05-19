import os
import torch
import torchaudio
import numpy as np
import re
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datetime import timedelta
import argparse
import sys

def format_timestamp(seconds):
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg"""
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    print(f"📼 Extracting audio from {video_path} to {audio_path}...")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Error: ffmpeg is not installed or not in PATH. Please install ffmpeg.")
        sys.exit(1)
    
    # Extract audio using ffmpeg
    try:
        cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"✅ Audio extracted to: {audio_path}")
        return audio_path
    except subprocess.SubprocessError as e:
        print(f"❌ Error extracting audio: {e}")
        sys.exit(1)

def load_audio(file_path, target_sr=16000):
    """Load audio file and return as numpy array"""
    print(f"🎧 Loading audio: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    # Get file size to validate
    file_size = os.path.getsize(file_path)
    print(f"📊 Audio file size: {file_size/1024/1024:.2f} MB")
    if file_size < 10000:  # Less than 10KB is suspicious
        print(f"⚠️ Warning: Audio file is very small ({file_size} bytes). It might be corrupted or empty.")
    
    # Load audio file with torchaudio
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Verify audio has content
        if waveform.numel() == 0 or waveform.abs().max().item() < 0.001:
            print(f"⚠️ Warning: Audio appears to be silent or extremely quiet. Max amplitude: {waveform.abs().max().item()}")
            
            # Try to debug and fix
            print("🔍 Trying to diagnose audio file...")
            try:
                info = torchaudio.info(file_path)
                print(f"Audio info: {info}")
            except Exception as e:
                print(f"Could not get audio info: {e}")
                
            # Try direct FFmpeg inspection
            try:
                cmd = ["ffmpeg", "-i", file_path, "-hide_banner"]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(f"FFmpeg info: {result.stderr}")
            except Exception as e:
                print(f"FFmpeg inspection failed: {e}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            print(f"🔄 Converting from {waveform.shape[0]} channels to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            print(f"🔄 Resampling audio from {sample_rate}Hz to {target_sr}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy array
        audio_array = waveform.squeeze().numpy()
        audio_length_seconds = len(audio_array) / target_sr
        
        # Normalize audio if it's very quiet
        if abs(audio_array).max() < 0.1:
            print(f"🔄 Audio is quiet (max amplitude: {abs(audio_array).max():.4f}), normalizing...")
            if abs(audio_array).max() > 0:  # Make sure we don't divide by zero
                audio_array = audio_array / abs(audio_array).max() * 0.7
        
        print(f"✅ Audio loaded successfully: {audio_length_seconds:.2f} seconds, mean amplitude: {abs(audio_array).mean():.6f}")
        
        # Plot amplitude histogram for debugging (text-based)
        hist, bins = np.histogram(audio_array, bins=10, range=(-1, 1))
        total = sum(hist)
        if total > 0:
            print("📊 Audio amplitude distribution:")
            for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
                bin_end = bins[i+1]
                percentage = count / total * 100
                bar = "█" * int(percentage / 5)
                print(f"  [{bin_start:.1f} to {bin_end:.1f}]: {bar} {percentage:.1f}%")
        
        return audio_array, target_sr, audio_length_seconds
    
    except Exception as e:
        print(f"❌ Error loading audio with torchaudio: {e}")
        
        # Fallback to ffmpeg if torchaudio fails
        try:
            print("🔄 Trying to load audio with ffmpeg...")
            temp_wav = file_path + ".temp.wav"
            cmd = ["ffmpeg", "-i", file_path, "-ar", str(target_sr), "-ac", "1", temp_wav, "-y"]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Load the wav file
            waveform, sample_rate = torchaudio.load(temp_wav)
            audio_array = waveform.squeeze().numpy()
            audio_length_seconds = len(audio_array) / target_sr
            
            # Normalize audio if it's very quiet
            if abs(audio_array).max() < 0.1:
                print(f"🔄 Audio is quiet (max amplitude: {abs(audio_array).max():.4f}), normalizing...")
                if abs(audio_array).max() > 0:  # Make sure we don't divide by zero
                    audio_array = audio_array / abs(audio_array).max() * 0.7
            
            # Remove temp file
            os.remove(temp_wav)
            
            print(f"✅ Audio loaded successfully with ffmpeg: {audio_length_seconds:.2f} seconds")
            return audio_array, target_sr, audio_length_seconds
        except Exception as e2:
            print(f"❌ All audio loading methods failed: {e2}")
            
            # One last attempt with a different method
            try:
                print("🔄 Trying one last method to convert the audio...")
                # Try converting to wav with a different method
                output_wav = file_path + ".emergency.wav"
                os.system(f'ffmpeg -y -i "{file_path}" -acodec pcm_s16le -ar {target_sr} -ac 1 "{output_wav}"')
                
                if os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
                    waveform, sample_rate = torchaudio.load(output_wav)
                    audio_array = waveform.squeeze().numpy()
                    audio_length_seconds = len(audio_array) / target_sr
                    os.remove(output_wav)
                    print(f"✅ Emergency method worked! Audio length: {audio_length_seconds:.2f} seconds")
                    return audio_array, target_sr, audio_length_seconds
                else:
                    print("❌ Emergency conversion failed or produced tiny file")
            except Exception as e3:
                print(f"❌ Emergency method failed: {e3}")
            
            print("\n❌ CRITICAL ERROR: Could not load the audio file after multiple attempts.")
            print("Please check if the file is a valid audio file and not corrupted.")
            print("You might try converting it first with something like:")
            print(f"ffmpeg -i {file_path} -ar 16000 -ac 1 converted_audio.wav")
            sys.exit(1)

def process_audio_with_whisper(audio_array, sample_rate, language="ru", task="translate"):
    """Process audio with Whisper model and return segments with timestamps"""
    print(f"🔄 Loading Whisper large-v2 model...")
    
    try:
        # Load processor and model
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        model.eval()
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        use_half = device == "cuda"
        if use_half:
            model = model.half()  # Use FP16 precision for faster inference
        
        print(f"✅ Model loaded on {device}" + (" with FP16 precision" if use_half else ""))
        
        # Process audio in chunks to avoid memory issues
        max_length_seconds = 3  # Process 3 seconds at a time
        segments = []
        
        if len(audio_array) / sample_rate > max_length_seconds:
            print(f"🔄 Processing long audio in chunks of {max_length_seconds} seconds...")
            chunk_size = int(max_length_seconds * sample_rate)
            total_chunks = (len(audio_array) // chunk_size) + 1
            
            for i in range(0, len(audio_array), chunk_size):
                chunk_index = i // chunk_size + 1
                chunk_start_time = i / sample_rate
                print(f"🔄 Processing chunk {chunk_index}/{total_chunks} starting at {chunk_start_time:.2f}s...")
                
                chunk_end = min(i + chunk_size, len(audio_array))
                audio_chunk = audio_array[i:chunk_end]
                
                # Process this chunk
                chunk_segments = process_chunk(audio_chunk, sample_rate, processor, model, 
                                              device, use_half, chunk_start_time, language, task)
                
                # Even if no segments were created but we have text, create a dummy segment
                if len(chunk_segments) == 0:
                    print("⚠️ No segments detected but trying to create a manual segment...")
                    
                    # Get the raw output directly
                    with torch.no_grad():
                        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
                        inputs = processor(
                            audio_chunk,
                            sampling_rate=sample_rate,
                            return_tensors="pt",
                            return_attention_mask=True
                        )
                        input_features = inputs.input_features
                        attention_mask = inputs.attention_mask if hasattr(inputs, "attention_mask") else None
                        
                        # Match the precision (float16 vs float32)
                        if use_half:
                            input_features = input_features.half()
                            if attention_mask is not None:
                                attention_mask = attention_mask.half()
                        else:
                            input_features = input_features.float()
                            if attention_mask is not None:
                                attention_mask = attention_mask.float()
                        
                        # Move to device
                        input_features = input_features.to(device)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        
                        try:
                            predicted_ids = model.generate(
                                input_features,
                                attention_mask=attention_mask,
                                return_timestamps=False,  # Don't require timestamps
                                forced_decoder_ids=forced_decoder_ids,
                                num_beams=2
                            )
                        except Exception as e:
                            print(f"⚠️ Error in fallback generate: {e}")
                            # Try one last time with different settings
                            try:
                                # Reset model to CPU with float32 for maximum compatibility
                                temp_model = model.to("cpu").float()
                                input_features = input_features.to("cpu").float()
                                if attention_mask is not None:
                                    attention_mask = attention_mask.to("cpu").float()
                                
                                predicted_ids = temp_model.generate(
                                    input_features,
                                    attention_mask=attention_mask,
                                    return_timestamps=False,
                                    forced_decoder_ids=forced_decoder_ids,
                                    num_beams=1
                                )
                                
                                # Move model back to original device and precision
                                model.to(device)
                                if use_half:
                                    model.half()
                            except Exception as e2:
                                print(f"❌ Final fallback also failed: {e2}")
                                # Create a dummy segment with an error message
                                chunk_duration = len(audio_chunk) / sample_rate
                                chunk_segments = [{
                                    "timestamp": (chunk_start_time, chunk_start_time + chunk_duration),
                                    "text": "[Translation failed due to technical error]"
                                }]
                                segments.extend(chunk_segments)
                                print(f"⚠️ Added error placeholder for chunk at {chunk_start_time:.2f}s")
                                continue
                        
                        # Get transcription without timestamps
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        
                        if transcription.strip():
                            print(f"📝 Manual transcription: {transcription[:100]}...")
                            # Create a dummy segment for this chunk
                            chunk_duration = len(audio_chunk) / sample_rate
                            chunk_segments = [{
                                "timestamp": (chunk_start_time, chunk_start_time + chunk_duration),
                                "text": transcription.strip()
                            }]
                            print(f"✅ Created manual segment: {chunk_start_time:.2f}s-{chunk_start_time + chunk_duration:.2f}s")
                
                segments.extend(chunk_segments)
                print(f"  ✅ Processed chunk with {len(chunk_segments)} segments")
        else:
            print("🔄 Processing entire audio at once...")
            segments = process_chunk(audio_array, sample_rate, processor, model, 
                                    device, use_half, 0.0, language, task)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["timestamp"][0])
        print(f"✅ Total segments generated: {len(segments)}")
        
        return segments
    
    except Exception as e:
        print(f"❌ Error processing audio with Whisper: {e}")
        sys.exit(1)

def process_chunk(audio_chunk, sample_rate, processor, model, device, use_half, 
                 chunk_offset=0.0, language="ru", task="translate"):
    """Process a single audio chunk with Whisper"""
    try:
        # Prepare input features
        inputs = processor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True  # Explicitly request attention mask
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask if hasattr(inputs, "attention_mask") else None
        
        if use_half:
            input_features = input_features.half()
            if attention_mask is not None:
                attention_mask = attention_mask.half()
        
        input_features = input_features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Generate translation with timestamps
        with torch.no_grad():
            # Check if language is supported, default to English if not
            try:
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
                print(f"DEBUG - Using language: {language}, task: {task}")
            except Exception as e:
                print(f"DEBUG - Error with language {language}: {e}")
                print(f"DEBUG - Falling back to English")
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task=task)
            
            # Print first 10 decoder IDs for debugging
            print(f"DEBUG - First few decoder IDs: {forced_decoder_ids[:10] if forced_decoder_ids else 'None'}")
            
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                return_timestamps=True,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=2,  # Increase beam size slightly
                temperature=0.0,  # Use deterministic output for debugging
                do_sample=False  # Disable sampling for debugging
            )
        
        # Decode output
        output = processor.batch_decode(predicted_ids, return_timestamps=True, skip_special_tokens=False)[0]
        
        # Debug: Print raw output
        print(f"DEBUG - Raw output type: {type(output)}")
        if isinstance(output, dict):
            print(f"DEBUG - Output keys: {list(output.keys())}")
            if "chunks" in output:
                print(f"DEBUG - Number of chunks: {len(output['chunks'])}")
                if len(output["chunks"]) > 0:
                    print(f"DEBUG - First chunk: {output['chunks'][0]}")
        else:
            raw_text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
            print(f"DEBUG - Raw text excerpt: {raw_text[:200]}...")
            
        # Also print a spectrogram summary of the audio chunk to check if it contains audio
        chunk_energy = np.abs(audio_chunk).mean()
        chunk_max = np.abs(audio_chunk).max()
        print(f"DEBUG - Audio chunk energy: mean={chunk_energy:.6f}, max={chunk_max:.6f}")
        
        # Extract segments
        segments = []
        print(f"DEBUG - Initializing segments list")
        
        # Handle different output formats
        if isinstance(output, dict) and "chunks" in output:
            for chunk in output["chunks"]:
                start_time = chunk["timestamp"][0] + chunk_offset
                end_time = chunk["timestamp"][1] + chunk_offset
                text = chunk["text"].strip()
                
                if text:  # Skip empty segments
                    segments.append({
                        "timestamp": (start_time, end_time),
                        "text": text
                    })
                else:
                    # Parse timestamps manually
                    text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
                    
                    # Since Whisper isn't returning timestamps in the expected format,
                    # we need to create artificial timestamps for the entire chunk
                    
                    # Print full text for debugging
                    print(f"DEBUG - Full translated text: {text}")
                    
                    # Clean up text by removing special tokens
                    print(f"DEBUG - Before cleanup: {text[:100]}...")
                    
                    # First try with a more targeted pattern to remove just the special tokens
                    clean_text = re.sub(r'<\|startoftranscript\|>', '', text)
                    clean_text = re.sub(r'<\|ru\|>', '', clean_text)
                    clean_text = re.sub(r'<\|translate\|>', '', clean_text)
                    clean_text = re.sub(r'<\|endoftext\|>', '', clean_text)
                    clean_text = re.sub(r'<\|notimestamps\|>', '', clean_text)
                    
                    # Then handle any other special tokens generally
                    clean_text = re.sub(r'<\|[^|]+\|>', '', clean_text)
                    clean_text = clean_text.strip()
                    
                    print(f"DEBUG - After cleanup: {clean_text[:100]}...")
                    print(f"DEBUG - Clean text length: {len(clean_text)}")
                    
                    # If we have text content, create a segment for the entire chunk
                    if clean_text:
                        print(f"DEBUG - Creating segments from clean text")
                        # Create a segment spanning the entire chunk with a small margin
                        chunk_duration = len(audio_chunk) / sample_rate
                        start_time = chunk_offset
                        end_time = chunk_offset + chunk_duration
                        print(f"DEBUG - Chunk duration: {chunk_duration:.2f}s, start: {start_time:.2f}s, end: {end_time:.2f}s")
                        
                        # Create artificial segments by splitting on sentences
                        # Split text into sentences
                        sentence_pattern = re.compile(r'([.!?])\s+')
                        sentences = sentence_pattern.sub(r'\1\n', clean_text).split('\n')
                        
                        # Filter out empty sentences
                        sentences = [s.strip() for s in sentences if s.strip()]
                        
                        if sentences:
                            print(f"DEBUG - Split into {len(sentences)} sentences")
                            
                            # Divide the chunk duration among the sentences
                            segment_duration = chunk_duration / len(sentences)
                            
                            for i, sentence in enumerate(sentences):
                                segment_start = start_time + (i * segment_duration)
                                segment_end = start_time + ((i + 1) * segment_duration)
                                
                                if sentence:  # Skip empty segments
                                    segments.append({
                                        "timestamp": (segment_start, segment_end),
                                        "text": sentence
                                    })
                                    print(f"DEBUG - Created segment: {segment_start:.2f}s-{segment_end:.2f}s: {sentence}")
                        else:
                            # Just one segment for the whole chunk
                            segments.append({
                                "timestamp": (start_time, end_time),
                                "text": clean_text
                            })
                            print(f"DEBUG - Created single segment: {start_time:.2f}s-{end_time:.2f}s")
                    else:
                        print("DEBUG - No text content in this chunk")
        
        return segments
    
    except Exception as e:
        print(f"❌ Error processing chunk: {e}")
        return []

def write_srt_file(segments, output_path):
    """Write segments to SRT file"""
    print(f"💾 Writing SRT file to {output_path}...")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = format_timestamp(segment["timestamp"][0])
                end = format_timestamp(segment["timestamp"][1])
                text = segment["text"]
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        
        print(f"✅ SRT file written successfully with {len(segments)} entries")
    except Exception as e:
        print(f"❌ Error writing SRT file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate SRT subtitles from video or audio files using Whisper")
    parser.add_argument("input_file", help="Path to the input video or audio file")
    parser.add_argument("--language", "-l", default="ru", help="Language code for transcription/translation")
    parser.add_argument("--task", "-t", default="translate", choices=["transcribe", "translate"],
                        help="Task: transcribe (original language) or translate (to English)")
    parser.add_argument("--output", "-o", help="Output SRT file path (default: input_file_name.srt)")
    
    args = parser.parse_args()
    
    # Determine if input is video or audio
    input_ext = os.path.splitext(args.input_file)[1].lower()
    
    if input_ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv']:
        # Video file - extract audio first
        audio_path = extract_audio_from_video(args.input_file)
    else:
        # Assume audio file
        audio_path = args.input_file
    
    # Set output path if not specified
    if not args.output:
        args.output = os.path.splitext(args.input_file)[0] + ".srt"
    
    # Load audio
    audio_array, sample_rate, audio_length = load_audio(audio_path)
    
    # Check if audio is valid
    if audio_length <= 0:
        print("❌ Error: Audio file appears to be empty or invalid")
        sys.exit(1)
    
    # Process with Whisper
    segments = process_audio_with_whisper(audio_array, sample_rate, args.language, args.task)
    
    # Write SRT file
    write_srt_file(segments, args.output)
    
    print(f"✅ Process completed. SRT file saved to: {args.output}")
    print(f"To add subtitles to your video, run:")
    print(f"ffmpeg -i {args.input_file} -i {args.output} -c copy -c:s mov_text {os.path.splitext(args.input_file)[0]}_subtitled{input_ext}")

if __name__ == "__main__":
    main() 