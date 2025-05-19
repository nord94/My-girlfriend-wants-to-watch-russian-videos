from pytube import YouTube
import os
import sys

def download_youtube_video(url, output_path=None):
    """
    Download a YouTube video to MP4 format.
    
    Args:
        url (str): YouTube video URL
        output_path (str, optional): Directory to save the video. Defaults to current directory.
    
    Returns:
        str: Path to the downloaded video file
    """
    try:
        # Create YouTube object
        yt = YouTube(url)
        
        # Get video information
        print(f"Title: {yt.title}")
        print(f"Length: {yt.length} seconds")
        print(f"Views: {yt.views}")
        
        # Get the highest resolution progressive stream (video with audio)
        video = yt.streams.get_highest_resolution()
        
        # Set output path
        if output_path is None:
            output_path = os.getcwd()
        
        # Download the video
        print("\nDownloading...")
        video_path = video.download(output_path)
        print(f"\nDownload completed! Video saved to: {video_path}")
        
        return video_path
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <youtube_url> [output_directory]")
        sys.exit(1)
    
    url = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    download_youtube_video(url, output_path)

if __name__ == "__main__":
    main() 