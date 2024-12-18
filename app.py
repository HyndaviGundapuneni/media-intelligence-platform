import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import pytube
import whisper
import face_recognition
import geopy
from geopy.geocoders import Nominatim
from googletrans import Translator
import re

class MediaIntelligencePlatform:
    def __init__(self):
        # Load pre-trained models
        self.whisper_model = whisper.load_model("base")
        self.geolocator = Nominatim(user_agent="media_intelligence_platform")
        self.translator = Translator()

    def process_image(self, image_path):
        """
        Process image to detect people and location with advanced features
        """
        # Load image
        img = face_recognition.load_image_file(image_path)
        
        # Detect faces
        face_locations = face_recognition.face_locations(img)
        people_count = len(face_locations)
        
        # Try to extract GPS location from image metadata
        try:
            from PIL.ExifTags import TAGS, GPSTAGS
            from PIL import Image as PILImage

            img_pil = PILImage.open(image_path)
            exif_data = img_pil._getexif()
            
            # Extract GPS info
            gps_info = {}
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo':
                        gps_tags = {}
                        for t in value:
                            sub_tag = GPSTAGS.get(t, t)
                            gps_tags[sub_tag] = value[t]
                        
                        # Convert GPS coordinates
                        if 'GPSLatitude' in gps_tags and 'GPSLongitude' in gps_tags:
                            lat = self.convert_gps_coordinates(gps_tags['GPSLatitude'], gps_tags['GPSLatitudeRef'])
                            lon = self.convert_gps_coordinates(gps_tags['GPSLongitude'], gps_tags['GPSLongitudeRef'])
                            
                            # Reverse geocode
                            location = self.geolocator.reverse(f"{lat}, {lon}")
                            return {
                                'people_count': people_count,
                                'location': location.address if location else "Unknown Location",
                                'coordinates': {'lat': lat, 'lon': lon}
                            }
        except Exception as e:
            st.warning(f"Could not extract GPS information: {e}")
        
        return {
            'people_count': people_count,
            'location': "Location not detected",
            'coordinates': None
        }

    def convert_gps_coordinates(self, coord, ref):
        """
        Convert GPS coordinates from EXIF format
        """
        degrees = coord[0]
        minutes = coord[1]
        seconds = coord[2]
        
        # Convert to decimal degrees
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Apply reference (N/S, E/W)
        if ref in ['S', 'W']:
            decimal = -decimal
        
        return decimal

    def process_video(self, video_path):
        """
        Enhanced video processing with word-level timestamps
        """
        # Transcribe video with word-level timestamps
        result = self.whisper_model.transcribe(video_path, word_timestamps=True)
        
        # Process word-level timestamps
        word_timestamps = []
        for segment in result['segments']:
            for word in segment['words']:
                word_timestamps.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
        
        # Generate highlights
        highlights = self._generate_highlights(result['text'])
        
        # Detect faces in video
        video = cv2.VideoCapture(video_path)
        face_locations_list = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_locations_list.extend(face_locations)
        
        video.release()
        
        return {
            'transcript': result['text'],
            'word_timestamps': word_timestamps,
            'highlights': highlights,
            'people_count': len(set(face_locations_list))
        }

    def search_in_transcript(self, word_timestamps, search_word):
        """
        Find all occurrences of a word in the transcript
        """
        results = []
        for entry in word_timestamps:
            if search_word.lower() in entry['word'].lower():
                results.append(entry)
        return results

    def translate_text(self, text, target_language='es'):
        """
        Translate text to target language
        """
        try:
            translation = self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            st.error(f"Translation error: {e}")
            return text

    def _generate_highlights(self, transcript, num_highlights=3):
        """
        Generate video highlights based on transcript
        Uses a simple extraction method
        """
        # Split into sentences and select most informative
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
        
        # In a real-world scenario, you'd use more advanced NLP techniques
        highlights = sentences[:num_highlights]
        
        return [
            {
                'timestamp': f'{i*30}s', 
                'description': sentence
            } for i, sentence in enumerate(highlights)
        ]

def main():
    st.set_page_config(
        page_title="Comprehensive Media Intelligence Platform", 
        page_icon="🤖", 
        layout="wide"
    )

    st.title("🤖 Comprehensive Media Intelligence Platform")
    
    # Initialize the platform
    platform = MediaIntelligencePlatform()

    # Language selection for translation
    LANGUAGES = {
        'en': 'English', 
        'es': 'Spanish', 
        'fr': 'French', 
        'de': 'German', 
        'it': 'Italian', 
        'ja': 'Japanese', 
        'ko': 'Korean', 
        'zh-cn': 'Chinese (Simplified)', 
        'ar': 'Arabic', 
        'ru': 'Russian'
    }

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image or Video", 
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov']
    )

    # YouTube URL input
    youtube_url = st.text_input("Or paste a YouTube video URL")

    # Process file
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        file_type = uploaded_file.type.split('/')[0]

        # Display file
        if file_type == 'image':
            st.image(temp_file_path, caption='Uploaded Image')
            
            # Process image
            with st.spinner('Analyzing image...'):
                image_analysis = platform.process_image(temp_file_path)
            
            # Display results
            st.subheader("Image Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("People Detected", image_analysis['people_count'])
            with col2:
                st.metric("Location", image_analysis['location'])

        elif file_type == 'video':
            st.video(temp_file_path)
            
            # Process video
            with st.spinner('Analyzing video...'):
                video_analysis = platform.process_video(temp_file_path)
            
            # Transcript and Translation Tabs
            tab1, tab2, tab3 = st.tabs(["Overview", "Transcript Search", "Transcript Translation"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("People Detected", video_analysis['people_count'])
                with col2:
                    st.metric("Highlights", len(video_analysis['highlights']))
                
                st.subheader("Video Highlights")
                for highlight in video_analysis['highlights']:
                    st.markdown(f"**{highlight['timestamp']}**: {highlight['description']}")

            with tab2:
                # Word search in transcript
                search_word = st.text_input("Search a word in transcript")
                
                if search_word:
                    # Find word occurrences
                    word_results = platform.search_in_transcript(
                        video_analysis['word_timestamps'], 
                        search_word
                    )
                    
                    if word_results:
                        st.subheader(f"Occurrences of '{search_word}'")
                        
                        # Create a video with custom start time
                        selected_time = st.select_slider(
                            "Select occurrence", 
                            options=word_results,
                            format_func=lambda x: f"{x['word']} (at {x['start']:.2f}s)"
                        )
                        
                        # Video with custom start time
                        st.video(temp_file_path, start_time=selected_time['start'])
                        
                        # Detailed word occurrences
                        for result in word_results:
                            st.markdown(f"**{result['word']}** at {result['start']:.2f}s")
                    else:
                        st.warning(f"No occurrences of '{search_word}' found")

            with tab3:
                # Transcript Translation
                target_language = st.selectbox(
                    "Select Translation Language", 
                    list(LANGUAGES.keys()), 
                    format_func=lambda x: LANGUAGES[x]
                )
                
                # Translate transcript
                translated_transcript = platform.translate_text(
                    video_analysis['transcript'], 
                    target_language
                )
                
                st.subheader(f"Transcript in {LANGUAGES[target_language]}")
                st.text_area(
                    "Translated Transcript", 
                    translated_transcript, 
                    height=300
                )

        # Clean up temporary file
        os.unlink(temp_file_path)

    # Process YouTube URL
    elif youtube_url:
        try:
            # Download YouTube video
            with st.spinner('Downloading YouTube video...'):
                yt = pytube.YouTube(youtube_url)
                video = yt.streams.filter(progressive=True, file_extension='mp4').first()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    video.download(output_path=tempfile.gettempdir(), filename=temp_file.name)
                    video_path = os.path.join(tempfile.gettempdir(), temp_file.name)

            # Display video
            st.video(video_path)
            
            # Process video
            with st.spinner('Analyzing video...'):
                video_analysis = platform.process_video(video_path)
            
            # Transcript and Translation Tabs
            tab1, tab2, tab3 = st.tabs(["Overview", "Transcript Search", "Transcript Translation"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("People Detected", video_analysis['people_count'])
                with col2:
                    st.metric("Highlights", len(video_analysis['highlights']))
                
                st.subheader("Video Highlights")
                for highlight in video_analysis['highlights']:
                    st.markdown(f"**{highlight['timestamp']}**: {highlight['description']}")

            with tab2:
                # Word search in transcript
                search_word = st.text_input("Search a word in transcript")
                
                if search_word:
                    # Find word occurrences
                    word_results = platform.search_in_transcript(
                        video_analysis['word_timestamps'], 
                        search_word
                    )
                    
                    if word_results:
                        st.subheader(f"Occurrences of '{search_word}'")
                        
                        # Create a video with custom start time
                        selected_time = st.select_slider(
                            "Select occurrence", 
                            options=word_results,
                            format_func=lambda x: f"{x['word']} (at {x['start']:.2f}s)"
                        )
                        
                        # Video with custom start time
                        st.video(video_path, start_time=selected_time['start'])
                        
                        # Detailed word occurrences
                        for result in word_results:
                            st.markdown(f"**{result['word']}** at {result['start']:.2f}s")
                    else:
                        st.warning(f"No occurrences of '{search_word}' found")

            with tab3:
                # Transcript Translation
                target_language = st.selectbox(
                    "Select Translation Language", 
                    list(LANGUAGES.keys()), 
                    format_func=lambda x: LANGUAGES[x]
                )
                
                # Translate transcript
                translated_transcript = platform.translate_text(
                    video_analysis['transcript'], 
                    target_language
                )
                
                st.subheader(f"Transcript in {LANGUAGES[target_language]}")
                st.text_area(
                    "Translated Transcript", 
                    translated_transcript, 
                    height=300
                )

            # Clean up video file
            os.unlink(video_path)

        except Exception as e:
            st.error(f"Error processing YouTube video: {e}")

if __name__ == "__main__":
    main()