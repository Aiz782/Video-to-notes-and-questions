import streamlit as st
from dotenv import load_dotenv
import os
import openai
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import pandas as pd
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load subtopics dataset
subtopics_df = pd.read_csv(r'C:\Users\LENOVO\Downloads\data\subtopics.csv')  # Ensure you have a subtopics.csv with columns 'id', 'topicId', and 'name'

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prompts for generating summaries and questions
summary_prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points in English language only
within 250 words. Please provide the summary of the text given here:  """

# Function to extract transcript details from a YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        return None

# Function to download audio from YouTube video using yt-dlp
def download_audio(youtube_video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_video_url, download=True)
        audio_file = ydl.prepare_filename(info_dict)
        audio_file = audio_file.replace('.webm', '.mp3').replace('.m4a', '.mp3')
    
    return audio_file

# Function to convert audio to text using OpenAI Whisper API
def audio_to_text(audio_file):
    audio = AudioSegment.from_mp3(audio_file)
    chunk_size = 10 * 60 * 1000  # 10 minutes
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    transcript = ""
    for i, chunk in enumerate(chunks):
        chunk.export(f"chunk_{i}.mp3", format="mp3")
        with open(f"chunk_{i}.mp3", "rb") as chunk_file:
            response = openai.Audio.transcribe("whisper-1", chunk_file)
            transcript += response['text'].strip() + " "

    return transcript.strip()

# Function to generate content based on a given prompt and text using Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to find matching topic with a threshold of 70% using sentence embeddings
def find_matching_topic(transcript_text, threshold=0.4):
    transcript_embedding = model.encode(transcript_text, convert_to_tensor=True)
    subtopics_embeddings = model.encode(subtopics_df['name'].tolist(), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(transcript_embedding, subtopics_embeddings)[0]
    best_match_idx = cosine_scores.argmax().item()
    best_match_score = cosine_scores[best_match_idx].item()

    if best_match_score >= threshold:
        return subtopics_df.iloc[best_match_idx][['id', 'topicId']]
    return None

# Streamlit App
st.title("YouTube Transcript to Detailed Notes and Topic Identifier")

youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("v=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Notes and Topic"):
    transcript_text = extract_transcript_details(youtube_link)
    if not transcript_text:
        st.write("Transcript not found. Generating transcript from audio...")
        audio_file = download_audio(youtube_link)
        transcript_text = audio_to_text(audio_file)
    
    if transcript_text:
        summary = generate_gemini_content(transcript_text, summary_prompt)
        st.markdown("## Detailed Notes:")
        st.write(summary)

        matched_topic = find_matching_topic(transcript_text)
        
        if matched_topic is not None:
            st.markdown("## Related Topic (ID and TopicID):")
            st.write(matched_topic)
        else:
            st.markdown("## No Matching Topic Found")
          