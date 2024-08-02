from flask import Flask, request, jsonify, render_template
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time
import logging
import traceback
import re

# Set OpenAI API key directly in the code
openai.api_key = 'sk-proj-rxBOlgWDXN8Na8Gg5pITT3BlbkFJ5NegmL83pXCtdvtjXuZO'  

# Load subtopics dataset
subtopics_df = pd.read_csv('subtopics.csv')

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Function to extract video ID from a YouTube URL
def extract_video_id(youtube_url):
    try:
        video_id = None
        # Regular expression to extract video ID from YouTube URL
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

        youtube_match = re.match(youtube_regex, youtube_url)
        if youtube_match:
            video_id = youtube_match.group(6)

        logger.debug(f"Extracted video ID: {video_id}")
        return video_id
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        logger.error(traceback.format_exc())
        return None

# Function to extract transcript details from a YouTube video
def extract_transcript_details(video_id):
    try:
        logger.debug(f"Extracting transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_list])
        logger.debug("Transcript successfully extracted.")
        return transcript
    except Exception as e:
        logger.error(f"Error extracting transcript: {e}")
        logger.error(traceback.format_exc())
        return None

# Function to download audio from YouTube video using yt-dlp
def download_audio(video_id):
    try:
        logger.debug(f"Downloading audio for video ID: {video_id}")
        youtube_video_url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
           'format': 'bestaudio/best',
           'postprocessors': [{
           'key': 'FFmpegExtractAudio',
           'preferredcodec': 'mp3',
           'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
        'quiet': True,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'verbose': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_video_url, download=True)
            audio_file = ydl.prepare_filename(info_dict)
            audio_file = audio_file.replace('.webm', '.mp3').replace('.m4a', '.mp3')

        logger.debug("Audio successfully downloaded.")
        return audio_file
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        logger.error(traceback.format_exc())
        return None

# Function to convert audio to text using OpenAI Whisper API
def audio_to_text(audio_file):
    try:
        logger.debug(f"Converting audio to text for file: {audio_file}")
        audio = AudioSegment.from_mp3(audio_file)
        chunk_size = 5 * 60 * 1000  # 5 minutes
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

        transcript = ""
        for i, chunk in enumerate(chunks):
            chunk_file_name = f"chunk_{i}.mp3"
            chunk.export(chunk_file_name, format="mp3")
            logger.debug(f"Exported chunk {i} to {chunk_file_name}")
            with open(chunk_file_name, "rb") as chunk_file:
                response = openai.Audio.transcribe("whisper-1", chunk_file)
                logger.debug(f"Transcribed chunk {i}: {response['text'].strip()}")
                transcript += response['text'].strip() + " "

        logger.debug("Audio successfully converted to text.")
        return transcript.strip()
    except Exception as e:
        logger.error(f"Error converting audio to text: {e}")
        logger.error(traceback.format_exc())
        return None

# Function to summarize text using OpenAI API
def summarize_text_with_openai(text, max_tokens=2000, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            logger.debug(f"Attempt {attempt + 1} to summarize text using OpenAI API")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": f"Summarize the following YouTube transcript in points, within 250 words:\n\n{text}"}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            summary = response['choices'][0]['message']['content'].strip()
            logger.debug(f"Generated summary: {summary}")
            return summary
        except openai.error.OpenAIError as e:
            attempt += 1
            logger.warning(f"Error generating summary, retrying {attempt}/{retries}...")
            logger.error(traceback.format_exc())
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            return "Error generating summary"
    return "Failed to generate summary after several attempts"

# Function to generate summary for chunks of transcript
def generate_summary_for_chunks(transcript_text, chunk_size=1500):
    transcript_chunks = [transcript_text[i:i + chunk_size] for i in range(0, len(transcript_text), chunk_size)]
    summaries = []

    for i, chunk in enumerate(transcript_chunks):
        logger.debug(f"Generating summary for chunk {i}")
        summary = summarize_text_with_openai(chunk)
        if "Error generating summary" in summary:
            logger.error(f"Failed to generate summary for chunk {i}")
        else:
            logger.debug(f"Summary for chunk {i}: {summary}")
        summaries.append(summary)

    combined_summary = " ".join(summaries)
    logger.debug(f"Generated summaries: {summaries}")
    return combined_summary

# Function to find matching topic with a threshold of 70% using sentence embeddings
def find_matching_topic(transcript_text, threshold=0.4):
    try:
        logger.debug("Finding matching topic.")
        transcript_embedding = sentence_model.encode(transcript_text, convert_to_tensor=True)
        subtopics_embeddings = sentence_model.encode(subtopics_df['name'].tolist(), convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(transcript_embedding, subtopics_embeddings)[0]
        best_match_idx = cosine_scores.argmax().item()
        best_match_score = cosine_scores[best_match_idx].item()

        if best_match_score >= threshold:
            best_match = subtopics_df.iloc[best_match_idx][['id', 'topicId']]
            logger.debug(f"Found matching topic: {best_match['id']} with score: {best_match_score}")
            return {
                'id': int(best_match['id']),
                'topicId': int(best_match['topicId'])
            }
        logger.debug("No matching topic found.")
        return None
    except Exception as e:
        logger.error(f"Error finding matching topic: {e}")
        logger.error(traceback.format_exc())
        return None

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['GET'])
def process_video():
    youtube_link = request.args.get('youtube_link')

    logger.info(f"Received YouTube link: {youtube_link}")

    if not youtube_link:
        logger.error("No YouTube link provided")
        return jsonify({'error': 'No YouTube link provided'}), 400

    try:
        video_id = extract_video_id(youtube_link)
        if not video_id:
            logger.error("Invalid YouTube link provided")
            return jsonify({'error': 'Invalid YouTube link provided'}), 400

        transcript_text = extract_transcript_details(video_id)
        if not transcript_text:
            logger.info("Transcript not available, attempting to download audio")
            audio_file = download_audio(video_id)
            if audio_file is None:
                logger.error("Failed to download audio")
                return jsonify({'error': 'Failed to download audio'}), 500

            transcript_text = audio_to_text(audio_file)
            if transcript_text is None:
                logger.error("Failed to convert audio to text")
                return jsonify({'error': 'Failed to convert audio to text'}), 500

        if transcript_text:
            logger.info("Transcript successfully obtained")
            # Split transcript and generate summary for each chunk
            summary = generate_summary_for_chunks(transcript_text)
            matched_topic = find_matching_topic(transcript_text)

            if matched_topic is None:
                matched_topic = {'id': 'No Matching Topic', 'topicId': 'N/A'}

            return jsonify({
                'summary': summary,
                'id': matched_topic['id'],
                'topicId': matched_topic['topicId']
            })
        else:
            logger.error("Failed to process the video")
            return jsonify({'error': 'Failed to process the video'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)

