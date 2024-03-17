import os
from pytube import YouTube
import moviepy.editor as mp
from urllib.parse import urlparse, parse_qs
import whisperx
import json
import gc
from pyannote.audio import Pipeline
from pyannote.core import Segment
from torch import device

from torch.cuda import empty_cache

OUTPUT_PATH = './data/audios'


def save_transcript(filename_base: str):
    transcript_fpath = f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}_transcript.json"
    if not os.path.exists(transcript_fpath):
        model = whisperx.load_model("tiny", 'cuda', compute_type='float16', language='en')
        audio = whisperx.load_audio(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.wav")
        transcription_result = model.transcribe(audio, batch_size=16)

        with open(transcript_fpath, "w") as file:
            file_content = json.dumps(transcription_result, ensure_ascii=False)
            file_content = file_content.replace("{", "{\n")
            file_content = file_content.replace("}", "}\n")
            file_content = file_content.replace("\",", "\",\n")
            file.write(file_content)
        
        del model
        gc.collect()
        empty_cache()


def create_main_speaker_sample(filename_base: str):
    speaker_fpath = f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}_speaker.wav"
    if not os.path.exists(speaker_fpath):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.environ['HUGGINGFACEHUB_API_TOKEN']).to(device('cuda'))
        diarization = pipeline(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.wav")
        selected_segments = [segment for segment, _, speaker 
                                    in diarization.itertracks(yield_label=True) 
                                    if speaker == diarization.argmax()]

        clip = mp.AudioFileClip(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.wav")

        concatenated_audio = None
        clip_chunks = []
        for segment in selected_segments:
            start_time, end_time = segment.start, segment.end
            segment_audio = clip.subclip(start_time, end_time)
            clip_chunks.append(segment_audio)
        
        concatenated_audio = mp.concatenate_audioclips(clip_chunks)
        concatenated_audio.write_audiofile(speaker_fpath)


def load_youtube(video_link, verbose=False):
    youtubeObject = YouTube(video_link)
    youtubeAudio = youtubeObject.streams.get_audio_only('mp4')

    title = youtubeObject.streams[0].title
    desc = youtubeObject.description

    video_link_parsed = urlparse(video_link)
    filename_base = parse_qs(video_link_parsed.query)['v'][0]

    if not os.path.exists(os.environ.get('SAVE_PATH', OUTPUT_PATH)):
        os.makedirs(os.environ.get('SAVE_PATH', OUTPUT_PATH))
    if not os.path.exists(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.mp4"):
        try:
            youtubeAudio.download(os.environ.get('SAVE_PATH', OUTPUT_PATH), f'{filename_base}.mp4')
        except Exception as e:
            if verbose:
                print(f"An error has occurred, {e}")
            return 0
    
    if not os.path.exists(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.wav"):
        clip = mp.AudioFileClip(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.mp4")
        clip.write_audiofile(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}.wav")

    if not os.path.exists(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}_desc.txt"):
        with open(f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}_desc.txt", 'w') as f:
            f.write(f'{title}\n\n{desc}')
    
    save_transcript(filename_base)
    create_main_speaker_sample(filename_base)

    if verbose:
        print("Download is completed successfully")
    return filename_base
