import os
import gc

from TTS.api import TTS
from torch.cuda import empty_cache

OUTPUT_PATH = './data/audios'


def perform_tts(filename_base: str, text: str):
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
    tts_model.tts_to_file(text=text,
                file_path=f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}_answer.wav",
                speaker_wav=f"{os.environ.get('SAVE_PATH', OUTPUT_PATH)}/{filename_base}_speaker.wav",
                language="en")
    
    del tts_model
    gc.collect()
    empty_cache()
