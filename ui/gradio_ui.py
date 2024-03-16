import sys
import gradio as gr

sys.path.insert(0, 'E:\\_UNIVER\\UCU\\gen_ai_school\\hack\\')

from gemma.main import Main
from ui.load_youtube import load_youtube
from ui.perform_tts import perform_tts
from dotenv import load_dotenv


def answer_question(question, video_link, pdfs, model):
    filename_base = load_youtube(video_link)
    if filename_base == 0:
        return 'Some error occured in loading video!!!', ''
    
    file_paths = [
        f'./data/audios/{filename_base}_transcript.json',
        f'./data/audios/{filename_base}_desc.txt',
    ]

    if pdfs is not None:
        file_paths.extend(pdfs)

    main = Main(file_paths, use_gemma=(model=='Gemma-2B'))

    answer_text = Main.remove_abbreviations(str(main.answer(question)))
    perform_tts(filename_base, answer_text)
    answer_audio = f'./data/audios/{filename_base}_answer.wav'
    return answer_text, answer_audio

if __name__ == "__main__":
    load_dotenv()
    
    demo = gr.Interface(
        title='ConsultAudio project demo',
        fn=answer_question,
        inputs=[gr.Text(label='Incoming question'), gr.Text(label='Link to the lecture video'), gr.File(file_count='multiple', file_types=['pdf'], label='Additional files'), gr.Radio(['GPT-3.5', 'Gemma-2B'], value='GPT-3.5', label='Model on backend')],
        outputs=[gr.Text(label='Text answer'), gr.Audio(label='Audio answer')],
    )
    
    demo.launch()
