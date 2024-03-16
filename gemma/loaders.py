import os
import re
import PyPDF2
import json

from langchain_community.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

from gemma.utility import scrape_links_for_text



class Loader(BaseLoader):
    def __init__(self, file_path, encoding="utf-8"):
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding

    def find_urls(self, text):
        url_pattern = r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        return re.findall(url_pattern, text)
    

class TXTLoader(Loader):
    def load(self):
        with open(self.file_path, 'r', encoding=self.encoding, errors="ignore") as file:
            text = file.read()

        text = text.strip()
        texts = [text]

        urls = self.find_urls(text)
        scraped_text = scrape_links_for_text(urls)
        for t in scraped_text:
            texts.append(t)

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata) for text in texts]
    

class PDFLoader(Loader):
    def load(self):
        reader = PyPDF2.PdfReader(self.file_path)

        text = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text()

        text = text.strip()
        texts = [text]

        urls = self.find_urls(text)
        scraped_text = scrape_links_for_text(urls)
        for t in scraped_text:
            texts.append(t)

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata) for text in texts]
    

class JSONLoader(Loader):
    def load(self):
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            data = json.load(file)

        text = ""
        segments = data.get("segments", [])
        for segment in segments:
            text_segment = segment.get("text", "").strip()
            if text_segment:
                text += text_segment + "\n"

        text = text.strip()
        texts = [text]

        urls = self.find_urls(text)
        scraped_text = scrape_links_for_text(urls)
        for t in scraped_text:
            texts.append(t)

        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata) for text in texts]
    

def load_data_from_file(file_paths, encoding="utf-8"):
    docs = []

    for file_path in file_paths:
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() in ['.txt', '.pdf', '.json']:
            if file_extension.lower() == '.txt':
                loader = TXTLoader(file_path)
            elif file_extension.lower() == '.json':
                loader = JSONLoader(file_path)
            elif file_extension.lower() == '.pdf':
                loader = PDFLoader(file_path)

            loaded_docs = loader.load()
            for doc in loaded_docs:
                docs.append(doc)
        else:
            print(f"Unsupported file type: {file_extension}")

    return docs
