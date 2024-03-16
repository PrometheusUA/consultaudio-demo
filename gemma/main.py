import os
import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough

from langchain.schema.output_parser import StrOutputParser
import warnings
warnings.filterwarnings('ignore')

from gemma.loaders import load_data_from_file
from gemma.prompts import template



class Main:
    def __init__(self, file_paths, use_gemma=False):
        data = load_data_from_file(file_paths=file_paths)

        text_splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n", "\n", " ", ""],
            chunk_size=1600,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(data)

        headers = {"x-api-key": os.environ['OPENAI_API_KEY']}

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            deployment="text-embedding-3-small",
            headers=headers
            )

        vectordb = Chroma.from_documents(docs, embeddings)
        self.retriever = vectordb.as_retriever()

        self.prompt = ChatPromptTemplate.from_template(template)
        if use_gemma:
            repo_id = "google/gemma-2b-it"

            self.chat = HuggingFaceEndpoint(
                repo_id=repo_id, temperature=1.5, # max_length=50,
            ) 
        else:
            self.chat = ChatOpenAI()

    @staticmethod
    def remove_abbreviations(text):
        pattern = r"\([A-Za-z]+\)"
        
        cleaned_text = re.sub(pattern, "", text)
        return cleaned_text 

    def answer(self, question):
        qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.chat
            | StrOutputParser()
        )

        return qa_chain.invoke(question)


if __name__ == "__main__":
    file_paths = [
        './gemma/contex.txt'
    ]

    main = Main(file_paths)

    question = "Could you tell me the content of lecture?"

    print(main.answer(question))
