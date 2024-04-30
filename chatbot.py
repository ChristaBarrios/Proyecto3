import os
import re
import PyPDF2
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


class ChatModel():
    def __init__(self, pdf_file: str):
        # read text and create rags
        self.pdf_file = pdf_file
        self.read_pdf()
        text = re.sub(r'\n', ' ', self.text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
        splits = text_splitter.split_text(text)
        rags = [Document(page_content=txt) for txt in splits]

        vectorstore = Chroma.from_documents(documents=rags, embedding=OpenAIEmbeddings(),
                                            persist_directory="data/chroma_db")
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
        self.chat_model = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    def read_pdf(self):
        self.text = ''
        with open(self.pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            for n in range(num_pages):
                page = reader.pages[n]
                self.text += ' ' + page.extract_text()

    def answer(self, input):
        return self.chat_model.invoke(input)
