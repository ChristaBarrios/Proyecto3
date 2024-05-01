import re
import PyPDF2
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def read_pdf(pdf_file):
    text = ''
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        for n in range(num_pages):
            page = reader.pages[n]
            text += ' ' + page.extract_text()
    return text


pdf_file = 'shrek-script-pdf.pdf'
text = read_pdf(pdf_file)

text = re.sub(r'\n', ' ', text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
splits = text_splitter.split_text(text)
rags = [Document(page_content=txt) for txt in splits]

vectorstore = Chroma.from_documents(documents=rags, embedding=OpenAIEmbeddings(),
                                            persist_directory="data/chroma_db")