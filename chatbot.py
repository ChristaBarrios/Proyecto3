import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


class ChatModel():
    def __init__(self):

        vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),
                             persist_directory="data/chroma_db")
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
        self.chat_model = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    def answer(self, input):
        return self.chat_model.invoke(input)
