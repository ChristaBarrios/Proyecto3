from chatbot import ChatModel
import gradio as gr

chat_model = ChatModel()

def chat_bot(input_question, history):
    response = chat_model.answer(input_question)
    return response['answer']


iface = gr.ChatInterface(fn=chat_bot,
                         title="Shrek-Bot",
                         description="Hey there! I'm Shrek-Bot, here to guide you through the \
                         wondrous world of one of the greatest movies ever made: Shrek 1! Go ahead, \
                         put me to the test - ask me anything!")
iface.launch()
