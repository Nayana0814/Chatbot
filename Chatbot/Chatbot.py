import os
import json
import datetime
import nltk
import ssl
import csv
import streamlit as st
from PIL import Image
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from nltk.tokenize import word_tokenize
from gtts import gTTS
import base64

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
nltk.download('punkt_tab')

file_path = os.path.abspath("./intents.json")
with open(file_path,"r") as file:
    intents = json.load(file)
    
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
responses = {}
for intent in intents:
  for pattern in intent['patterns']:
    patterns.append(pattern)
    tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

df = pd.DataFrame({'Pattern': patterns, 'Tag': tags})

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)
df['Processed Text'] = df['Pattern'].apply(preprocess_text)
#vectorizer and classifier
vectorizer = TfidfVectorizer()
clf=LogisticRegression(random_state=0, max_iter=10000)
#training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x,y)

def add_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #002266;  
        }
        </style>
        """,
        unsafe_allow_html=True
    )
add_background()

def chatbot(input_text):
    if input_text.lower() == "how are you":
        return "I'm fine, Thank you"
    if input_text.lower() == "what's up":
        return "Nothing much"
    if input_text.lower() == "bye":
        return "Thank you for chatting with me, have a great day!"
    if input_text.lower() == "goodbye":
        return "Thank you for chatting with me, have a great day!"
    input_text = preprocess_text(input_text)
    input_text = vectorizer.transform([input_text])
    tag=clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def text_to_audio(text, file_name="response.mp3"):
    tts = gTTS(text)
    tts.save(file_name)
    return file_name

def audio_download_link(file_path):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mpeg;base64,{b64}" download="response.mp3">Download the Audio Response</a>'
    return href

counter = 0
def main():
    global counter
    #.................
    image = Image.open('chatbot.jpg') 
    col1, col2, col3 = st.columns([0.6, 0.2, 0.6])
    with col1:
        st.empty()
    with col2:
        st.image(image, use_container_width=True)
    with col3:
        st.empty()
    #.................
    st.title("Intents of Chatbot using NLP")
    #sidebar menu with options
    menu = ["Home","Conversation History","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    #Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot, please type a message and press Enter to start conversation!")

        #Check if the chat_log.csv file exists, and if not, create with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv','w',newline='',encoding='utf-8') as csvfile:
                csv_writer=csv.writer(csvfile)
                csv_writer.writerow(['User Input','Chatbot Response','Timestamp'])
        counter +=1
        user_input = st.text_input("You: ",key=f"user_input_{counter}")
        if user_input:
            #convert the user input to a string
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot: ",value=response, height=120,max_chars=None, key=f"chatbot_{user_input_str}")

            audio_file = text_to_audio(response)
            st.audio(audio_file, format="audio/mp3")

            st.markdown(audio_download_link(audio_file), unsafe_allow_html=True)

            #Get the current timestamp
            timestamp=datetime.datetime.now().strftime(f"%Y-%m-%d, %H:%M:%S")

            #Save user input and chatbot response to chat_log.csv file
            with open('chat_log.csv','a',newline='',encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str,response,timestamp])
            if response.lower() in ['goodbye','bye']:
                st.write("Thank you for chatting with me, have a great day!")
                st.stop()

    #Conversation History Menu
    elif choice == "Conversation History":
        #Display the conversation history in a collapsible expander
        st.header("Conversation History")
        #with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv','r',encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader) #SKip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    elif choice=="About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user queries. It uses Natural Language Processing (NLP) techniques and a Logistic Regression Model to classify user intents and provide meaningful responses while creating a user – friendly platform for interaction.")
        st.subheader("Project Overview")
        st.write("""
        The project is in two parts:
        1. NLP techniques ( tokenization, text preprocessing and TF-TDF vectorization) and Logistic Regression algorithm is used to train the chatbot.
        2. For building the Chatbot interface, Streamlit web framework is used.
        """)
        st.subheader("Dataset: ")
        st.write("""
        THe dataset used in this project is a collection of labelled intents and entities. The Dataset is the foundation for training and testing the machine learning model. The JSON dataset contains labeled intents, patterns (sample user inputs) and the responses which will be user by the model for training and testing the chatbot:
        -Intents: The intent of the user input(Eg."greeting","budget","About")
        -Entities:The entities extracted from user input(e.g."Hi","How do I create a budget?")
        -Text: The user input text.
        """)
        st.subheader("Streamlit Chatbot interface: ")
        st.write("The chatbot interface is built using Streamlit.")
        st.subheader("Conclusion: ")
        st.write("In tis project, a chatbot was built that can userstand user queries and give responses. It provides an overall understanding of the project along with problem statement, the objectives and the requirements while highlighting the technical and functionalities of the chatbot development. It demonstrates the use of simple yet effective techniques like Logistic Regression along with vectorization methods to create an efficient chatbot.")
if __name__=='__main__':
    main()
