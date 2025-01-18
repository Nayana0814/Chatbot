IMPLEMENTING CHATBOT USING NLP

This project implements an intent-based chatbot using Natural Language Processing (NLP) techniques and Logistic Regression. It can understand user inputs, classify intents and provide appropriate responses via a Streamlit interface.

Features:
- Identifies user intents using Logistic Regression.
- Interactive Streamlit-based user interface.
- Supports dynamic interaction logging.
- Gives text and audio responses,

Technologies used:
-Python, CSS
- Libraries: NLTK, scikit-learn, pandas, Streamlit, pillow, gTTS

Installation and Setup:
- please run the following in the command prompt for installing the required libraries:
	pip install -r requirements.txt
- run the Streamlit app by using the following command in the local command prompt:
	streamlit run chatbot.py

Dataset:
The dataset used is a JSON file containing intents, patterns and responses. It is located in the 'intents.json' file.

Usage:
- Run the Streamlit interface using the command mentioned above.
- Enter your query in the input field.
- View the chatbot's response in real-time.

Folder Structure:
- chatbot.py: Main Python script for the chatbot.
- intents.json: Dataset.
- requirements.txt: List of required python libraries.
- README.md: Project documentation.
- Chatbot.json: Image\Logo for the interface.

Contributor:
- Nayana Chandran Puravankara (Project Developer)
