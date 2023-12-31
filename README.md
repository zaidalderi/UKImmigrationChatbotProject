# UK Immigration Chatbot Project

Welcome to the repository for the UK Immigration Chatbot, a Masters Project dedicated to providing information and guidance on UK Immigration rules.

## Prerequisites

Before running the chatbot, ensure the following resources are available:

- **GloVe Embeddings**: Download the `glove.42B.300d.zip` file which contains the necessary embeddings. It can be [downloaded here](https://nlp.stanford.edu/data/glove.42B.300d.zip). Make sure to extract the contents to the appropriate directory so that the chatbot can access it.

## Repository Contents

This repository contains all the essential files required to run the chatbot:

- **chatbotLandingPage.html**: This is the landing page of the chatbot, serving as the primary entry point for users.
  
- **intentClassificationBestModel.h5**: A trained deep learning model responsible for classifying user inputs to understand their intent.
  
- **intentsFile.json**: Acts as the chatbot's knowledge base. It holds both the training data for the model and the responses the bot can provide based on recognized intents.
  
- **server.py**: This file contains the back-end code, bridging the front-end webpage with the trained model to ensure seamless integration and user experience.
  
- **webScrapingScript.py**: A script dedicated to scraping the UK Government website. It fetches immigration rules and structures them into a comprehensive JSON file for the chatbot to utilize.

- **ImmigrationChatbotIntentClassification.ipynb**: A Jupyter Notebook that details the entire process of data cleaning, preprocessing of the `intentsFile.json`, and training a Bi-LSTM neural network for intent classification.

## Getting Started

To begin using the UK Immigration Chatbot:

1. Clone this repository to your local machine.
2. Ensure all dependencies are installed.
3. Run the `server.py` script to start the backend server.
4. Open the `chatbotLandingPage.html` file in a web browser to access and interact with the chatbot.

Thank you for visiting the UK Immigration Chatbot project repository!
