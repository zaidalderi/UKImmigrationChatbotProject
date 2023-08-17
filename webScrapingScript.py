# Importing required libraries and modules
import requests
from bs4 import BeautifulSoup
import json
import spacy
from spacy.lang.en import English
import openai

# Initialize OpenAI API key
openai.api_key = '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'

# Initializing the English language model from spaCy
nlp = English()

# Function to generate tags for intents
def generate_tag(patterns):
    tag_words = set()

    # Tokenize the input string using spaCy
    doc = nlp(patterns.lower())
    for token in doc:
        # Filter out stop words, punctuations, and digits
        if not token.is_stop and not token.is_punct and not token.is_digit:
            tag_words.add(token.text)

    # Create a tag by joining the words with underscores
    tag = '_'.join(sorted(list(tag_words)))

    return tag

# Function to generate question patterns based on responses
def generate_patterns(intent):
    responses = intent['responses']
    concatenated_responses = ' '.join(responses)
    generated_patterns = []
    prompt = f"Generate questions from the following response:{concatenated_responses}"

    # Make a call to OpenAI's Completion API with a prompt to generate questions
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        n=3,
        stop=None,
        temperature=0.7
    )
        
    for choice in res.choices:
        generated_patterns.append(choice.text.strip())

    return generated_patterns

# List of URLs to scrape data from
url = [...]

# List to store scraped data
all_data = []

# Loop through each URL and scrape the relevant data
for link in url:
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')

    content = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li'])

    data = {}
    current_header = ""
    current_text = []
    ignore_headers = [...]

    for tag in content:
        if tag.name in ['p', 'li']:
            if current_header not in ignore_headers:
                current_text.append(tag.get_text(separator = ' ', strip=True))
        elif tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if current_header:
                if current_header not in ignore_headers and current_text:
                    current_text = list(dict.fromkeys(current_text[::-1]))[::-1]
                    data[current_header] = current_text
            current_header = tag.get_text(strip=True)
            current_text = []

    if current_header:
        if current_header not in ignore_headers and current_text:
            current_text = list(dict.fromkeys(current_text[::-1]))[::-1]
            data[current_header] = current_text

    all_data.append(data)

# Transforming the scraped data to generate intents
new_all_data = []
for data in all_data:
    for key, value in data.items():
        intent = {"patterns": [key], "responses": value, "tag": generate_tag(key)}
        new_all_data.append(intent)

final_data = {"intents": new_all_data}

for intent in final_data["intents"]:
    concatenated_responses = ' '.join(intent["responses"])
    intent["responses"] = [concatenated_responses]

# Write the final data to a JSON file
with open('intentsFile.json', 'w', encoding='utf-8') as outfile:
    json.dump(final_data, outfile, indent=4, ensure_ascii=False)
