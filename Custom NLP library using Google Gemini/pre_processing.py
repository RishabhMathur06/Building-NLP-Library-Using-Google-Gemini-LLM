import ast
from google.generativeai import embedding

def clean_text(input_text, model):
    # Question to be asked
    question = f''' Given the input sentence, clean this text but without changing the words {input_text}'''

    # Generate the response
    response = model.generate_content(question)

    return response.text.strip()

def lemmatize_text(input_text, model):
    # Question to be asked
    question = f''' Given the input sentence, perform lemmatization on it {input_text}. 
                    Output must be the lemmatized sentence'''

    # Generate response
    response = model.generate_content(question)

    return response.text.strip()

def stem_text(input_text, model):
    # Question to be asked
    question = f''' Given the input sentence, perform stemming on this sentence {input_text}
                    Output must be stemmed sentence.'''
    
    response = model.generate_content(question)

    return response.text.strip()

def tokenize_text(input_text, model):
    # Question to be asked
    question = f''' Given the input sentence, perform tokenization on it {input_text}
                    Output must be the tokenized sentence.'''

    response = model.generate_content(question)

    return response.text.strip()

def extract_patterns(input_text, patterns, model):
    # Question to be asked
    question = f''' Given the input sentence, extract {patterns} on it
                    {input_text}
                    output just contain the extracted pattern and not the pattern name.'''
    
    response = model.generate_content(question)

    # Split the output by newline character
    output_list = response.text.split('\n')

    # Remove any empty string from the list
    output_list = [item for item in output_list if item]

    return output_list

def remove_html_tags(input_text, html_tags, model):
    # Question to be asked
    question = f''' Given the input sentence {input_text}, remove the html tags: {html_tags}'''

    response = model.generate_content(question)

    return response.text.strip()

def replace_text(input_text, replacement_rules, model):
    # Question to be asked
    question = f''' Given the input sentence {input_text}, replace the words: {replacement_rules}'''

    response = model.generate_content(question)

    return response.text.strip()

def generate_embeddings(input_text):

    response = embedding.embed_content(content=input_text, model = "models/embedding-001")

    return response