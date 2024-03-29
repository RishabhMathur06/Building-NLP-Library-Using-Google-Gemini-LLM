import ast

def analyze_sentiment(input_text, category, explanation, model):

    category = category if category else 'positive, negative, neutral'

    explanation_text = 'short Explanation: ' if explanation else 'no explanation'

    # Question to be asked.
    question = f'''Given the input text, perform sentiment analysis on it
                    category : {category}
                    {input_text}
                    {explanation_text}
                '''
    
    response = model.generate_content(question)

    return response.text.strip()

def classify_topic(input_text, topics, explanation, model):

    topics = f''' topics are: {topics}''' if topics else 'topics are: story, comedy, horror, romance, news, entertainment'

    explanation_text = 'short explanation: ' if explanation else 'no explanation'

    question = f''' Given the input text, perform topic classification on it
                    topics: {topics}
                    {input_text}
                    {explanation_text}
                '''
    
    response = model.generate_content(question)

    return response.text.strip()

def spam_detection(input_text, category, explanation, model):

    category = category if category else 'spam, not_spam, unknown'

    explanation_text = 'short explanation: ' if explanation else 'no explanation'

    question = f''' Given the input sentence, perform spam detection on it
                    {category}
                    {input_text}
                    {explanation}
                '''
    
    response = model.generate_content(question)

    return response.text.strip()

def detect_ner(input_text, ner_tags, model):
    ner_tags = ner_tags if ner_tags else 'person, location, date, number, organization, time, money, percent, facility, product, event, language, law, ordinal, misc, quantity, cardinal'

    question = f''' Given the input sentence, perform named entity recognition (ner) on it
                {input_text}
                NER tags are: {ner_tags}
                '''
    
    response = model.generate_content(question)

    return response.text.strip()

def detect_pos(input_text, pos_tags, model):
    # check if POS tags are specified, otherwise default to a placeholder
    pos_tags = pos_tags if pos_tags else 'noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity, noun_phrase, verb_phrase, adjective_phrase, adverb_phrase, pronoun_phrase, preposition_phrase, conjunction_phrase, interjection_phrase, determiner_phrase, cardinal_phrase, foreign_phrase, number_phrase, date_phrase, time_phrase, ordinal_phrase, money_phrase, percent_phrase, symbol_phrase, punctuation_phrase, emoticon_phrase, hashtag_phrase, email_phrase, url_phrase, mention_phrase, phone_phrase, ip_phrase, cashtag_phrase, entity_phrase'

    # question to be asked
    question = f'''Given the input text, perform POS detection on it
    POS Tags are: {pos_tags}
    {input_text}
    answer must be in the format
    word: POS
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for machine translation
def translate_text(input_text, source_language, target_language, model):
    # question to be asked
    question = f'''Translate the below input text from {source_language} to {target_language}:
    {input_text}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()

def summarize_text(input_text, summary_length, model):
    # question to be asked
    question = f'''Summarize the below input text of {summary_length} length:
    {input_text}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for question answering
def answer_question(question_text, model):
    # question to be asked
    question = f'''Answer the following question:
    {question_text}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for text generation
def generate_text(prompt, generation_length, model):
    # question to be asked
    question = f'''Generate a {generation_length} text,
    {prompt}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()

# function for Semantic Role Labeling (SRL)
def perform_srl(input_text, model):
    # question to be asked
    question = f'''Given the input text, perform Semantic Role Labeling detection on it
    {input_text}
    Output must be
    Predicate:
    Roles:
    -
    -
    -
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for intent recognition
def recognize_intent(input_text, model):
    # question to be asked
    question = f'''Given the input text, perform Intent Recognition detection on it
    {input_text}
    Output must be
    Intent:
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for paraphrase detection
def paraphrasing_detection(input_text, explanation, model):

    # Check if explanation is required
    explanation_text = 'short explanation: ' if explanation else 'no explanation'

    # Question to be asked for determining paraphrasing
    question = f'''Given the input text, determine if two sentences are paraphrases of each other.
    Sentence 1: {input_text[0]}
    Sentence 2: {input_text[1]}
    Answer must be 'yes' or 'no'.
    {explanation_text}
    '''

    # Generate response
    response = model.generate_content(question)
    return response.text.strip()