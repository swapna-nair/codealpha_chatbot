import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def load_faqs(filepath="faqs.json"):
    """Loads FAQs from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Tokenizes, lowercases, removes punctuation, and lemmatizes the text."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

def get_best_match(user_query, faqs):
    """Finds the most similar FAQ using TF-IDF and cosine similarity."""
    # Preprocess the user query
    processed_query = preprocess(user_query)
    
    # Preprocess all FAQ questions
    faq_questions = [preprocess(faq["question"]) for faq in faqs]
    
    # Combine query with FAQ questions for vectorization
    corpus = [processed_query] + faq_questions
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarity between query (index 0) and all FAQs (index 1 to end)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Find the index of the highest similarity score
    best_match_index = np.argmax(cosine_similarities)
    best_match_score = cosine_similarities[best_match_index]
    
    # Threshold for matching
    if best_match_score > 0.2:
        return faqs[best_match_index]["answer"]
    else:
        return "I'm sorry, I don't have an answer for that. Please try rephrasing your question."

def chat():
    print("Welcome to the FAQ Chatbot! Type 'exit' or 'quit' to stop.")
    faqs = load_faqs()
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\nChatbot: Goodbye!")
            break
        
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
        
        if not user_input.strip():
            continue
            
        response = get_best_match(user_input, faqs)
        print(f"Chatbot: {response}")

if __name__ == '__main__':
    chat()
