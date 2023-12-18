import requests
from bs4 import BeautifulSoup

def get_page_content(url):
    page = requests.get(url)
    if page.status_code == 200:
        return page.content
    else:
        print(f"Failed to retrieve page content. Status code: {page.status_code}")
        return None

def get_text_score(text):
    # You can implement a scoring mechanism based on relevance to a topic
    # For this example, a dummy score is generated
    # You can replace this with your relevance scoring logic
    # Here, the score is simply based on the length of the text
    return min(len(text) / 100, 10.0)  # Limit score to be between 0 and 10

def scrape_website(url):
    src = get_page_content(url)

    if src:
        soup = BeautifulSoup(src, "html.parser")
        text_elements = soup.find_all("p")  # Adjust this to target specific elements containing text

        data = []
        for idx, element in enumerate(text_elements, start=1):
            text = element.get_text().strip()
            score = get_text_score(text)
            data.append({"Text": text, "Score": score})
            print(f"Text {idx} (Arabic Language):")
            print(text)
            print(f"Score: {score:.2f}")
            print()

        return data

if __name__ == "__main__":
    # Add the URLs of Arabic websites containing text related to your topic
    urls = [
        "https://www.islamweb.net/ar/article/207535/%D8%A3%D8%B9%D8%AF%D8%A7%D8%A1-%D8%A7%D9%84%D8%A5%D9%8A%D9%85%D8%A7%D9%86-%D8%A7%D9%84%D8%AB%D9%84%D8%A7%D8%AB%D8%A9",  
        "https://www.islamweb.net/ar/article/21250/%D8%AD%D9%82%D9%8A%D9%82%D8%A9-%D8%A7%D9%84%D8%A5%D9%8A%D9%85%D8%A7%D9%86-%D8%B9%D9%86%D8%AF-%D8%A3%D9%87%D9%84-%D8%A7%D9%84%D8%B3%D9%86%D8%A9", 
        # Add more URLs as needed
    ]

    dataset = []
    for url in urls:
        scraped_data = scrape_website(url)
        dataset.extend(scraped_data)

######################

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Stop words removal
    stop_words = set(stopwords.words('arabic'))  # Assuming the text is in Arabic
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming (using Porter Stemmer)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()  # Lemmatizer for Arabic may differ
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]

    return " ".join(lemmatized_tokens)

# Example dataset (replace this with your collected dataset)
# Here, 'Text' is the column containing the text data and 'Score' is the relevance score
dataset = [
    {"Text": "والمقصود بالشهادتين ليس مجرد النطق بهما ، بل التصديق بمعانيهما والإقرار بهما ظاهرا وباطناً فهذه الشهادة هي التي تنفع صاحبها عند الله عز وجل .", "Score": 1.41},
    {"Text": "فكل هذه الأدلة وغيرها كثير يدل على أن إيمان القلب هو أصل الإيمان ومادته ، ومن ضيع إيمان القلب فلا إيمان له بل هو معدود من الزنادقة والمنافقين .", "Score": 1.43},
    # Add more data as needed
]

# Preprocess the text in the dataset
preprocessed_texts = []
for data in dataset:
    preprocessed_text = preprocess_text(data['Text'])
    preprocessed_texts.append(preprocessed_text)

# Discretization using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_texts)

# Convert the discretized data into a DataFrame with scores
discretized_data = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
discretized_data['Score'] = [data['Score'] for data in dataset]

# Display the preprocessed and discretized dataset
print(discretized_data)


####################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example dataset (replace this with your dataset)
texts = [
    "This is an example sentence.",
    "Another example sentence here.",
    # Add more text samples
]

# Tokenize the text and convert to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Define parameters for the model
num_words = len(tokenizer.word_index) + 1  # Add 1 for padding token (index 0)
embedding_dim = 100  # Example: Dimension of word embeddings
max_seq_length = max(len(seq) for seq in sequences)  # Maximum sequence length

# Pad sequences to ensure uniform length for input
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_seq_length))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example labels (replace with your labels)
labels = [0, 1]  # Binary classification labels

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.2)
