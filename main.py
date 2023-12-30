import customtkinter, csv, math, random, os, pickle, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
from CTkMessagebox import CTkMessagebox
from tkinter import ttk
from PIL import Image, ImageTk
from tqdm import tqdm

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("CyberScan - Content Detector")
root.geometry("1370x710")
root.resizable(False, False)

# ---------- DATA PREPROCESSING --------- #
# ---------- Tokenization & Vocabulary Building ---------#
def tokenize(text):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    words = []
    word = ""
    for char in text:
        char = char.lower()
        if char.isalpha() or char.isdigit():
            word += char
        elif char in punctuation or char.isspace():
            if word:
                words.append(word)
            if char != ' ':
                words.append(char)
            word = ""
    if word:
        words.append(word)
    return words

# ---------- Negation Handling ---------#
def handle_negation(text):
    words = text.split()
    negation = False 

    for i, word in enumerate(words):
        if word.lower() in ["not", "no", "never"]:
            negation = not negation
        elif negation:
            words[i] = "NOT_" + word

    modified_text = ' '.join(words)
    return modified_text

# ---------- Cleaning Data ---------#
def clean_text(text):
    start_idx = 0
    while True:
        start_idx = text.find('http', start_idx)
        if start_idx == -1:
            break
        end_idx = text.find(' ', start_idx)
        if end_idx == -1:
            end_idx = len(text)
        text = text[:start_idx] + ' ' + text[end_idx:]

    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        text = text.replace(char, ' ')

    text = text.replace('\n', ' ')

    words = text.split()
    words = [word for word in words if not any(char.isdigit() for char in word)]
    text = ' '.join(words)

    text = text.lower()

    stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers','herself', 'it', 'one', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which','who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been','being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if','or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between','into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out','on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why','how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor','only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',"couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

    words = text.split()
    words = [word for word in words if word not in stopwords]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    cleaned_text = ' '.join(words)

    cleaned_text = handle_negation(cleaned_text)

    tokens = tokenize(cleaned_text)

    return cleaned_text

# ---------- Stemming Cleaned Data ---------# 
def porter_stemming(word):
    def has_vowel(word):
        return any(char in 'aeiou' for char in word)

    def has_double_consonant(word):
        for i in range(len(word) - 1):
            if word[i] == word[i + 1] and word[i] not in 'lsz':
                return True
        return False

    def ends_with_cvc(word):
        if len(word) >= 3:
            c, v, cvc = word[-3], word[-2], word[-1]
            if c not in 'aeiou' and v in 'aeiou' and cvc not in 'aeiouwxY':
                return True
        return False

    # Step 1a
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif word.endswith('ss'):
        word = word
    elif word.endswith('s'):
        word = word[:-1]

    # Step 1b
    if word.endswith('eed'):
        if word.count('eed') > 1:
            word = word[:-1]
        else:
            if len(word) > 4:
                word = word[:-1]
    elif word.endswith('ed') and has_vowel(word[:-2]):
        word = word[:-2]
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            word += 'e'
        elif has_double_consonant(word) and not word.endswith('l') and not word.endswith('s') and not word.endswith('z'):
            word = word[:-1]
        elif ends_with_cvc(word):
            word += 'e'

    # Step 1c
    if word.endswith('y') and has_vowel(word[:-1]):
        word = word[:-1] + 'i'

    # Step 2
    if word.endswith('ational'):
        if len(word[:-7]) > 0 and has_vowel(word[:-7]):
            word = word[:-7] + 'ate'
    elif word.endswith('tional'):
        if len(word[:-6]) > 0 and has_vowel(word[:-6]):
            word = word[:-6] + 'tion'
    elif word.endswith('enci'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'ence'
    elif word.endswith('anci'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'ance'
    elif word.endswith('izer'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'ize'
    elif word.endswith('abli'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'able'
    elif word.endswith('alli'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'al'
    elif word.endswith('entli'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'ent'
    elif word.endswith('eli'):
        if len(word[:-3]) > 0 and has_vowel(word[:-3]):
            word = word[:-3] + 'e'
    elif word.endswith('ousli'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'ous'
    elif word.endswith('ization'):
        if len(word[:-7]) > 0 and has_vowel(word[:-7]):
            word = word[:-7] + 'ize'
    elif word.endswith('ation'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'ate'
    elif word.endswith('ator'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'ate'
    elif word.endswith('alism'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'al'
    elif word.endswith('iveness'):
        if len(word[:-7]) > 0 and has_vowel(word[:-7]):
            word = word[:-7] + 'ive'
    elif word.endswith('fulness'):
        if len(word[:-7]) > 0 and has_vowel(word[:-7]):
            word = word[:-7] + 'ful'
    elif word.endswith('ousness'):
        if len(word[:-7]) > 0 and has_vowel(word[:-7]):
            word = word[:-7] + 'ous'
    elif word.endswith('aliti'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'al'
    elif word.endswith('iviti'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'ive'
    elif word.endswith('biliti'):
        if len(word[:-6]) > 0 and has_vowel(word[:-6]):
            word = word[:-6] + 'ble'

    # Step 3
    if word.endswith('icate'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'ic'
    elif word.endswith('ative'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5]
    elif word.endswith('alize'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'al'
    elif word.endswith('iciti'):
        if len(word[:-5]) > 0 and has_vowel(word[:-5]):
            word = word[:-5] + 'ic'
    elif word.endswith('ical'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4] + 'ic'
    elif word.endswith('ful'):
        if len(word[:-3]) > 0 and has_vowel(word[:-3]):
            word = word[:-3]
    elif word.endswith('ness'):
        if len(word[:-4]) > 0 and has_vowel(word[:-4]):
            word = word[:-4]

    # Step 4
    if word.endswith('al'):
        if len(word[:-2]) > 1:
            word = word[:-2]
    elif word.endswith('ance'):
        if len(word[:-4]) > 1:
            word = word[:-4]
    elif word.endswith('ence'):
        if len(word[:-4]) > 1:
            word = word[:-4]
    elif word.endswith('er'):
        if len(word[:-2]) > 1:
            word = word[:-2]
    elif word.endswith('ic'):
        if len(word[:-2]) > 1:
            word = word[:-2]
    elif word.endswith('able'):
        if len(word[:-4]) > 1:
            word = word[:-4]
    elif word.endswith('ible'):
        if len(word[:-4]) > 1:
            word = word[:-4]
    elif word.endswith('ant'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ement'):
        if len(word[:-5]) > 1:
            word = word[:-5]
    elif word.endswith('ment'):
        if len(word[:-4]) > 1:
            word = word[:-4]
    elif word.endswith('ent'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ion'):
        if len(word[:-3]) > 1 and (word[-4] == 's' or word[-4] == 't'):
            word = word[:-3]
    elif word.endswith('ou'):
        if len(word[:-2]) > 1:
            word = word[:-2]
    elif word.endswith('ism'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ate'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('iti'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ous'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ive'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ize'):
        if len(word[:-3]) > 1:
            word = word[:-3]
    elif word.endswith('ing'):
        if len(word[:-3]) > 1:
            word = word[:-3]

    # Step 5a
    if word.endswith('e'):
        if len(word) > 2:
            word = word[:-1]
        elif len(word) == 2 and not ends_with_cvc(word):
            word = word[:-1]

    # Step 5b
    if has_double_consonant(word):
        if word[-1] == word[-2]:
            word = word[:-1]

    return word

# ---------- Load | Compute Preprocessing Cache ---------#
def load_preprocessing():
    if os.path.exists('files/data_preprocessing.pickle'):        
        with open('files/data_preprocessing.pickle', 'rb') as f:
            cached_data = pickle.load(f)
        cleaned_data_t = cached_data['cleaned_data_t']
        stemmed_data_w = cached_data['stemmed_data_w']
        class_mapping_t = cached_data.get('class_mapping_t', {})
        return cleaned_data_t, stemmed_data_w, class_mapping_t
    else:
        # ---------- Data Collection---------# 
        class_mapping = {
            "0": "No Offensive Speech Detected",
            "1": "Offensive Speech Detected"
        }
        def read_data(filename):    
            dataset= []
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                next(reader)
                for row in reader:
                    text=row['tweet']
                    label=row['Toxicity']
                    mapped_label = class_mapping[label]
                    dataset.append((text, mapped_label))
            return dataset
        def read_data1(filename):
            dataset = []
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                next(reader)
                for row in reader:
                    text = row['tweet']
                    label = row['label']
                    mapped_label = class_mapping[label]
                    dataset.append((text, mapped_label))
            return dataset
        def read_data2(filename):
            dataset = []
            with open(filename, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    label = row.get('oh_label', '')
                    text = row.get('Text', '')
                    if label and text:
                        mapped_label = class_mapping.get(label, 'Unknown')
                        dataset.append((text, mapped_label))
            return dataset

        dataset1 = read_data('dataset/FinalBalancedDataset.csv')
        dataset2 = read_data1('dataset/train.csv')
        dataset3 = read_data2('dataset/twitter_sexism_parsed_dataset.csv')

        dataset = dataset1 + dataset2 + dataset3

        cleaned_data = [(clean_text(text), label) for text, label in tqdm(dataset, desc="Cleaning Data")]

        #stemmed_data = [(list(map(porter_stemming, tokens)), label) for tokens, label in tqdm(cleaned_data, desc="Stemming Data")]

        stemmer = PorterStemmer()
        stemmed_data = [(stemmer.stem(text), label) for text, label in tqdm(cleaned_data, desc="Stemming Data")]

        data_to_save = {'cleaned_data_t': cleaned_data, 'stemmed_data_w': stemmed_data, 'class_mapping_t': class_mapping}
        with open('files/data_preprocessing.pickle', 'wb') as f:
            pickle.dump(data_to_save, f)

        return cleaned_data, stemmed_data, class_mapping

# ---------- Data Splitting ---------#
def split_data(data):
    split_index = int(len(data) * 0.7)

    random.shuffle(data)

    with tqdm(total=len(data), desc="Splitting Data") as pbar:
        train_data = data[:split_index]
        test_data = data[split_index:]
        pbar.update(split_index)

    return train_data, test_data

# ---------- Load | DATA SPLITTED ---------#
def load_split_data():
    if os.path.exists('files/data_splitted.pickle'):
        with open('files/data_splitted.pickle', 'rb') as f:
            cached_data = pickle.load(f)
        train_data = cached_data['train_data']
        test_data = cached_data['test_data']
        
        return train_data, test_data
    else:
        train_data, test_data = split_data(stemmed_data_w)

        data_to_save = {'train_data': train_data, 'test_data': test_data}
        with open('files/data_splitted.pickle', 'wb') as f:
            pickle.dump(data_to_save, f)

        return train_data, test_data

# ---------- FEATURE EXTRACTION | TF-IDF --------- #
def tfidf_feature_extraction(train_data, test_data):
    tfidf_vectorizer = TfidfVectorizer()

    train_texts = [text for text, _ in train_data]
    test_texts = [text for text, _ in test_data]

    d_train_features = tfidf_vectorizer.fit_transform(train_texts)
    d_test_features = tfidf_vectorizer.transform(test_texts)

    d_train_labels = [label for _, label in train_data]
    d_test_labels = [label for _, label in test_data]

    return d_train_features, d_train_labels, d_test_features, d_test_labels

# ---------- Load | TF-IDF --------- #
def load_featureExt():
    if os.path.exists('files/feature_extraction.pickle'):
        with open('files/feature_extraction.pickle', 'rb') as f:
            cached_data = pickle.load(f)
        d_train_features = cached_data['d_train_features']
        d_train_labels = cached_data['d_train_labels']

        d_test_features = cached_data['d_test_features']
        d_test_labels = cached_data['d_test_labels']
        return d_train_features, d_train_labels, d_test_features, d_test_labels
    else:
        pbar = tqdm(total=len(train_data), desc="TF-IDF Train")

        d_train_features, d_train_labels, d_test_features, d_test_labels = tfidf_feature_extraction(train_data, test_data)

        for _ in d_train_features:
            pbar.update(1)
        pbar.close()

        pbar = tqdm(total=len(test_data), desc="TF-IDF Train")

        for _ in d_test_features:
            pbar.update(1)
        pbar.close()

        data_to_pickle = {'d_train_features': d_train_features, 'd_train_labels': d_train_labels, 'd_test_features': d_test_features, 'd_test_labels': d_test_labels}

        with open('files/feature_extraction.pickle', 'wb') as f:
            pickle.dump(data_to_pickle, f)

        return d_train_features, d_train_labels, d_test_features, d_test_labels

# ---------- NAIVE BAYES | Training --------- #
def train_naive_bayes(d_train_features, d_train_labels):
    nb_classifier = MultinomialNB()

    nb_classifier.fit(d_train_features, d_train_labels)

    return nb_classifier

# ---------- NAIVE BAYES | RETURNING FUNCTIONS ---------#
cleaned_data_t, stemmed_data_w, class_mapping_t = load_preprocessing()

train_data, test_data = load_split_data()

d_train_features, d_train_labels, d_test_features, d_test_labels = tfidf_feature_extraction(train_data, test_data)

naive_bayes_classifier = train_naive_bayes(d_train_features, d_train_labels)

naive_bayes_predictions = naive_bayes_classifier.predict(d_test_features)

total_documents = len(train_data)
num_off = sum(label == "Offensive Speech Detected" for _, label in train_data)
num_not_off = total_documents - num_off

prior_off = num_off / total_documents
prior_not_off = num_not_off / total_documents

def build_vocab(data):
    vocab = []
    for text, _ in data:
        words = tokenize(text)
        vocab.extend(words)
    return list(set(vocab))

vocab = build_vocab(train_data)

def calculate_word_probabilities(data, vocab):
    word_probs_off = {word: 0 for word in vocab}
    word_probs_not_off = {word: 0 for word in vocab}
    
    total_words_off = 0
    total_words_not_off = 0
    
    for text, label in data:
        words = tokenize(text)
        if label == "Offensive Speech Detected":
            total_words_off += len(words)
            for word in words:
                word_probs_off[word] += 1
        else:
            total_words_not_off += len(words)
            for word in words:
                word_probs_not_off[word] += 1
    
    for word in vocab:
        word_probs_off[word] = (word_probs_off[word] + 1) / (total_words_off + len(vocab))
        word_probs_not_off[word] = (word_probs_not_off[word] + 1) / (total_words_not_off + len(vocab))
    
    return word_probs_off, word_probs_not_off

word_probs_off, word_probs_not_off = calculate_word_probabilities(train_data, vocab)

def predict_class(input_text, vocab, word_probs_off, word_probs_not_off, prior_off, prior_not_off):
    words = input_text.split()
    log_likelihood_off = 0
    log_likelihood_not_off = 0
    
    for word in words:
        if word in vocab:
            log_likelihood_off += math.log(word_probs_off[word])
            log_likelihood_not_off += math.log(word_probs_not_off[word])
    
    log_likelihood_off += math.log(prior_off)
    log_likelihood_not_off += math.log(prior_not_off)
    
    if log_likelihood_off > log_likelihood_not_off:
        return "Offensive Speech Detected"
    elif log_likelihood_off < log_likelihood_not_off:
        return "No Offensive Speech Detected"

# ---------- SVM MODEL | CLASS  --------- #
class SVModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.svm_classifier = LinearSVC()

    # ---------- Tokenization & Vocabulary Building ---------#
    def tokenize(self, text):
        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        words = []
        word = ""
        for char in text:
            char = char.lower()
            if char.isalpha() or char.isdigit():
                word += char
            elif char in punctuation or char.isspace():
                if word:
                    words.append(word)
                if char != ' ':
                    words.append(char)
                word = ""
        if word:
            words.append(word)
        return words

    # ---------- Negation Handling ---------#
    def handle_negation(self, text):
        words = text.split()
        negation = False

        for i, word in enumerate(words):
            if word.lower() in ["not", "no", "never"]:
                negation = not negation
            elif negation:
                words[i] = "NOT_" + word

        modified_text = ' '.join(words)
        return modified_text

    # ---------- Cleaning Data ---------#
    def clean_text(self, text):
        start_idx = 0
        while True:
            start_idx = text.find('http', start_idx)
            if start_idx == -1:
                break
            end_idx = text.find(' ', start_idx)
            if end_idx == -1:
                end_idx = len(text)
            text = text[:start_idx] + ' ' + text[end_idx:]

        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for char in punctuation:
            text = text.replace(char, ' ')

        text = text.replace('\n', ' ')

        words = text.split()
        words = [word for word in words if not any(char.isdigit() for char in word)]
        text = ' '.join(words)

        text = text.lower()

        stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers','herself', 'it', 'one', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which','who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been','being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if','or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between','into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out','on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why','how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not','only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',"couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

        words = text.split()
        words = [word for word in words if word not in stopwords]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        cleaned_text = ' '.join(words)

        cleaned_text = self.handle_negation(cleaned_text)

        tokens = self.tokenize(cleaned_text)

        return cleaned_text

    # ---------- LOAD | Preprocessing Cache ---------#
    def load_preprocessing(self):
        if os.path.exists('preload/data_preprocessing.pickle'):        
            with open('preload/data_preprocessing.pickle', 'rb') as f:
                cached_data = pickle.load(f)
            sv_cleaned_data_t = cached_data['sv_cleaned_data_t']
            sv_stemmed_data_w = cached_data['sv_stemmed_data_w']
            sv_class_mapping_t = cached_data.get('sv_class_mapping_t', {})
            return sv_cleaned_data_t, sv_stemmed_data_w, sv_class_mapping_t
        else:
            # ---------- Data Collection---------# 
            sv_class_mapping = {
                "0": "No Offensive Speech Detected",
                "1": "Offensive Speech Detected"
            }
            def read_data(filename):    
                dataset= []
                with open(filename, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    next(reader)
                    for row in reader:
                        text=row['tweet']
                        label=row['Toxicity']
                        mapped_label = sv_class_mapping[label]
                        dataset.append((text, mapped_label))
                return dataset
            def read_data1(filename):
                dataset = []
                with open(filename, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    next(reader)
                    for row in reader:
                        text = row['tweet']
                        label = row['label']
                        mapped_label = sv_class_mapping[label]
                        dataset.append((text, mapped_label))
                return dataset
            def read_data2(filename):
                dataset = []
                with open(filename, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        label = row.get('oh_label', '')
                        text = row.get('Text', '')
                        if label and text:
                            mapped_label = sv_class_mapping.get(label, 'Unknown')
                            dataset.append((text, mapped_label))
                return dataset

            dataset1 = read_data('dataset/FinalBalancedDataset.csv')
            dataset2 = read_data1('dataset/train.csv')
            dataset3 = read_data2('dataset/twitter_sexism_parsed_dataset.csv')

            dataset = dataset1 + dataset2 + dataset3

            sv_cleaned_data = [(self.clean_text(text), label) for text, label in tqdm(dataset, desc="Cleaning Data")]

            sv_stemmer = PorterStemmer()
            sv_stemmed_data = [(sv_stemmer.stem(text), label) for text, label in tqdm(sv_cleaned_data, desc="Stemming Data")]

            data_to_save = {'sv_cleaned_data_t': sv_cleaned_data, 'sv_stemmed_data_w': sv_stemmed_data, 'sv_class_mapping_t': sv_class_mapping}
            with open('preload/data_preprocessing.pickle', 'wb') as f:
                pickle.dump(data_to_save, f)

            return sv_cleaned_data, sv_stemmed_data, sv_class_mapping

    # ---------- FEATURE EXTRACTION --------- #
    def extract_features(self, sv_stemmed_data):
        sv_stemmed_texts = [text for text, label in sv_stemmed_data]
        sv_labels = [label for text, label in sv_stemmed_data]

        sv_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

        sv_tfidf_features = sv_vectorizer.fit_transform(sv_stemmed_texts)
        sv_data = list(zip(sv_tfidf_features, sv_labels))

        feature_names = sv_vectorizer.get_feature_names_out()

        return sv_tfidf_features, sv_labels, sv_vectorizer

    # ---------- LOAD | Feature Extration Cache --------- #
    def load_featureExt(self):
        if os.path.exists('preload/feature_extraction.pickle') and os.path.exists('preload/vectorizer.pickle'):
            with open('preload/feature_extraction.pickle', 'rb') as svm_file:
                cached_data = pickle.load(svm_file)
            sv_tfidf_features = cached_data['sv_tfidf_features']
            sv_labels = cached_data['sv_labels']

            with open('preload/vectorizer.pickle', 'rb') as vecto:
                sv_vectorizer = pickle.load(vecto)

            return sv_tfidf_features, sv_labels, sv_vectorizer
        else:
            pbar = tqdm(total=len(sv_stemmed_data), desc="TF-IDF Features")

            sv_stemmed_texts = [text for text, label in sv_stemmed_data]
            sv_labels = [label for text, label in sv_stemmed_data]
            
            filtered_stemmed_texts = [text for text in sv_stemmed_texts if text.strip() != '']

            filtered_labels = [label for text, label in zip(sv_stemmed_texts, sv_labels) if text.strip() != '']

            sv_tfidf_features, sv_labels, sv_vectorizer = self.extract_features(list(zip(filtered_stemmed_texts, filtered_labels)))

            for _ in sv_tfidf_features:
                pbar.update(1)
            pbar.close()

            # ---------- Save Features --------- #
            data_save_features = {'sv_tfidf_features': sv_tfidf_features, 'sv_labels': sv_labels}
            with open('preload/feature_extraction.pickle', 'wb') as svm_file:
                pickle.dump(data_save_features, svm_file)

            # ---------- Save Vectorizer | ONLY --------- #
            with open('preload/vectorizer.pickle', 'wb') as vecto:
                pickle.dump(sv_vectorizer, vecto)

            return sv_tfidf_features, sv_labels, sv_vectorizer

    # ---------- Splitting | Train | Test --------- #
    def splitting_data(self, sv_stemmed_data, test_size=0.3, random_state=42):
        sv_tfidf_features, sv_labels, sv_vectorizer = self.extract_features(sv_stemmed_data)
        
        sv_set_train, sv_set_test, sv_lab_train, sv_lab_test = train_test_split(sv_tfidf_features, sv_labels, test_size=test_size, random_state=random_state)
        
        return sv_set_train, sv_set_test, sv_lab_train, sv_lab_test
    
    # ---------- LOAD | Splitting --------- #
    def load_splittingData(self, test_size=0.3, random_state=42):
        if os.path.exists('preload/splitted_data.pickle'):
            with open('preload/splitted_data.pickle', 'rb') as f:
                cached_data = pickle.load(f)
            sv_set_train = cached_data['sv_set_train']
            sv_set_test = cached_data['sv_set_test']
            sv_lab_train = cached_data['sv_lab_train']
            sv_lab_test = cached_data['sv_lab_test']
            return sv_set_train, sv_set_test, sv_lab_train, sv_lab_test
        else:
            with tqdm(total=len(sv_stemmed_data), desc="Splitting Data") as pbar:
                sv_set_train, sv_set_test, sv_lab_train, sv_lab_test = self.splitting_data(sv_stemmed_data, test_size, random_state)
                pbar.update(len(sv_stemmed_data))

            data_to_pickle = {'sv_set_train': sv_set_train, 'sv_set_test': sv_set_test, 'sv_lab_train': sv_lab_train, 'sv_lab_test': sv_lab_test}
            with open('preload/splitted_data.pickle', 'wb') as f:
                pickle.dump(data_to_pickle, f)

        return sv_set_train, sv_set_test, sv_lab_train, sv_lab_test

    # ---------- Training SVM --------- #
    def training(self, sv_set_train, sv_lab_train):
        svm_classifier = LinearSVC(C=1.0, dual=False)
        svm_classifier.fit(sv_set_train, sv_lab_train)
        return svm_classifier

    # ---------- LOAD | Trained Model --------- #
    def loadtrained(self):
        if os.path.exists('preload/trained_model.pickle'):
            with open('preload/trained_model.pickle', 'rb') as f:
                cached_data = pickle.load(f)
            svm_classifier = cached_data['svm_classifier']
            return svm_classifier
        else:
            with tqdm(total=sv_set_train.shape[0], desc="Training Model") as pbar:
                svm_classifier = self.training(sv_set_train, sv_lab_train)
                pbar.update(sv_set_train.shape[0])

            data_to_pickle = {'svm_classifier': svm_classifier}
            with open('preload/trained_model.pickle', 'wb') as f:
                pickle.dump(data_to_pickle, f)

        return svm_classifier

    # ---------- Testing Model --------- #
    def testing_model(self, svm_classifier, sv_set_test, sv_lab_test):
        sv_lab_pred = svm_classifier.predict(sv_set_test)

        accuracy = accuracy_score(sv_lab_test, sv_lab_pred)
        precision = precision_score(sv_lab_test, sv_lab_pred, pos_label='Offensive Speech Detected')
        recall = recall_score(sv_lab_test, sv_lab_pred, pos_label='Offensive Speech Detected')
        f1 = f1_score(sv_lab_test, sv_lab_pred, pos_label='Offensive Speech Detected')
        conf_matrix = confusion_matrix(sv_lab_test, sv_lab_pred, labels=['No Offensive Speech Detected', 'Offensive Speech Detected'])

        # "No Offensive Speech Detected"
        precision_no_off = precision_score(sv_lab_test, sv_lab_pred, pos_label='No Offensive Speech Detected')
        recall_no_off = recall_score(sv_lab_test, sv_lab_pred, pos_label='No Offensive Speech Detected')
        f1_no_off = f1_score(sv_lab_test, sv_lab_pred, pos_label='No Offensive Speech Detected')
    
        accuracy_per = accuracy * 100
        precision_per = precision * 100
        recall_per = recall * 100
        f1_per = f1 * 100 

        sv_evaluation_results = f"Accuracy: {accuracy_per:.2f}%\n"
        sv_evaluation_results += f"Precision: {precision_per:.2f}%\n"
        sv_evaluation_results += f"Recall: {recall_per:.2f}%\n"
        sv_evaluation_results += f"F1-Score: {f1_per:.2f}%\n\n"
        sv_evaluation_results += "Classification Matrix (SVM):\n"
        sv_evaluation_results += f"                     precision  recall  F1      support\n"
        sv_evaluation_results += f"\n                       {precision_no_off:.2f}        {recall_no_off:.2f}     {recall_no_off:.2f}      {conf_matrix[0, 0] + conf_matrix[0, 1]}\n"
        sv_evaluation_results += f"\n                       {precision:.2f}        {recall:.2f}     {f1:.2f}      {conf_matrix[1, 0] + conf_matrix[1, 1]}\n"
        sv_evaluation_results += f"\nMacroAvg   {(precision + (1 - precision)) / 2:.2f}        {(recall + (1 - recall)) / 2:.2f}     {(f1 + (1 - f1)) / 2:.2f}      {len(sv_lab_test)}\n"
        sv_evaluation_results += f"\nWeightAvg   {accuracy:.2f}        {accuracy:.2f}     {accuracy:.2f}      {len(sv_lab_test)}"
        sv_evaluation_results += f"\n\nConfusion Matrix:\n"
        sv_evaluation_results += f" [[{conf_matrix[0, 0]}   {conf_matrix[0, 1]}]\n"
        sv_evaluation_results += f"  [{conf_matrix[1, 0]}   {conf_matrix[1, 1]}]]"

        return sv_evaluation_results

    # ---------- LOAD | Tested Model --------- #
    def loadtested(self):
        if os.path.exists('preload/tested_model.pickle'):
            with open('preload/tested_model.pickle', 'rb') as f:
                cached_data = pickle.load(f)
            sv_evaluation_results = cached_data['sv_evaluation_results']
            return sv_evaluation_results
        else:
            with tqdm(total=sv_set_test.shape[0], desc="Testing Model") as pbar:
                pbar.update(sv_set_test.shape[0])
            sv_evaluation_results = self.testing_model(svm_classifier, sv_set_test, sv_lab_test)
            
            data_to_pickle = {'sv_evaluation_results': sv_evaluation_results}
            with open('preload/tested_model.pickle', 'wb') as f:
                pickle.dump(data_to_pickle, f)

        return sv_evaluation_results
    
    # ---------- LOAD | Classifier & Vectorizer files --------- #
    def load_class_vect(self, svm_classifier_path, vectorizer_path):
        with open(svm_classifier_path, 'rb') as svm_file:
            loaded_data = pickle.load(svm_file)
        
        if 'svm_classifier' in loaded_data:
            loaded_classifier = loaded_data['svm_classifier']
        else:
            raise ValueError("Loaded model file does not contain 'svm_classifier'")

        with open(vectorizer_path, 'rb') as vecto:
            loaded_vectorizer = pickle.load(vecto)

        return loaded_classifier, loaded_vectorizer

    def predict_offensiveness(self, user_input, svm_classifier_path, vectorizer_path):
        cleaned_input = self.clean_text(user_input)

        input_stemmer = PorterStemmer()

        stemmed_words = [input_stemmer.stem(word) for word in cleaned_input.split()]

        stemmed_input = ' '.join(stemmed_words)

        loaded_classifier, loaded_vectorizer = self.load_class_vect(svm_classifier_path, vectorizer_path)

        input_features = loaded_vectorizer.transform([stemmed_input])

        prediction = loaded_classifier.predict(input_features)

        if prediction[0] == 'No Offensive Speech Detected':
            return "No Offensive Speech Detected."
        else:
            return "Offensive Speech Detected"

svm_model = SVModel()

# ---------- RETURN SVM FUNCTIONS ---------# 
svm_classifier_path = 'preload/trained_model.pickle'
vectorizer_path = 'preload/vectorizer.pickle'

sv_cleaned_data, sv_stemmed_data, sv_class_mapping = svm_model.load_preprocessing()

sv_tfidf_features, sv_labels, sv_vectorizer = svm_model.load_featureExt()

sv_set_train, sv_set_test, sv_lab_train, sv_lab_test = svm_model.load_splittingData()

svm_classifier = svm_model.loadtrained()

sv_evaluation_results = svm_model.loadtested()

# ---------- COLUMN 1 --------- # 
# ---------- Settings Frame --------- # 
settings_frame = customtkinter.CTkFrame(master=root)
settings_frame = customtkinter.CTkFrame(root, width=200, height=600)
settings_frame.pack(pady=20, padx=(10, 20), fill="both", expand=True, side="left")

# ---------- Settings Description ---------# 
text = "Preferences"
description_label = customtkinter.CTkLabel(settings_frame, text=text, wraplength=190, font=("Liberation Sans", 16, "bold"), anchor="w")
description_label.pack(pady=15, padx=10, fill="x", expand=False)

# ---------- Themes Parameter ---------# 
theme_label = customtkinter.CTkLabel(settings_frame, text= "Appearance Mode", wraplength=190, anchor="w")
theme_label.pack(padx=10, fill="x")
theme_label.place(y=50, x=10)

themeoption_var = customtkinter.StringVar(value="Systems Default")

def theme_callback(choice):
    if choice == "Light Mode":
        customtkinter.set_appearance_mode("Light")
    elif choice == "Dark Mode":
        customtkinter.set_appearance_mode("Dark")
    else:
        customtkinter.set_appearance_mode("System")

combobox = customtkinter.CTkOptionMenu(master=settings_frame, anchor="w", values=["Systems Default", "Light Mode", "Dark Mode"], command=theme_callback,variable=themeoption_var)
combobox.pack()
combobox.configure(height=35)
combobox.place(y=80, x=10)

# ---------- Help Button ---------# 
def show_help():
    help_text = """
    Welcome to CyberScan - Content Detector!

    To use the app, follow these steps:
    
    1. Enter your content in the provided text box.
    2. Click the "Quick Scan" button to analyze your content for offensive speech.
    3. The app will provide a result indicating whether the content is offensive or not.
    4. You can also use the "Full Scan" button for a more comprehensive analysis.
    
    Feel free to customize the appearance of the app using the "Appearance Mode" option in the Settings.

    If you encounter any issues or have questions, please contact support team at 219044598@uj.student.ac.za

    Thank you for using CyberScan!
    """
    lines = help_text.split("\n")
    help_line_length = max(len(line) for line in lines)
    help_width = help_line_length * 6

    help_popup = CTkMessagebox(title="How to Use CyberScan Application?", cancel_button="circle", message=help_text, option_1="Exit", width=help_width)

help_button = customtkinter.CTkButton(master=settings_frame, text="Help", cursor='hand2', command=show_help, anchor="center")
help_button.pack(pady=10, padx=10, fill="x")
help_button.configure(height=40)
help_button.place(y=140, x=10)

# ---------- Close Application ---------# 
def close_applic():
    root.destroy()

log_out = customtkinter.CTkButton(master=settings_frame, text="Exit", command=close_applic, cursor='hand2', fg_color="#990000", hover_color="#660000", anchor="center")
log_out.pack()
log_out.place(y=200, x=10)
log_out.configure(height= 40)

# ---------- COLUMN 2 --------- # 
# ---------- Dashboard Frame ---------# 
canvas_dashboard = customtkinter.CTkFrame(master=root)
canvas_dashboard = customtkinter.CTkFrame(root, width=600, height=600)
canvas_dashboard.pack(pady=20, padx=(10, 10), fill="both", expand=True, side="left")

# ---------- Logo ---------# 
light_image = Image.open("images/logo-no-background.png").convert("RGBA")
dark_image = Image.open("images/logo-no.png").convert("RGBA")

logo_image = customtkinter.CTkImage(light_image=light_image, dark_image=dark_image, size=(310, 150))
logo_label = customtkinter.CTkLabel(canvas_dashboard, image=logo_image)
logo_label.configure(text="")
logo_label.place(relx=0.5, rely=0.5, anchor="center")
logo_label.pack(pady=20)

# ---------- Container Frame ---------# 
container = customtkinter.CTkFrame(master=canvas_dashboard)
container = customtkinter.CTkFrame(canvas_dashboard, width=400, height=250)
container.pack(pady=0, padx=50, fill="both", expand=True)

# ---------- Description ---------# 
text = "A CyberScan software is a system that lets you test your content to find out if it is harmful or not. Use the below ChatBox to analyse your message before sending."
description_label = customtkinter.CTkLabel(container, text=text, wraplength=390)
description_label.pack(pady="15")

#---------- Separator ---------
style = ttk.Style()
style.configure("Horizontal.TSeparator", background="black") 
line_separator = ttk.Separator(container, orient="horizontal", style="Horizontal.TSeparator")
line_separator.pack(fill="x", padx=50, pady=10)

# ---------- Message ---------# 
def msg_offfocus(e):
    if not message.get():
        message.delete(0, "end")
        message.insert(0, "Enter your content here")
    message.configure(state="readonly")

def msg_onfocus(e):
    if message.get() == "Enter your content here":
        message.delete(0, "end")
    message.configure(state="normal")

message_txt = customtkinter.CTkLabel(container, text="Message:")
message_txt.pack()
message = customtkinter.CTkEntry(master=container, placeholder_text="Enter your content here")
message.bind('<FocusIn>', msg_onfocus) 
message.bind('<FocusOut>', msg_offfocus) 
message.pack(pady=2, padx=10)
message.configure(height=40, width=300)

def naive_analyze():
    input_text_value = message.get().strip()
    if input_text_value:
        predicted_class = predict_class(input_text_value, vocab, word_probs_off, word_probs_not_off, prior_off, prior_not_off)
        true_labels = [predicted_class]
      
    new_icon_label = show_info_icon(info_img) 
    if hasattr(naive_analyze, "icon_label"):
        naive_analyze.icon_label.pack_forget()
        naive_analyze.icon_label.place_forget()

    result_container.configure(text="Results: " + predicted_class)
    result_container.pack(side="left", padx=0)
    new_icon_label.place(y=15, x=300)

    results_msg = results_message(input_text_value)
    results_label.configure(text=results_msg)

    report_naive_msg = report_naive(input_text_value, d_test_labels, naive_bayes_predictions, word_probs_off, word_probs_not_off)
    model_report_label.configure(text=report_naive_msg)

    explanation = generate_tooltip_message(input_text_value, predicted_class, word_probs_off, word_probs_not_off)

    naive_analyze.icon_label = new_icon_label
    new_icon_label.bind("<Button-1>", lambda event: explanation_box(input_text_value, predicted_class, explanation))

def explain_svm_prediction(input_text, svm_classifier, sv_vectorizer):
    input_features = sv_vectorizer.transform([input_text])
    
    decision_function = svm_classifier.decision_function(input_features)
    
    result = "No Offensive Speech Detected"
    if decision_function[0] > 0:
        result = "Offensive Speech Detected"
    
    decision_function_values = {
        'No Offensive Speech Detected': max(0, -decision_function[0]),
        'Offensive Speech Detected': max(0, decision_function[0])
    }
    
    return result, decision_function_values

def calculate_word_percentage(input_text, sv_vectorizer, svm_classifier):
    input_features = sv_vectorizer.transform([input_text])
    decision_function = svm_classifier.decision_function(input_features)
    
    offensive_probability = max(0, decision_function[0])
    non_offensive_probability = max(0, -decision_function[0])
    
    word_percentages = {}
    
    for word in input_text.split():
        input_text_temp = input_text.replace(word, '')
        input_features_temp = sv_vectorizer.transform([input_text_temp])
        decision_function_temp = svm_classifier.decision_function(input_features_temp)
        offensive_probability_temp = max(0, decision_function_temp[0])
        non_offensive_probability_temp = max(0, -decision_function_temp[0])
        word_percentages[word] = {
            'Offensive Speech Detected': offensive_probability - offensive_probability_temp,
            'No Offensive Speech Detected': non_offensive_probability - non_offensive_probability_temp
        }
    
    return word_percentages

def svm_analyze():
    input_text_value = message.get().strip() 
    if input_text_value:
        result = svm_model.predict_offensiveness(input_text_value, svm_classifier_path, vectorizer_path)

    new_icon_label = show_info_icon(info_img) 
    if hasattr(svm_analyze, "icon_label"):
        svm_analyze.icon_label.pack_forget()
        svm_analyze.icon_label.place_forget()

    result_container.configure(text="Results: " + result)
    result_container.pack(side="left", padx=0)
    new_icon_label.place(y=15, x=300)

    results_msg = results_message(input_text_value)
    results_label.configure(text=results_msg)

    result, decision_function_values = explain_svm_prediction(input_text_value, svm_classifier, sv_vectorizer)
    
    report_svm_msg = svm_model.loadtested()

    svm_exp_results = "\n\nDECISION ANALYSIS:"

    word_percentages = calculate_word_percentage(input_text_value, sv_vectorizer, svm_classifier)

    for word, percentages in word_percentages.items():
        svm_exp_results += f"\nWord: {word},\n"
        for label, percentage in percentages.items():
            svm_exp_results += f"   {label}: {percentage:.2f}\n"

    svm_exp_results += "\nAverage Probability:\n"
    for label, decision_value in decision_function_values.items():
        svm_exp_results += f"   {label}: {decision_value:.2f}\n"

    model_report_label.configure(text=report_svm_msg + svm_exp_results)

    explanation = generate_tooltip_message_svm(input_text_value, result, label, decision_value)

    svm_analyze.icon_label = new_icon_label
    new_icon_label.bind("<Button-1>", lambda event: explanation_box(input_text_value, result, explanation))

def explanation_box(input_text_value, predicted_class, explanation):
    lines = explanation.split("\n")
    max_line_length = max(len(line) for line in lines)
    msg_width = max_line_length * 8 
    
    msg = CTkMessagebox(title=f"Why {predicted_class}?", cancel_button="circle", message=explanation, icon="images/question.png", option_1="Ok", width=msg_width)

def generate_tooltip_message(input_text, predicted_class, word_probs_off, word_probs_not_off):
    words = tokenize(input_text)
    
    explanation = f"Your input: {words}\n\n"

    explanation += "Conditional Probabilities:\n"
    total_prob_off = 0.0
    total_prob_not_off = 0.0

    for word in words:
        word_lower = word.lower() 
        prob_off = word_probs_off.get(word_lower, 0)
        prob_not_off = word_probs_not_off.get(word_lower, 0)

        label_off = class_mapping_t["1"]
        label_not_off = class_mapping_t["0"]
        
        total_prob_off += prob_off
        total_prob_not_off += prob_not_off

    explanation += f"P('{input_text}' | Offensive Speech) = {total_prob_off:.4f}\n"
    explanation += f"P('{input_text}' | No Offensive Speech) = {total_prob_not_off:.4f}"

    if predicted_class == "Offensive Speech Detected":
        new_predicted_class = "Offensive"
    else:
        new_predicted_class = "Not Offensive"

    explanation += f"\n\nHence, the predicted speech is {new_predicted_class}."

    return explanation

def generate_tooltip_message_svm(input_text, result, label, decision_value):
    words = tokenize(input_text)

    explanation = f"Your input: {words}\n\nDecision Interpretation:\n"

    result, decision_function_values = explain_svm_prediction(input_text, svm_classifier, sv_vectorizer)
    
    for label, decision_value in decision_function_values.items():
        explanation += f"   {label}: {decision_value:.2f}\n"

    explanation += f"\nHence, {result}."

    return explanation

# ---------- Analyse buttons | container ---------# 
analysebutton_cont = customtkinter.CTkFrame(master=container, fg_color="transparent", bg_color="transparent")
analysebutton_cont.configure(height=40, width=300)
analysebutton_cont.pack(pady=15, padx=10)

# ---------- Naive Bayes | Quick Scan ---------# 
analyse_button = customtkinter.CTkButton(master=analysebutton_cont, text="Quick Scan", command=naive_analyze, cursor='hand2')
analyse_button.pack(side="left" ,padx=0, anchor="w")
analyse_button.configure(height=40)

#---------- Separator ---------
style2 = ttk.Style()
style2.configure("Horizontal.TSeparator") 
line_separator2 = ttk.Separator(analysebutton_cont, orient="horizontal", style="Horizontal.TSeparator")
line_separator2.pack(side="left", fill="x", padx=4, pady=5)

# ---------- Support Vector Machine | Full Scan ---------# 
analyse_button_svm = customtkinter.CTkButton(master=analysebutton_cont, text="Full Scan", command=svm_analyze, cursor='hand2')
analyse_button_svm.pack(side="left",pady=0, padx=0, anchor="e")
analyse_button_svm.configure(height=40)

# ---------- Clean button | container ---------# 
cleanbutton_cont = customtkinter.CTkFrame(master=container, bg_color="transparent", fg_color="transparent")
cleanbutton_cont.configure(height=40, width=300)
cleanbutton_cont.pack(padx=10)

# ---------- Clear button ---------# 
clear_icon = customtkinter.CTkImage(Image.open("images/bin.png").convert("RGBA"))

def clear_text():
    message.delete(0, 'end')
    result_container.configure(text="Results: ")

    try:
        if hasattr(naive_analyze, 'icon_label'):
            naive_analyze.icon_label.pack_forget()
            naive_analyze.icon_label.place_forget()
    except NameError:
        pass

    try:
        if hasattr(svm_analyze, 'icon_label'):
            svm_analyze.icon_label.pack_forget()
            svm_analyze.icon_label.place_forget()
    except NameError:
        pass

    try:
        naive_report_label.configure(text="")
    except NameError:
        pass

    try:
        svm_report_label.configure(text="")
    except NameError:
        pass

clear_button = customtkinter.CTkButton(master=cleanbutton_cont, text="Clear", cursor="hand2", fg_color="#990000", hover_color="#660000", command=clear_text)
clear_button.configure(height=40)
clear_button.pack(anchor="center")

# ---------- Results Frame | Container2 ---------# 
container2 = customtkinter.CTkFrame(master=canvas_dashboard)
container2.pack(pady=15, padx=180, fill="both", expand=False)

# ---------- Result Container ---------# 
result_container = customtkinter.CTkLabel(master=container2, text="Results: ", wraplength=400)
result_container.pack(pady=15, padx=0, fill="both", expand=True)

def create_info_image():
    info_icon_light = Image.open("images/info.png").convert("RGBA")
    info_icon_dark = Image.open("images/info_light.png").convert("RGBA")
    info_img = customtkinter.CTkImage(light_image=info_icon_light, dark_image=info_icon_dark, size=(15, 15))
    return info_img

def show_info_icon(info_img):
    info_label = customtkinter.CTkLabel(container2, image=info_img, cursor="hand2")
    info_label.configure(text="")
    info_label.pack(side="left")
    return info_label

info_img = create_info_image()

# ---------- COLUMN 3 ---------# 
# ---------- Overall Results ---------# 
overall_frame = customtkinter.CTkScrollableFrame(master=root, width=350, height=600)
overall_frame.pack(pady=20, padx=(10, 20), fill="both", expand=False, side="left")

# ---------- Overall Description ---------#  
result_description_label = customtkinter.CTkLabel(overall_frame, text= "Classification Report", anchor="w", font=("Liberation Sans", 16, "bold"), wraplength=400)
result_description_label.pack(pady=10, padx=10, fill="x")

def results_message(input_text):
    words = input_text
    
    resul = f"MESSAGE:\n[{words}]"

    return resul

# ---------- REsults Description ---------#  
results_label = customtkinter.CTkLabel(overall_frame, text="", anchor="w", justify="left", wraplength=350)
results_label.pack(pady=10, padx=10, anchor="w", fill="x", expand=False)

def report_naive(input_text, test_labels, naive_bayes_predictions, word_probs_off, word_probs_not_off):
    words = tokenize(input_text)

    correct_predictions = sum(1 for true_label, predicted_label in zip(test_labels, naive_bayes_predictions) if true_label == predicted_label)
    total_predictions = len(test_labels)
    accuracy = correct_predictions / total_predictions
    accuracy_percentage = accuracy * 100

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for true_label, predicted_label in zip(test_labels, naive_bayes_predictions):
        if true_label == "No Offensive Speech Detected" and predicted_label == "No Offensive Speech Detected":
            true_positive += 1
        elif true_label == "No Offensive Speech Detected" and predicted_label == "Offensive Speech Detected":
            false_negative += 1
        elif true_label == "Offensive Speech Detected" and predicted_label == "No Offensive Speech Detected":
            false_positive += 1
        elif true_label == "Offensive Speech Detected" and predicted_label == "Offensive Speech Detected":
            true_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    precision_percentage = precision * 100
    recall_percentage = recall * 100
    f1_score_percentage = f1_score * 100

    results_naive = f"Accuracy: {accuracy_percentage:.2f}%\n"
    results_naive += f"Precision: {precision_percentage:.2f}%\n"
    results_naive += f"Recall: {recall_percentage:.2f}%\n"
    results_naive += f"F1-Score: {f1_score_percentage:.2f}%\n\n"
    results_naive += "Classification Matrix(Naive Bayes):\n"
    results_naive += f"                   precision  recall   F1      support\n"
    results_naive += f"\n                       {precision:.2f}        {recall:.2f}     {f1_score:.2f}      {true_positive + false_negative}\n"
    results_naive += f"\nmacroAvg    {(precision + (1 - precision)) / 2:.2f}        {(recall + (1 - recall)) / 2:.2f}     {(f1_score + (1 - f1_score)) / 2:.2f}    {total_predictions}\n"
    results_naive += f"\nweighAvg     {accuracy:.2f}       {accuracy:.2f}      {accuracy:.2f}     {total_predictions}\n"
    results_naive += f"\nConfusion Matrix:\n"
    results_naive += f" [[{true_positive}   {false_negative}]\n"
    results_naive += f" [{false_positive}   {true_negative}]]"

    results_naive += "\n\nCONDITIONAL PROBABLITIES:\n"
    total_prob_off = 0.0
    total_prob_not_off = 0.0

    for word in words:
        word_lower = word.lower() 
        prob_off = word_probs_off.get(word_lower, 0)
        prob_not_off = word_probs_not_off.get(word_lower, 0)

        label_off = class_mapping_t["1"]
        label_not_off = class_mapping_t["0"]
        
        results_naive += f"P('{word}')| {label_off}: {prob_off:.4f}\n\n"
        results_naive += f"P('{word}')| {label_not_off}: {prob_not_off:.4f}\n\n"

        total_prob_off += prob_off
        total_prob_not_off += prob_not_off

    results_naive += f"Average Probability:\n"
    results_naive += f"P('{input_text}' | Offensive Speech) = {total_prob_off:.4f}\n"
    results_naive += f"P('{input_text}' | No Offensive Speech) = {total_prob_not_off:.4f}"

    return results_naive

# ---------- Report Description ---------#  
model_report_label = customtkinter.CTkLabel(overall_frame, text="", anchor="w", justify="left", wraplength=345)
model_report_label.pack(pady=15, padx=10, anchor="w", fill="x", expand=False)

root.bind('<Return>', lambda event: naive_analyze())

root.mainloop()
