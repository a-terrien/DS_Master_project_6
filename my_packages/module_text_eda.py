# Télécharger la liste de stopwords
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize
from re import sub
from string import punctuation
import unicodedata

# Les mots à supprimer
unwanted_words = set(['', 'products', 'product', 'free', 'rs', 'buy', 'delivery', 'shipping', 'cash', 'cm', 
                      'flipkart', 'com', 'flipkartcom', 'online', 'price', 'sales', 'features', 'Genuine', 'india',
                      'specifications', 'discounts', 'prices', 'key', 'great'])
flipkart_stopword = list(stop_words)+list(unwanted_words)

# Fonctions de nettoyage et de normalisation du texte // version NLTK 

# Fonctions de nettoyage (CE1)
def cleaning_text(doc, stopword_step=True):
    # Définir une expression régulière pour identifier les points collés aux 
    # mots qui ne sont pas des domaines informatiques (.com, .gov, etc.) 
    # et qui sont dû à des fautes de frappes
    pattern_dot = r"(?<=\w)\.(?=\w)"

    # Définir une expression régulière pour identifier les 's collés aux mots
    pattern_s = r"(?<=\w)'s"

    # Appliquer les substitutions sur le texte
    doc = sub(pattern_dot, " ", doc)
    doc = sub(pattern_s, "", doc)

    # Passage en minuscule
    doc = doc.lower()

    # Suppression des chiffres
    doc = ''.join([ch for ch in doc if not ch.isdigit()])
    
    # Suppression de la ponctuation
    doc = doc.translate(str.maketrans('', '', punctuation))
    
    # Suppression des accents
    doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('utf-8')
    
    # Choix de suppression des mots communs ou de liaisons 
    if stopword_step == True:
      # Suppression des mots de liaison
      #stop_words = set(stopwords.words('english'))
      words = doc.split()
      words = [word for word in words if word not in flipkart_stopword]
      doc = ' '.join(words)

    # Correction de fautes d'orthographe hypersimplifié utilisant la librairie
    # enchant
    # doc = correct_doc(doc) -> pas assez de connaissance 
    # pour être sûr de l'utiliser correctement

    # Suppression des espaces et des retours à la ligne
    doc = sub(r'\s+', ' ', doc).strip()
    return doc

# fonction qui nettoie et tokénise un texte (CE2)
def tokenize(doc):
    doc = cleaning_text(doc, stopword_step=True)
    tokens = word_tokenize(doc)
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

# fonction qui nettoie et tokénise pour un texte traité par une technique dl (CE2)
def tokenize_dl(doc):
    doc = cleaning_text(doc, stopword_step=False)
    tokens = word_tokenize(doc)
    return tokens

# Fonction de stemmatisation (CE3) 
stemmer = PorterStemmer()
def stemming(doc):
    tokens = tokenize(doc)
    stems = [stemmer.stem(token) for token in tokens]
    return stems

# Fonction de lemmatisation (CE4)
lemmatizer = WordNetLemmatizer()
def lemmatisation(doc):
    tokens = tokenize(doc)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

# Fonction de préparation d'un doc passant par une tokenisation
def transform_bow_fct(doc) :
    tokens = tokenize(doc)  
    doc = ' '.join(tokens)
    return doc

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(doc) :
    tokens = tokenize_dl(doc)  
    doc = ' '.join(tokens)
    return doc

# Fonction de préparation d'un doc passant par une lemmatisation
def transform_bow_lemm_fct(doc) :
    lemms = lemmatisation(doc)  
    doc = ' '.join(lemms)
    return doc

# Fonction de préparation d'un doc passant par un stemming
def transform_bow_stem_fct(doc) :
    stems = stemming(doc)  
    doc = ' '.join(stems)
    return doc

