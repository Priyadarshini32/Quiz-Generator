# --- Imports ---
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
import random
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string

import traceback
from flashtext import KeywordProcessor
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import pickle
import time
import spacy
import os
import gdown
import pke
import stanza


# --- Define download_if_not_exists function here ---
def download_if_not_exists(filename, file_id):
    if not os.path.exists(filename):
        # If file_id is a full URL, extract the ID
        if file_id.startswith("http"):
            import re
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', file_id)
            if match:
                file_id = match.group(1)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists.")

# --- Define function to download and extract Stanza model folder ---
def download_and_extract_stanza(folder_name, file_id):
    import zipfile
    
    if not os.path.exists(folder_name):
        try:
            # For Google Drive folders, we need to download as zip
            # The folder ID is extracted from the URL
            if file_id.startswith("http"):
                import re
                match = re.search(r'/folders/([a-zA-Z0-9_-]+)', file_id)
                if match:
                    file_id = match.group(1)
            
            zip_filename = f"{folder_name}.zip"
            # For folders, use the download format that downloads as zip
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Downloading Stanza model folder as {zip_filename} from Google Drive...")
            
            # Try gdown for folder download
            try:
                import gdown
                gdown.download_folder(f"https://drive.google.com/drive/folders/{file_id}", 
                                    output=folder_name, 
                                    quiet=False, 
                                    use_cookies=False)
                print(f"Stanza model folder {folder_name} downloaded successfully using gdown.")
                return
            except Exception as e:
                print(f"gdown folder download failed: {e}")
            
            # Alternative: try to download the folder as zip
            try:
                # This URL format downloads the entire folder as zip
                zip_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                import requests
                import shutil
                
                response = requests.get(zip_url, stream=True)
                if response.status_code == 200:
                    with open(zip_filename, 'wb') as f:
                        shutil.copyfileobj(response.raw, f)
                    
                    # Extract the zip file
                    print(f"Extracting {zip_filename}...")
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        zip_ref.extractall('.')
                    
                    # Remove the zip file after extraction
                    os.remove(zip_filename)
                    print(f"Stanza model folder {folder_name} extracted successfully.")
                else:
                    raise Exception(f"Failed to download: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"Direct download failed: {e}")
                print("Please manually download the Stanza model folder and place it in the working directory.")
                print("The code will fall back to automatic Stanza download.")
                
        except Exception as e:
            print(f"Error downloading Stanza model folder: {e}")
            print("Will fall back to automatic Stanza download.")
    else:
        print(f"{folder_name} already exists.")

# --- Download all model files ---
download_if_not_exists('t5_question_model.pkl', 'https://drive.google.com/file/d/1DvtdMdzHRRzv0TWpp26hg4vtCYdBnoP7/view?usp=sharing')
download_if_not_exists('t5_summary_model.pkl', 'https://drive.google.com/file/d/1xXlW0qrZ1-P5YDjW4R_SQ2JRDrRGk9Qu/view?usp=sharing')
download_if_not_exists('sentence_transformer_model.pkl', 'https://drive.google.com/file/d/13R2f3a_3RxTzjl4hsvhNTTc82ZKnfa3L/view?usp=sharing')

# Download Stanza model folder from Google Drive
download_and_extract_stanza('stanza_resources', '1vEP-aUg1AfPa7k0Qp0fiZuedPEBiH5k2')

# --- Device setup ---
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Importing necessary libraries
import os

# Getting the current directory
current_dir = os.getcwd()

# Constructing the path to the strings.json file
strings_json_path = os.path.join(current_dir, 'tests', 'data', 'strings.json')

# Now you can use strings_json_path in your code

# Importing necessary libraries
import os
import time
import pickle
from sense2vec import Sense2Vec

# Define the path to the sense2vec model
model_path = 'sense2vec/tests/data'

# Load the Sense2Vec model
s2v = Sense2Vec().from_disk(model_path)

# Define the path to the summary model and tokenizer
summary_model_path = 't5_summary_model.pkl'
summary_tokenizer_path = 't5_summary_tokenizer.pkl'

# Check if summary model exists
if os.path.exists(summary_model_path):
    # Load summary model
    with open(summary_model_path, 'rb') as f:
        summary_model = pickle.load(f)
    print("Summary model found on disk. Loaded successfully.")
else:
    # Download and save summary model
    print("Summary model not found on disk. Downloading...")
    start_time = time.time()
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    end_time = time.time()
    print("Downloaded summary model in", (end_time - start_time) / 60, "minutes. Saving to disk...")
    with open(summary_model_path, 'wb') as f:
        pickle.dump(summary_model, f)
    print("Summary model saved to disk.")

# Check if summary tokenizer exists
print("Loading summary tokenizer from HuggingFace hub...")
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
print("Summary tokenizer loaded successfully.")

# Similarly, repeat the above steps for question model and tokenizer, and sentence transformer model

#Getting question model and tokenizer
if os.path.exists("t5_question_model.pkl"):
    with open('t5_question_model.pkl', 'rb') as f:
        question_model = pickle.load(f)
    print("Question model found in the disc, model is loaded successfully.")
else:
    print("Question model does not exists in the path specified, downloading the model from web....")
    start_time= time.time()
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    end_time = time.time()
    print("downloaded the question model in ",(end_time-start_time)/60," min , now saving it to disc...")
    with open("t5_question_model.pkl", 'wb') as f:
        pickle.dump(question_model,f)
    print("Done. Saved the model to disc.")

print("Loading question tokenizer from HuggingFace hub...")
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
print("Question tokenizer loaded successfully.")

#Loading the models in to GPU if available
summary_model = summary_model.to(device)
question_model = question_model.to(device)

#Getting the sentence transformer model and its tokenizer
# paraphrase-distilroberta-base-v1
if os.path.exists("sentence_transformer_model.pkl"):
    with open("sentence_transformer_model.pkl",'rb') as f:
        sentence_transformer_model = pickle.load(f)
    print("Sentence transformer model found in the disc, model is loaded successfully.")
else:
    print("Sentence transformer model does not exists in the path specified, downloading the model from web....")
    start_time=time.time()
    sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
    end_time=time.time()

    print("downloaded the sentence transformer in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open(os.path.join(os.getcwd(), "sentence_transformer_model.pkl"),'wb') as f:
        pickle.dump(sentence_transformer_model,f)

    print("Done saving to disc.")

# --- Load Stanza model from downloaded folder ---
stanza_resources_path = 'stanza_resources'

if os.path.exists(stanza_resources_path):
    try:
        # Set the STANZA_RESOURCES_DIR environment variable to point to our downloaded folder
        os.environ['STANZA_RESOURCES_DIR'] = os.path.abspath(stanza_resources_path)
        
        # Initialize Stanza pipeline using the downloaded resources
        stanza_nlp = stanza.Pipeline('en', 
                                   processors='tokenize,pos', 
                                   dir=stanza_resources_path,
                                   use_gpu=torch.cuda.is_available())
        print("Stanza model loaded from downloaded folder successfully.")
    except Exception as e:
        print(f"Error loading stanza model from folder: {e}")
        print("Falling back to automatic download...")
        try:
            stanza.download('en')
            stanza_nlp = stanza.Pipeline('en', processors='tokenize,pos')
            print("Stanza model downloaded and loaded successfully.")
        except Exception as e2:
            print("Error with stanza fallback download:", e2)
            stanza_nlp = None
else:
    # Fallback: download Stanza model automatically
    print("Stanza model folder not found. Downloading automatically...")
    try:
        stanza.download('en')
        stanza_nlp = stanza.Pipeline('en', processors='tokenize,pos')
        print("Stanza model downloaded and loaded successfully.")
    except Exception as e:
        print("Error downloading stanza model:", e)
        stanza_nlp = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocesstext (content):
  """
  this function takes a piece of text (content), tokenizes it into sentences, capitalizes the first letter of each sentence, and then concatenates the processed sentences into a single string, which is returned as the final result. The purpose of this function could be to format the input content by ensuring that each sentence starts with an uppercase letter.
  """
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text,model,tokenizer):
  """
  This function takes the given text along with the model and tokenizer, which summarize the large text into useful information
  """
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)

  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary

def get_keywords_nltk_main(originaltext):
    """
    Main keyword extraction using NLTK with enhanced features.
    Uses POS tagging, frequency analysis, and filtering for better results.
    """
    keywords = []
    print("Using NLTK as main keyword extraction method.")
    
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk import pos_tag
        from collections import Counter
        import re

        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        stop_words = set(stopwords.words('english'))
        
        # Enhanced preprocessing
        # Remove extra whitespace and normalize text
        text = re.sub(r'\s+', ' ', originaltext.strip())
        
        # Tokenize into sentences and then words
        sentences = sent_tokenize(text)
        all_words = []
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            # Filter words: alphabetic, not stopwords, length > 2
            filtered_words = [w for w in words if w.isalpha() and 
                            w not in stop_words and len(w) > 2]
            all_words.extend(filtered_words)
        
        # POS tagging on the filtered words
        tagged_words = pos_tag(all_words)
        
        # Extract nouns, proper nouns, and adjectives
        target_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
        pos_filtered = [word for word, pos in tagged_words if pos in target_pos]
        
        # Count frequency and get most common words
        word_freq = Counter(pos_filtered)
        
        # Get top words by frequency, but ensure diversity
        common_words = word_freq.most_common(20)
        
        # Extract keywords with some diversity (not just most frequent)
        keywords = []
        used_roots = set()
        
        for word, freq in common_words:
            # Simple stemming check to avoid very similar words
            root = word[:4] if len(word) > 4 else word
            if root not in used_roots and len(keywords) < 15:
                keywords.append(word.capitalize())
                used_roots.add(root)
        
        # If we don't have enough keywords, add some more unique ones
        if len(keywords) < 10:
            additional_words = [word for word, pos in tagged_words 
                             if pos in target_pos and 
                             word.capitalize() not in keywords]
            
            # Remove duplicates while preserving order
            additional_unique = []
            for word in additional_words:
                if word.capitalize() not in keywords and word.capitalize() not in additional_unique:
                    additional_unique.append(word.capitalize())
            
            keywords.extend(additional_unique[:15-len(keywords)])
        
        print(f"NLTK extracted {len(keywords)} keywords successfully.")
        return keywords[:15]  # Return top 15 keywords
        
    except Exception as e:
        print(f"NLTK main keyword extraction failed: {e}")
        return []

def get_nouns_multipartite_fallback(content):
    """
    PKE fallback keyword extraction using MultipartiteRank.
    Includes improved error logging for debugging.
    """
    print("Using PKE as fallback keyword extraction method.")
    out = []
    try:
        nlp = spacy.load("en_core_web_sm")
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')

        pos = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}

        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')

        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')

        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
            
        print(f"PKE extracted {len(out)} keywords successfully.")
    except Exception as e:
        print("Exception in PKE keyword extraction:")
        traceback.print_exc()
        out = []

    return out

def get_keywords_stanza_final_fallback(originaltext):
    """
    Final fallback using Stanza NLP pipeline for POS tagging.
    Only used if both NLTK and PKE fail.
    """
    print("Using Stanza as final fallback keyword extraction method.")
    keywords = []
    
    if stanza_nlp is not None:
        try:
            # Process text with Stanza
            doc = stanza_nlp(originaltext)
            
            # Extract nouns and proper nouns
            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.upos in ['NOUN', 'PROPN'] and len(word.text) > 2 and word.text.isalpha():
                        keywords.append(word.text.capitalize())
            
            # Remove duplicates while preserving order
            keywords = list(dict.fromkeys(keywords))
            
            if keywords:
                print(f"Stanza keyword extraction successful. Found {len(keywords)} keywords.")
                return keywords[:15]  # Return top 15
                
        except Exception as e:
            print(f"Stanza keyword extraction failed: {e}")
    
    # If Stanza is not available or fails, return empty list
    print("Stanza is not available or failed.")
    return []

def get_keywords(originaltext):
    """
    Main keyword extraction function with new hierarchy:
    1. NLTK (main method)
    2. PKE (fallback)  
    3. Stanza (final fallback)
    """
    # Try NLTK first (main method)
    keywords = get_keywords_nltk_main(originaltext)
    if keywords and len(keywords) >= 5:  # Need at least 5 keywords for good question generation
        print("NLTK main keyword extraction successful.")
        return keywords[:15]  # Return top 15

    print("NLTK main extraction failed or insufficient keywords, trying PKE fallback...")
    
    # Try PKE as fallback
    keywords = get_nouns_multipartite_fallback(originaltext)
    if keywords and len(keywords) >= 5:
        # Filter out short or non-alphabetic keywords
        keywords = [kw for kw in keywords if isinstance(kw, str) and kw.replace(" ", "").isalpha() and len(kw) > 2]
        if keywords:
            print("PKE fallback keyword extraction successful.")
            return keywords[:15]

    print("PKE fallback failed, trying Stanza as final fallback...")
    
    # Try Stanza as final fallback
    keywords = get_keywords_stanza_final_fallback(originaltext)
    if keywords:
        return keywords[:15]
    
    # Ultimate fallback - simple word extraction
    print("All advanced methods failed, using simple word extraction...")
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(originaltext.lower())
        keywords = [w.capitalize() for w in words if w.isalpha() and 
                   len(w) > 3 and w not in stop_words]
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates
        return keywords[:15]
        
    except Exception as e:
        print(f"Ultimate fallback failed: {e}")
        return []

def get_question(context,answer,model,tokenizer):
  """
  This function takes the input context text, pretrained model along with the tokenizer and the keyword and the answer and then generates the question from the large paragraph
  """
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

def filter_same_sense_words(original,wordlist):

  """
  This is used to filter the words which are of same sense, where it takes the wordlist which has the sense of the word attached as the string along with the word itself.
  """
  filtered_words=[]
  base_sense =original.split('|')[1]
  #print (base_sense)
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  """
  This function takes the given word along with the wordlist and then gives out the max-score which is the levenshtein distance for the wrong answers
  because we need the options which are very different from one another but relating to the same context.
  """
  score=[]
  normalized_levenshtein = NormalizedLevenshtein()
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    """
    This function takes the input word, sentence to vector model and top similar words and also the question
    Then it computes the sense of the given word
    then it gets the words which are of same sense but are most similar to the given word
    after that we we return the list of words which satisfy the above mentioned criteria
    """
    output = []
    #print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      output = filter_same_sense_words(sense,most_similar)
      #print ("Similar ",output)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)

    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """
    The mmr function takes document and word embeddings, along with other parameters, and uses the Maximal Marginal Relevance (MMR) algorithm to extract a specified number of keywords/keyphrases from the document. The MMR algorithm balances the relevance of keywords with their diversity, helping to select keywords that are both informative and distinct from each other.
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    """
    the get_distractors_wordnet function uses WordNet to find a relevant synset for the input word and then generates distractor words by looking at hyponyms of the hypernym associated with the input word. These distractors are alternative words related to the input word and can be used, for example, in educational or language-related applications to provide choices for a given word.
    """
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]

      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0:
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors(word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval):
    """
    This function generates distractor words (answer choices) for a given target word in the context of a provided sentence.
    It selects distractors based on their similarity to the target word's context and ensures that the target word itself is not included among the distractors.
    """
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    if len(distractors) == 0:
        return distractors

    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)

    embedding_sentence = origsentence + " " + word.capitalize()
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    max_keywords = min(len(distractors_new), 4)  # Ensure max 4 distractors
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)

    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    return final[1:]  # Return distractors excluding the correct answer


def get_mca_questions(context, num_questions):
    """
    This function generates multiple-choice questions based on a given context.
    It summarizes the context, extracts important keywords, generates questions related to those keywords,
    and provides randomized answer choices, including the correct answer, for each question.
    
    Args:
        context (str): The input text to generate questions from
        num_questions (int): Number of questions to generate
    """
    print("Starting MCQ generation process...")
    print("=" * 60)
    print("Keyword extraction method: NLTK (main) -> PKE (fallback) -> Stanza (final fallback)")
    print("=" * 60)
    
    summarized_text = summarizer(context, summary_model, summary_tokenizer)

    imp_keywords = get_keywords(context)
    print(f"Final extracted keywords: {imp_keywords}")
    print("=" * 60)
    
    output_list = []
    attempted = set()
    # Loop until the desired number of questions is reached or all keywords are exhausted
    while len(output_list) < num_questions and len(attempted) < len(imp_keywords):
        for answer in imp_keywords:
            if answer in attempted:
                continue
            attempted.add(answer)
            ques = get_question(summarized_text, answer, question_model, question_tokenizer)
            # Post-process: skip questions that are too short or contain the answer verbatim
            if len(ques.split()) < 5 or answer.lower() in ques.lower():
                print(f"Skipping low-quality question: {ques}")
                continue
            distractors = get_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2)
            if not distractors:
                distractors = [k for k in imp_keywords if k.lower() != answer.lower()]
            print(f"Keyword: {answer}, Distractors: {distractors}")
            # Ensure at least 3 distractors (excluding the answer)
            distractor_set = set([d.capitalize() for d in distractors if d.lower() != answer.lower()])
            if len(distractor_set) < 3:
                print(f"Skipping keyword '{answer}' due to insufficient distractors.")
                continue
            options = [answer.capitalize()] + random.sample(list(distractor_set), 3)
            random.shuffle(options)
            alpha_list = ['(a)', '(b)', '(c)', '(d)']
            correct_index = options.index(answer.capitalize())
            output_list.append((ques, options, alpha_list[correct_index]))
            print(f"Generated question: {ques}\nOptions: {options}\nCorrect: {alpha_list[correct_index]}")
            if len(output_list) == num_questions:
                break
    if not output_list:
        print("No questions could be generated from the provided context.")
    
    print("=" * 60)
    print(f"MCQ generation completed! Generated {len(output_list)} questions.")
    print("Summary: The keyword extraction uses a new hierarchy: NLTK (main) -> PKE (fallback) -> Stanza (final fallback)")
    print("=" * 60)
    
    return output_list[:num_questions]