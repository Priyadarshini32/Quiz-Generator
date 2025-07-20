#import all the neccessary libraries
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
import random
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('brown')
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('omw-1.4')
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
#from similarity.normalized_levenshtein import NormalizedLevenshtein
import pickle
import time
import spacy
import os
import gdown

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
download_if_not_exists('t5_question_model.pkl', 'YOUR_T5_QUESTION_MODEL_FILE_ID')
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
download_if_not_exists('sentence_transformer_model.pkl', 'YOUR_SENTENCE_TRANSFORMER_MODEL_FILE_ID')
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

def get_nouns_multipartite(content):
    """
    This function takes the content text given and then outputs the phrases which are build around the nouns , so that we can use them for context based distractors
    """
    out=[]
    try:
        nlp=spacy.load("en_core_web_sm")
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        #pos = {'PROPN','NOUN',}
        pos = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}

        #pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection( pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)


        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        #traceback.print_exc()

    return out

def get_keywords(originaltext):
    """
    Try to extract keywords using get_nouns_multipartite (pke).
    If that fails, use a simple NLTK-based fallback with filtering for meaningful keywords.
    """
    keywords = get_nouns_multipartite(originaltext)
    if keywords:
        # Filter out short or non-alphabetic keywords
        keywords = [kw for kw in keywords if kw.isalpha() and len(kw) > 2]
        return keywords[:10]  # Limit to top 10

    # Fallback: Use NLTK to extract nouns as keywords
    print("pke keyword extraction failed, using NLTK fallback.")
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        from collections import Counter

        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(originaltext)
        words = [w for w in words if w.isalpha() and w.lower() not in stop_words and len(w) > 2]
        tagged = pos_tag(words)
        # Extract nouns and proper nouns
        keywords = [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
        # Remove duplicates, capitalize, and limit to top 10
        filtered_keywords = list(dict.fromkeys([w.capitalize() for w in keywords if w.isalpha() and len(w) > 2]))[:10]
        return filtered_keywords
    except Exception as e:
        print("NLTK fallback keyword extraction failed:", e)
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
    """
    summarized_text = summarizer(context, summary_model, summary_tokenizer)

    imp_keywords = get_keywords(context)
    print(f"Extracted keywords: {imp_keywords}")
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
    return output_list[:num_questions]  # Ensure only the requested number of questions are returned

def download_if_not_exists(filename, file_id):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists.")

# Replace the FILE_IDs below with your actual IDs from Google Drive
download_if_not_exists('t5_question_model.pkl', 'https://drive.google.com/file/d/1DvtdMdzHRRzv0TWpp26hg4vtCYdBnoP7/view?usp=sharing')
download_if_not_exists('t5_summary_model.pkl', 'https://drive.google.com/file/d/1xXlW0qrZ1-P5YDjW4R_SQ2JRDrRGk9Qu/view?usp=sharing')
download_if_not_exists('sentence_transformer_model.pkl', 'https://drive.google.com/file/d/13R2f3a_3RxTzjl4hsvhNTTc82ZKnfa3L/view?usp=sharing')




    






