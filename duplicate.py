


# get_ipython().system('pip install pytesseract')
# get_ipython().system('pip install opencv-python==4.5.5.64')
# get_ipython().system('pip install -U spacy')
# get_ipython().system('python -m spacy download en_core_web_sm')



import cv2
import pytesseract
import language_tool_python
from gingerit.gingerit import GingerIt
import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from autocorrect import spell
from heapq import nlargest

def correct_sentence(line):
        lines = line.strip().split(' ')
        new_line = summary
        similar_word = {}
        for l in lines:
            new_line += spell(l) + " "
        # similar_word[l]=spell.candidates(l)
        return new_line

def run(filename):
    image = cv2.imread(filename)

    base_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 50))
    dilate = cv2.dilate(thresh, kernal, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h > 200 and w > 250:
            roi = base_image[y:y+h, x:x+w]
            cv2.rectangle(image, (x,y), (x+w, y+h), (36, 255, 12), 2)

    cv2.imwrite("sample_boxes.png", image)

    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    ocr_result_original = pytesseract.image_to_string(base_image)

    text=ocr_result_original

    stopwords=list(STOP_WORDS)

    nlp=spacy.load('en_core_web_sm')

    doc=nlp(text)

    tokens=[token.text for token in doc]
    print(tokens)

    
    

    word_frequencies={}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text]=1
                else:
                    word_frequencies[word.text]+=1

    print(word_frequencies)

    max_frequency=max(word_frequencies.values())
    print(max_frequency)

    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency

    print(word_frequencies)

    sentence_tokens=[sent for sent in doc.sents]
    print(sentence_tokens)

    sentence_scores={}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
                
    print(sentence_scores)

    select_length=int(len(sentence_tokens)*0.3)
    select_length

    summary=nlargest(select_length,sentence_scores,key=sentence_scores.get)

    final=' '.join(map(str,summary))

    return final






















