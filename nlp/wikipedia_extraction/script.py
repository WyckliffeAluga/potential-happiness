# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:38:17 2020

@author: wyckliffe
"""


import wikipedia 
import spacy 
from spacy.matcher import Matcher
from spacy import displacy 
from collections import Counter 
import math 
import re 


class Keywords(): 
    
    def __init__(self): 
        
        self.page = "New York City"
        
    
    def view(self): 
        ny = wikipedia.page(self.page)
        print(ny.content[:1000])
    
    def checking(self): 
        
        page  = wikipedia.page(self.page)
        words = spacy.load("en")
        page_nlp  = words(page.content[:10000])
        
      #  for chunk in page_nlp.noun_chunks: 
       #     print(chunk.text, "/", 
       #           chunk.root.text, "/", 
        #          chunk.root.dep_, "/", 
         #         chunk.root.head.text)
            
      #  for ent in page_nlp.ents: 
          #  print(ent.text, "/", 
           #       ent.start_char, "/", 
           #       ent.end_char, "/", 
           #      ent.label)  
        
    def matcher(self): 
        page  = wikipedia.page(self.page)
        nlp = spacy.load("en")
        page_nlp  = nlp(page.content[:10000])
        
        matcher = Matcher(nlp.vocab)
        matched_sentences = [] # collect data of matched sentences to be visualized 
        matched_phrases = [] # just the matched phrases withough the sentence 
        
        def collect_sents(matcher, doc, i,matches): 
            match_id , start, end = matches[i]
            span = doc[start : end] # matched span 
            sentence = span.sent # sentence containing the matched span 
            
            # append mock entity for match in displaCy style to matched sentences 
            # get the match span by ofsetting the start and end of the span with the 
            # start and end of the sentence in the doc 
            
            match_ents = [{'start': span.start_char - sentence.start_char, 
                           "end": span.end_char - sentence.start_char, 
                           "label": 'MATCH'}]
            
            matched_sentences.append({'text': sentence.text, 
                                      "ent": match_ents})
            matched_phrases.append(span.text)
        
       # pattern = [{"POS":"NOUN"}, {"POS":"NOUN"}]
       # matcher.add("nounphrase", collect_sents, pattern) # add pattern 
       # matches = matcher(page_nlp)
        
        #displacy.render(matched_sentences, style='ent', manual=True)
        
        # other patterns 
        patterns = [[{"POS":"NOUN", 
                      "IS_ALPHA" : True, 
                      "IS_STOP": False, 
                      "OP": "+"}]]
        matcher = Matcher(nlp.vocab)
        matched_sentences = []
        matched_phrases = []
        
        for pattern in patterns:
            matcher.add('keyword', collect_sents, pattern)
        
        matches = matcher(page_nlp)
        
        return sorted(matched_phrases)
            
    def C_value(self): 
        
       keywords = dict(Counter(self.matcher()))
       
       keyword_c_value = {} 
       
       for keyword in sorted(keywords.keys()): 
           
           parent_terms = list(filter(lambda t: t != keyword and re.match("\\b%s\\b" % keyword, t), 
                                      keywords.keys()))
           keyword_c_value[keyword] = keywords[keyword]
           
           print("TERM:", keyword, "PARENT tERMS:", parent_terms)
           
           for pt in parent_terms: 
               keyword_c_value[keyword] -= float(keywords[pt]) / float(len(parent_terms))
           keyword_c_value[keyword] *= math.log(len(keyword.split()), 2)   
           
       best_keywords = []
       
       for keyword in sorted(keyword_c_value, key=keyword_c_value.get, reverse=True)[:10] :
           best_keywords.append([keyword, keyword_c_value[keyword]])
           
       
       return best_keywords

k = Keywords()