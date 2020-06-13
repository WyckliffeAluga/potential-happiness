# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:49:26 2020

@author: wyckliffe
"""


import wikipedia
import spacy 
from spacy.matcher import Matcher 
import math
import re 
from collections import Counter 
import pandas as pd


class Keywords() : 
    
    def __init__(self, pagename , num_keywords) :
        
        self.num_keywords = num_keywords
        self.pagename = pagename
        
        self.nlp = spacy.load("en", disable=['parser', "ner", 'textcat'])
        self.matched_phrases = [] 
        patterns = [[{"POS": "NOUN", 
                           "IS_ALPHA":True, 
                           "IS_STOP": False, 
                           "OP":"+"}]]
        self.matcher = Matcher(self.nlp.vocab)
        
        for pattern in patterns: 
            self.matcher.add('keyword', self.collect_sentences, pattern)
            
        print(self.extract_keywords())
        
    
    def collect_sentences(self, matcher, doc, i, matches): 
        
        match_id, start, end = matches[i]
        span = doc[start : end ]
        self.matched_phrases.append(span.lemma_)
        
    def extract_keywords(self): 
        
        page = wikipedia.page(self.pagename)
        page_nlp = self.nlp(page.content)
        matches = self.matcher(page_nlp)
        
        keywords = dict(Counter(self.matched_phrases).most_common(100))
        
        keyword_values = {}
        
        for keyword in sorted(keywords.keys()): 
            parent_terms = list(filter(lambda t: t != keyword and re.match("\\b%s\\b" % keyword, t), 
                                       keywords.keys()))
            keyword_values[keyword] = keywords[keyword]
            
            for pt in parent_terms: 
                keyword_values[keyword] -= float(keywords[pt]) / float(len(parent_terms))
            
            keyword_values[keyword] *= 1 + math.log(len(keyword.split()), 2)
        
        best_keywords = [] 
        
        for keyword in sorted(keyword_values, key=keyword_values.get, reverse=True)[:self.num_keywords]: 
            best_keywords.append([keyword, keyword_values[keyword]])
        
        best_keywords = pd.DataFrame(best_keywords, columns=['Keyword', 'c_value'])
        
        return best_keywords
        
