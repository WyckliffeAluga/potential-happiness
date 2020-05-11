# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:23:35 2020

@author: wyckliffe
"""


import mailbox
import re 
import spacy 
import random
import rdflib 
from rdflib import Graph, Literal, RDF 
from rdflib.namespace import FOAF
from spacy.symbols import nsubj, xcomp, dobj, pobj, prep, attr, VERB, PRON, NOUN, PROPN, PUNCT
import pandas as pd 
import pickle




class Enron(): 
    
    def __init__(self) :
        
        self.mbox = mailbox.mbox("emails-all.mbx")
        print("loaded all emails")
        print("getting messages")
        self.msgs = self.mbox.keys()
        print("Loading spacy")
        self.nlp = spacy.load('en')
        print("start creating relationships and saving in graph")
        print("Building graphs relax")
        self.graph_email_relationships()
       
        
    def viewEmail(self): 
      
        for msg_key in self.msgs[5:10]: 
            message = self.mbox.get(msg_key)
            print(msg_key, "/", message['subject'], 
                  "/", message['From'], "/", message['Date'])
            print(".........................")
            print(message.get_payload())
            print(".........................")
            
    def cleanEmail(self, messageBody): 
        
        replies = re.compile('(\-+Original Message|\-\-+|~~+).*', re.DOTALL) # finds "Original message ... " 
        messageBody = re.sub(replies, '', messageBody) 
        messageBody = re.sub(r'=\d\d', ' ', messageBody) # remove funny email formatting 
        messageBody = re.sub(r'\s*>.*', '', messageBody) # remove quotes 
        messageBody = re.sub(r'https?://.*?\s', '' , messageBody) # remove links 
        bigSpace = re.compile(r'\n\n\n\n\n+.*' , re.DOTALL) # find large gaps 
        messageBody = re.sub(bigSpace, '', messageBody)
        bigIndent = re.compile(r'(\t| {4,}).*', re.DOTALL) # find big indentations 
        messageBody = re.sub(bigIndent, '' , messageBody)
        emailPaste  = re.compile(r'(From|Subject|To): .*', re.DOTALL)  # find pasted emails 
        messageBody = re.sub(emailPaste, '',  messageBody)
        messageBody = re.sub(r'=(\s*)\n', '\1', messageBody) # fix broken lines 
        messageBody = re.sub(r' ,([stm])', '\'1' , messageBody) # fix funny apostrophe 's and 't and 'm 
        messageBody = re.sub(r'([\?\.])\?', '\1', messageBody) # fix funny extra question marks 
        messageBody = re.sub(r'\x01', ' ', messageBody) # fix odd spaces 
        
        return messageBody.strip()
    
    def checkCleaning(self): 
        
          for msg_key in self.msgs[10:15]:
              message = self.mbox.get(msg_key)
              messageBody = self.cleanEmail(message.get_payload())
              print(msg_key, "/", message['subject'], 
                  "/", message['From'], "/", message['Date'])
              print(".........................")
              print(messageBody)
        
    def goodToFrom(self, message): 
        # filter out announcements and mailiing list emails, and emails to large numbers of people
        # we want to extract statements aboout people doing things and indentify who is making that statement 
        # If it is from an email list or to an email list and also makes sure it is from an @enron.com email who is sending it 
        # and there no more than 3 people receiving the email
        
        return (re.match(r'.*(admin|newsletter|list|announce|all[\._]|everyone[\.\_]).*',
                         message['From'], re.IGNORECASE) is None and
                         message['To'] is not None and
                         re.match(r'.*(admin|newsletter|list|announce|all[\._]|everyone[\.\_]).*', 
                         message['To'], re.IGNORECASE) is None and
                         re.match(r'.*@enron\.com', message['From'], re.IGNORECASE) and
                         len(message['To'].split()) <= 3)
    
    def check_to_from(self): 
        
          for msg_key in self.msgs[150:750]:
               message = self.mbox.get(msg_key)
               if self.goodToFrom(message):
                   messageBody = self.cleanEmail(message.get_payload())
                   if len(messageBody) > 0 : 
                       print(msg_key, "/", message['subject'], 
                      "/", message['From'], "/", message['Date'])
                       print(".........................")
                       print(messageBody)
                       
    def relationships(self, doc): 
        relationships = [] # initialize relationships 
        
        for possible_subject in doc: # for every word in the document 
        
            if possible_subject.dep == nsubj and possible_subject.head.pos == VERB: # ask if it is a subject that is has 
            # a dependency arrow of subject and if the parent is the verb itself. 
                subj = possible_subject # then we are good for the subject
                verb = possible_subject.head # extract the verb which is the head of the subject
                print(subj, verb)
                
                for vr in verb.rights : # arrows going to the right (out of the verb) all the things this verb is talking about
                    if vr.dep == xcomp : # we have to check if it is a composite object 
                        for vc in vr.children : # we now have to look at the children of the composite 
                            if vc.dep == dobj and vc.pos == NOUN: # we grab all the nouns
                                relationships.append((subj, verb, vr, vc))
                    if vr.dep == prep: # if it a preposition we are going to do the same 
                        for vc in vr.children: 
                            if vc.dep == pobj and vc.pos == NOUN: 
                                relationships.append((subj, verb, vr, vc))
                    if vr.dep == dobj and vr.pos == NOUN: # if the verb is talking about one object we just add that one object 
                        relationships.append((subj, verb, vr))
                    print(vr, vr.dep_, vr.pos)
                    
        return relationships
    
    def reference(self, doc, pronoun, messageFrom, messageTo): 
        # an anaphora resolution 
        
        # if the pronoun is I , myself, or me or we, we use message from name 
        if pronoun.text.lower() in ['i', 'myself', 'me', 'we']:
            return messageFrom
        
        # if pronoun is 'you' or 'your' return messageTo 
        elif pronoun.text.lower() == 'you' or pronoun.text.lower == 'your': 
            return messageTo
        
        # else , find the root verb for pronoun , then find the subject 
        else:    
            w = pronoun
            while w.head != w : # go up the tree to root 
                w = w.head
            
            # now find the nsubj if it exists 
            for c in w.children : 
                if c.dep == nsubj: 
                    return c.text 
            return None 
        
    def checkReference(self): 
        doc = self.nlp('Bob said he knows about the scam')
        rels = self.relationships(doc)
        for rel in rels: 
            for w in rel:
                if w.pos == PRON: 
                    print(self.reference(doc, w, 'me@aol.com', 'you@aol.com'))
                    
    
    def extract_relationships(self, doc, messageFrom, messageTo): 
        """
        Modified relationships to handle more cases

        Returns
        -------
        None.

        """
        relationships = [] 
        for possible_subject in doc : # go through every word in the nlp docc 
        
            if possible_subject.dep == nsubj and possible_subject.head.pos == VERB: # check if the word is a subject and pointed to a vern 
                subj = possible_subject
                verb = possible_subject.head 
                
                if subj.pos == PRON or subj.pos == PROPN : # check if the subject is a pronoun or a proper noun 
                       if subj.pos == PRON: 
                           ref = self.reference(doc, subj, messageFrom, messageTo) # find out who it refers to
                           if ref is not None: # if we get something back, we use it 
                               subj = ref
                           else: 
                               subj = subj.text # otherwise we are just going to use the original tex 
                       else: 
                           subj = subj.text # if it is a proper noun we just use the original text
                           
                       # we can also ignore worthless subjects 
                        
                       if subj.lower() in ['they', 'it']:
                           continue 
                        
                       for vr in verb.rights: # check what the verb is refering to 
                           if vr.dep == xcomp : # check whether it is a composite 
                               for vc in vr.children : # go through the children and extract nouns 
                                   if vc.dep == dobj and vc.pos == NOUN : 
                                       if vr.idx < vc.idx: # if the second part of the composite comes before the word in question 
                                           relationships.append((subj, verb.lemma_, vr.lemma_ + " " + vc.lemma_)) # we stick if before 
                                       else: 
                                           relationships.append((subj, verb.lemma_, vc.lemma_ + " " + vr.lemma_)) # stick it after
                                            
                           elif vr.dep == prep: # if it is a preposition do the same thing 
                               if vc in vr.children: 
                                   if vc.dep == pobj and vc.pos == NOUN: 
                                       relationships.append((subj, verb.lemma_, vc.lemma_))
                                        
                           elif vr.dep == dobj and (vr.pos == NOUN or vr.pos == PROPN): # if it a comounp and nounn 
                               has_compound = False 
                               for vc in vr.children: 
                                   if vr.dep_ == 'compound' and vc.pos == NOUN : 
                                       has_compound = True 
                                       if vr.idx < vc.idx : 
                                           relationships.append((subj, verb.lemma_ , vr.lemma_ + " " + vc.lemma_))
                                       else: 
                                           relationships.append((subj, verb.lemma_ , vc.lemma_ + " " + vr.lemma_))
                                            
                               if not has_compound: 
                                   relationships.append((subj, verb.lemma_, vr.lemma_))
                           elif vr.dep == attr : 
                               relationships.append((subj, verb.lemma_, vr.lemma_))
                                
            return relationships
        
    def extract(self, doc, messageFrom, messageTo): 
        
        relationships = []
        for possible_subject in doc:
            if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
                subj = possible_subject
                verb = possible_subject.head
                
                if subj.pos == PRON or subj.pos == PROPN:
                    if subj.pos == PRON:
                        ref = self.reference(doc, subj, messageFrom, messageTo)
                        if ref is not None:
                            subj = ref
                        else:
                            subj = subj.text
                    else:
                        subj = subj.text
                    
                    # ignore worthless subjects
                    if subj.lower() in ['they', 'it']:
                        continue
                        
                    for vr in verb.rights:
                        if vr.dep == xcomp:
                            for vc in vr.children:
                                if vc.dep == dobj and vc.pos == NOUN:
                                    if vr.idx < vc.idx:
                                        relationships.append((subj, verb.lemma_, vr.lemma_ + " " + vc.lemma_))
                                    else:
                                        relationships.append((subj, verb.lemma_, vc.lemma_ + " " + vr.lemma_))
                        elif vr.dep == prep:
                            for vc in vr.children:
                                if vc.dep == pobj and vc.pos == NOUN:
                                    relationships.append((subj, verb.lemma_, vc.lemma_))
                        elif vr.dep == dobj and (vr.pos == NOUN or vr.pos == PROPN):
                            has_compound = False
                            for vc in vr.children:
                                if vc.dep_ == 'compound' and vc.pos == NOUN:
                                    has_compound = True
                                    if vr.idx < vc.idx:
                                        relationships.append((subj, verb.lemma_, vr.lemma_ + " " + vc.lemma_))
                                    else:
                                        relationships.append((subj, verb.lemma_, vc.lemma_ + " " + vr.lemma_))
                            if not has_compound:
                                relationships.append((subj, verb.lemma_, vr.lemma_))
                        elif vr.dep == attr:
                            relationships.append((subj, verb.lemma_, vr.lemma_))
        return relationships
    
    def checkExtract_relationships(self, text): 
        
        text = self.nlp(text)
        
        print(self.extract(text, "me@aol.com", "you@aol.com"))
     
    def extract_email_relationships(self, msg_key): 
        
        message = self.mbox.get(msg_key)
       
        if message['From'] is not None and message['To'] is not None : # check if from and to contain something
            try: 
                message_nlp  = self.nlp(self.cleanEmail(message.get_payload()))# clean 
                return self.extract(message_nlp, message['From'], message['To'].split(",")[0]) # if multiple receivers take the first
            except: 
                return [] 
        
        else: 
            return []
        
      
    def check_extract_email_relationships(self): 
     
        key = random.randint(0, len(self.msgs)-1)
        print(key)
        print(self.extract_email_relationships(key))
        
    def graph_email_relationships(self) :
        import bsddb3
        
        g = Graph('Sleepycat', identifier='enron_relationships')
        g.open('enron_relationships.rdf', create =True)
        msg_key_idx = {}
        msg_key_idx_reverse = {}
        
        i = 0 
        message_count = len(self.msgs)
        
        for message_key in self.msgs[0:1000] : # do all messages 
            i  += 1
            
            if i % 100 == 0 : 
                print("Message %d of %d" %(i, message_count))
                
            # find relationships 
            rels = self.extract_email_relationships(message_key)
            
            # for each relationship 
            for (s, p, o) in rels: 
                
                r = (Literal(s), Literal(p), Literal(o))
                
                # add relationship to the graph 
                g.add(r)
                
                # remember which messages and the relationship 
                
                if r in msg_key_idx : 
                    msg_key_idx[r].append(message_key)
                else: 
                    msg_key_idx[r] = [message_key]
                    
                # remember the relationships this message had 
                
                if message_key in msg_key_idx_reverse: 
                    msg_key_idx_reverse[message_key].append(r)
                else: 
                    msg_key_idx_reverse[message_key] = [r]
                    
        message_key_index = pd.DataFrame(msg_key_idx) 
        message_key_index.to_csv(index=False)
        message_key_index_reverse = pd.DataFrame(msg_key_idx_reverse)  
        message_key_index_reverse.to_cv(index=False)
        pickel_g = open("g.obj", "w")
        pickle.dump(pickel_g)
        return (g, msg_key_idx, msg_key_idx_reverse)        
    
e = Enron()