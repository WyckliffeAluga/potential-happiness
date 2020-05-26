# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:55:00 2020

@author: wyckliffe
"""


import csv 
import re 
import json 
import matplotlib.pyplot as plt
from essentia.standard import * 


class Music() : 
    
    def __init__(self): 
        
        self.track_metadata = self.track_loading()
    
    def track_loading(self): 
        
        track_metadata = {}
        
        with open("raw_tracks.csv", encoding='utf-8') as f : 
            read = csv.reader(f, skipinitialspace=True, quotechar="\"")
            headers = [] 
            rownum = 0 
            
            for row in read: 
                if rownum == 0 : 
                    headers = row 
                    rownum += 1 
                else: 
                    trackid = '%06d'% int(row[0])
                    fname   = 'fma/fma_full/%s/%s.mp3' % (trackid[:3], trackid)
                    track_metadata[fname] = dict(zip(headers, row))
                    
                    if len (track_metadata[fname]['track_genres']) > 0 :
                        genres  = json.loads(re.sub(r'\'', '"', track_metadata[fname]['track_genres'])) # fix json 
                        track_metadata[fname]['track_genres'] = list(map(lambda g: g['genre_title'] , genres))
        
        return track_metadata
    
    def check_metadata(self): 
        
        return (len(self.track_metadata), 
                list(self.track_metadata.keys())[:10], 
                list(self.track_metadata['fma/fma_full/000/000048.mp3'].keys()), 
                self.track_metadata['fma/fma_full/000/000048.mp3']
                )
    
    def track_listens(self): 
        plt.hist(list(filter(lambda listens: listens < 4000, 
                             map(lambda t: int(self.track_metadata[t]['track_listens']), 
                                 self.track_metadata.keys()))), bins=50)
        plt.show() 
        
    def track_interests(self): 
        plt.hist(list(filter(lambda interest : interest < 4000 , 
                             map(lambda t : int(self.track_metadata[t]['track_interest']), 
                                 self.track_metadata.keys()))), bins=50 )
        plt.show() 
        
    def track_favorites(self): 
        plt.hist(list(filter(lambda favorites : favorites < 30, 
                             map(lambda t : int(self.track_metadata[t]['track_favorites']), 
                                 self.track_metadata.keys()))), bins=50 )

m = Music()