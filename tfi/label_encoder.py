# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:31:15 2015

@author: nightfox
"""

import numpy as np
import pandas as pd

data = pd.read_csv('data/train.csv', encoding='utf-8')
city_group = data['City Group'].tolist()


def labeller(a_list):
    m=0
    n=0
    c={}
    b=list(a_list)
    for item in a_list:
        c.setdefault(item,False)
        if c[item] is False:
            for i in a_list:
                if item==i:
                    b[a_list.index(i)]=m
            c[item]=m
            m+=1
        else:
            b[n] = c[item]
        n+=1
    return pd.Series(np.array(b))

citygroup=labeller(city_group)

print (citygroup)

 
 
