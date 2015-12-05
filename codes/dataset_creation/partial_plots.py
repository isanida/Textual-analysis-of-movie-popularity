# -*- coding: utf-8 -*-
"""
@author: Kiezuz
"""

titles={}
summaries= {}
genres = {}
score = {}
file_titles = open ("movie.metadata_plot.tsv", "r")

for line in file_titles:
    values = line.split ("\t") 
    titles[values[0]] =values[2]
#print titles

file_titles.close()
file_plots = open ("plot_summaries.txt", "r")
for line in file_plots:
    values = line.split ("\t") 
    summaries[values[0]] =values[1]
#print summaries
file_plots.close()

output_plots= open ("title_plots.txt","w")
for id in titles.keys():
    if id in summaries:
        output_plots.write(titles[id] + "\t" + summaries[id])
    
output_plots.close()
 