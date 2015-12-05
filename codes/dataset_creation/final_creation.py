# -*- coding: utf-8 -*-
"""
@author: ioanna
"""
# -*- coding: utf-8 -*-
final_plot = {}
final_scripts = {}
scores={}
genres = {}

file_plot = open ("title_plots.txt", "r")

for line in file_plot:
    values = line.split ("\t") 
    final_plot[values[0].lower()] =values[1].lower().replace("-", " ").replace("?", " ").replace(".", " ").replace(",", " ").replace("!", " ")                 
file_plot.close()

file_titlesscr = open ("movie_titles_metadata_script.txt", "r")

for line in file_titlesscr:
    values = line.split (" +++$+++ ") 
    scores[values[1]] = values[3]
    genres[values[1]] = values[5] 
for id in genres:
    genres[id] = genres[id].translate(None, '\n')
#print titlesscr
file_titlesscr.close()

file_scripts = open ("title_scripts.txt", "r")

for line in file_scripts:
    values = line.split (" ++$++ ") 
    #print values[0]
    final_scripts[values[0]] =values[1].lower().replace("-", " ").replace("?", " ").replace(".", " ").replace(",", " ").replace("!", " ").translate(None,'@ù"&%£éèàòì;äÄßüÜöÖ') 
file_scripts.close()
#print final_scripts

count=0
output_final= open ("outputfinalscoregenre.txt","w")
for id in final_plot:
    if id in final_scripts:
        count = count+ 1
        output_final.write(id + " ++$++ " + scores[id] + " ++$++ " + genres[id] + " ++$++ " + final_plot[id].strip() +" ++$++ "+ final_scripts[id].decode('ascii', errors= 'ignore'))
output_final.close()

