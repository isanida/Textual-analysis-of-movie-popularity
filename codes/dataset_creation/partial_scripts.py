# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 17:37:54 2014

@author: macbookair
"""

titlesscr={}
scripts={}
scores={}
genres = {}
file_titlesscr = open ("movie_titles_metadata_script.txt", "r")
for line in file_titlesscr:
    values = line.split (" +++$+++ ") 
    titlesscr[values[0]] =values[1]
    scores[values[0]] = values[3]
    genres[values[0]] = values[5] 
#print titlesscr
file_titlesscr.close()
file_scripts = open ("movie_lines_script.txt", "r")
previous_id = "m0"
script = ""
for line in file_scripts:
    values = line.split(" +++$+++ ")
    if values[2] == previous_id:
        script = script + values[4].strip()
    else:
        scripts[previous_id] = script + "\n"
        previous_id = values[2]
        script = values[4].strip()
script = script + "\n"

file_scripts.close()


output_scripts= open ("title_scripts.txt","w")
for id in titlesscr.keys():
    if id in scripts:
        output_scripts.write(titlesscr[id] + " ++$++ " + scripts[id])

output_scripts.close()
