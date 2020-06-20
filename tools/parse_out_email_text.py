#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import streamlit as st

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        #st.write("origin: ", content[1])

        ### remove punctuation method 1
        translator = str.maketrans('', '', string.punctuation)
        text_string = content[1].translate(translator)
        ### remove punctuation method 2
        #import re
        #text_string = re.sub('[' + string.punctuation + ']', '', content[1])

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        sn = SnowballStemmer("english")
        list_words = text_string.split(" ")
        words = ' '.join([sn.stem(word) for word in list_words])
        st.write("words: ",words)
        st.write("-----------------------")

    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print ("text: ", text)



if __name__ == '__main__':
    main()

