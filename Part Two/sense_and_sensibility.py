import PyPDF2
import os 
import string

file_path = os.path.dirname(os.path.abspath(__file__))
file_handle = open(file_path + "/Sense-and-Sensibility-by-Jane-Austen.pdf", "rb")
pdfReader = PyPDF2.PdfReader(file_handle)
page_number = len(pdfReader.pages) # this tells you total pages
page_object = pdfReader.pages[0] # We just get page 0 as example
page_text = page_object.extract_text() # this is the str type of full page

frequency_table = {}
for page in pdfReader.pages:
    page_text = page.extract_text()
    
    # remove punctuation
    page_text = page_text.replace("https://www.fulltextarchive.com", "")
    page_text = page_text.replace("--", " ")
    page_text = page_text.translate(str.maketrans("", "", string.punctuation + "0123456789"))
    page_text = page_text.replace("Full Text Archive", "")
    page_text = page_text.replace("CHAPTER", "")

    # remove heading and page number lines
    word_list = page_text.split()
    for word in word_list:
        word = word.lower()
        if word in frequency_table:
            frequency_table[word] += 1
        else:
            frequency_table[word] = 1


print(frequency_table)

for word in word_list:
    if word in frequency_table:
        frequency_table[word] += 1
    else:
        frequency_table[word] = 1