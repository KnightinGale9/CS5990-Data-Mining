# -------------------------------------------------------------------------
# AUTHOR: Zhong Ooi
# FILENAME: Similarity.py
# SPECIFICATION: Finding the cosine similarity between documents
# and finding which documents are the most similar.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"
doc_list =[doc1,doc2,doc3,doc4]
# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here
doc_matrix=[]
for doc in doc_list:
    doc=doc.replace(',', '')
    word_list=doc.split()
    document_term=[0,0,0,0,0,0,0,0]
    for word in word_list:
        if word.lower()=="soccer":
            document_term[0]= document_term[0]+1
        if word.lower()=="favorite":
            document_term[1] = document_term[1] + 1
        if word.lower()=="sport":
            document_term[2] = document_term[2] + 1
        if word.lower()=="like":
            document_term[3] = document_term[3] + 1
        if word.lower()=="one":
            document_term[4] = document_term[4] + 1
        if word.lower()=="support":
            document_term[5] = document_term[5] + 1
        if word.lower()=="olympic":
            document_term[6] = document_term[6] + 1
        if word.lower()=="games":
            document_term[7] = document_term[7] + 1
    doc_matrix.append(document_term.copy())
print("Document term matrix: [soccer, favorite, sport, like, one, support, olympic, games]")
for i in range(len(doc_matrix)):
    print("DOC", i+1, doc_matrix[i])
# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here

#Cosine Similarity between 2 vectors
# print("\ncosine similarity between 2 vectors")
output=[]
for i in range(len(doc_matrix)):
    for j in range(1,len(doc_matrix)-i):
        output.append([i,i+j,cosine_similarity([doc_matrix[i]],[doc_matrix[i+j]])])
        # print(f'DOC{i+1}', [doc_matrix[i]], f'DOC{i+j+1}', [doc_matrix[i + j]], f'cosine similarity:{float(output[len(output)-1][2])}')

#Cosine Similarity between multiple vectors
print("\nCosine similarity between multiple vectors")
output_multiple=[]
output_multiple.append([[0,1,2],cosine_similarity([doc_matrix[0],doc_matrix[1],doc_matrix[2]])])
output_multiple.append([[0,1,3],cosine_similarity([doc_matrix[0],doc_matrix[1],doc_matrix[3]])])
output_multiple.append([[0,2,3],cosine_similarity([doc_matrix[0],doc_matrix[2],doc_matrix[3]])])
output_multiple.append([[1,2,3],cosine_similarity([doc_matrix[1],doc_matrix[2],doc_matrix[3]])])
output_multiple.append(cosine_similarity(doc_matrix))
print(cosine_similarity(doc_matrix))

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here
maximum=max(output,key=lambda x:x[2])
print(f'\nThe most similar documents are: DOC{maximum[0]+1} and DOC{maximum[1]+1} with cosine similarity = {float(maximum[2][0])}.')
