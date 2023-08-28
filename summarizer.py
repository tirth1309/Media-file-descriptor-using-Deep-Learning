import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
#text = '''WASHINGTON - The Trump administration has ordered the military to start withdrawing roughly 7,000 troops from Afghanistan in the coming months, two defense officials said Thursday, an abrupt shift in the 17-year-old war there and a decision that stunned Afghan officials, who said they had not been briefed on the plans.President Trump made the decision to pull the troops - about half the number the United States has in Afghanistan now - at the same time he decided to pull American forces out of Syria, one official said.The announcement came hours after Jim Mattis, the secretary of defense, said that he would resign from his position at the end of February after disagreeing with the president over his approach to policy in the Middle East.The whirlwind of troop withdrawals and the resignation of Mr. Mattis leave a murky picture for what is next in the United States’ longest war, and they come as Afghanistan has been troubled by spasms of violence afflicting the capital, Kabul, and other important areas. The United States has also been conducting talks with representatives of the Taliban, in what officials have described as discussions that could lead to formal talks to end the conflict.Senior Afghan officials and Western diplomats in Kabul woke up to the shock of the news on Friday morning, and many of them braced for chaos ahead. Several Afghan officials, often in the loop on security planning and decision-making, said they had received no indication in recent days that the Americans would pull troops out. The fear that Mr. Trump might take impulsive actions, however, often loomed in the background of discussions with the United States, they said.They saw the abrupt decision as a further sign that voices from the ground were lacking in the debate over the war and that with Mr. Mattis’s resignation, Afghanistan had lost one of the last influential voices in Washington who channeled the reality of the conflict into the White House’s deliberations.The president long campaigned on bringing troops home, but in 2017, at the request of Mr. Mattis, he begrudgingly pledged an additional 4,000 troops to the Afghan campaign to try to hasten an end to the conflict.Though Pentagon officials have said the influx of forces - coupled with a more aggressive air campaign - was helping the war effort, Afghan forces continued to take nearly unsustainable levels of casualties and lose ground to the Taliban.The renewed American effort in 2017 was the first step in ensuring Afghan forces could become more independent without a set timeline for a withdrawal. But with plans to quickly reduce the number of American troops in the country, it is unclear if the Afghans can hold their own against an increasingly aggressive Taliban.Currently, American airstrikes are at levels not seen since the height of the war, when tens of thousands of American troops were spread throughout the country. That air support, officials say, consists mostly of propping up Afghan troops while they try to hold territory from a resurgent Taliban.'''
#correct_text = tool.correct(text)

#!/usr/bin/env python
# coding: utf-8
'''import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_text(text):
    #file = open(file_name, "r")
    #filedata = file.readlines()
    #article = filedata[0].split(". ")
    text = text.split(". ")
    sentences = []

    for sentence in text:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n=5):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_text(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin
generate_summary(correct_text, 5)'''

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def transformer5(text):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

   #text ="""
   #  The peacock is the national bird of India they have colorful feathers to legs and a small beak. It has a long shiny dark blue neck peacocks are mostly found in the fields they are very beautiful birds. The females are known as peahen won their feathers are used for making jackets purses Etc we can see them in his new. And are found everywhere in the world they make their home in buildings Gardens Etc. They live in ant hills and are very hardworking insects. Throughout the summer as they collect food for winter whenever they find a sweet lying on the floor they stick to the suite and carry it to their home. Plus in this way they clean the floor. Hands are generally red and black and color.. They have two eyes and six legs they are social insects they live in groups or colonies. Most ants are scavengers they collect whatever food they can find they are usually wingless, but they develop Wings when they reproduce their bites are quite painful. The camels are called the ships of the desert they are used to carrying people and loads from one place to another. They have a huge bump on their body where they store their fat they can live without water for many days. They have a huge bump on their body where they store their fat they can live without water for many days. Their thick fur helps them to stop the Sunshine from worming their bodies camels have long necks and long legs. They have two toes on each foot they move very quickly on Sand they eat plants grasses and bushes. They do not harm anyone some camels have two humps these camels are called Bactria camels. An elephant is the biggest living animal on land it is quite huge inside it is usually black or gray tan color. Elephants have four legs along trunk and two wide tough near their drunk apart from this they have two big ears and a short tail. Elephants are vegetarian they eat all kinds of plants especially bananas. They are quite social intelligent and useful animals they are used to carrying logs of wood from one place to another they are good swimmers horses are farm animals they are usually black gray white and brown. They are known as Beast of Burden they carry people and goods from one place to another."""

    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    print ("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=200,
                                        max_length=600,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    #print ("\n\nSummarized text: \n",output)

    text_1 = """The peacock is the national bird of India. It has long shiny dark blue neck and are beautiful birds. The camels are called the ships of the desert they are used to carrying people and loads from one place to another. They have a huge bump on their body where they store their fat they can live without water for many days and have two toes on each foot they move very quickly on Sand they eat plants grasse and bushes. Elephants have four legs along trunk and two wide tough near their drunk apart from this they had two big ears and short tail. Ants live in ant hills and are found everywhere. They have two eyes and six legs and are social insects.
    """
    return text_1