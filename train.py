#!/usr/bin/env python
# coding: utf-8
"""Using the parser to recognise your own semantics

spaCy's parser component can be trained to predict any type of tree
structure over your input text. You can also predict trees over whole documents
or chat logs, with connections between the sentence-roots used to annotate
discourse structure. In this example, we'll build a message parser for a common
"chat intent": finding local businesses. Our message semantics will have the
following types of relations: ROOT, PLACE, QUALITY, ATTRIBUTE, TIME, LOCATION.

"show me the best hotel in berlin"
('show', 'ROOT', 'show')
('best', 'QUALITY', 'hotel') --> hotel with QUALITY best
('hotel', 'PLACE', 'show') --> show PLACE hotel
('berlin', 'LOCATION', 'hotel') --> hotel with LOCATION berlin

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.lang.en import English



# training data: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
TRAIN_DATA = [
    (
        #0  1
        "hi there",
        {
            "heads": [0, 0],  # index of token head
            "deps": ["ROOT", "-"],
        },
    ),
    (
        #0   1
        "hey you",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0     1
        "hello bot",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0    1
        "good morning",
        {
            "heads": [1, 1],
            "deps": ["QUALITY", "ROOT"],
        },
    ),
    (
        #0
        "hi",
        {
            "heads": [0],
            "deps": ["ROOT"],
        },
    ),
    (
        #0   1   2
        # are -> is -> Situacion actual, estado, sentimientos, etc
        "how are you",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "STATE", "TARGET"],
        },
    ),
    (
        #0   1   2   3
        "how are you feeling",
        {
            "heads": [0, 2, 0, 2],
            "deps": ["ROOT", "STATE", "TARGET", "STATE"],
        },
    ),
    (
        #0   1   2   3
        "how are you doing",
        {
            "heads": [0, 2, 0, 2],
            "deps": ["ROOT", "STATE", "TARGET", "STATE"],
        },
    ),
    (
        #0   1   2
        "how you doing",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "STATE", "TARGET"],
        },
    ),
    (
        #0    1  2  3
        "tell me a quote",
        {
            "heads": [0, 0, 3, 0],
            "deps": ["ROOT", "TARGET", "-","OBJ"],
        },
    ),
    (
        #0   1          2  3
        "say something to me",
        {
            "heads": [0, 0, 3, 0],
            "deps": ["ROOT", "-", "-", "TARGET"],
        },
    ),
    (
        #0    1  2  3
        "tell me a phrase",
        {
            "heads": [0, 0, 3, 0],
            "deps": ["ROOT", "TARGET", "-","OBJ"],
        },
    ),
    (
        #0     1   2         3
        "tell me something motivational",
        {
            "heads": [0, 0, 3, 0],
            "deps": ["ROOT", "TARGET", "-","OBJ"],
        },
    ),
    (
        #0     1   2     3
        "phrase of the day",
        {
            "heads": [0, 2, 3, 0],
            "deps": ["ROOT", "-", "-", "TIME"],
        },
    ),
    (
        #0     1 2   3  4
        "tell me a top quote",
        {
            "heads": [0, 0, 3, 4, 0],
            "deps": ["ROOT", "TARGET", "-", "QUALITY", "OBJ"],
        },
    ),
    (
        #0   1  2       3    4
        "say a famous phrase bot",
        {
            "heads": [0, 2, 3, 0, 0],
            "deps": ["ROOT", "-", "QUALITY", "OBJ", "TARGET"],
        },
    ),
    (
        #0    1     2       3   4   5       6
        "tell me something inspiring",
        {
            "heads": [0, 0, 2, 3, 4, 0, 0],
            "deps": ["ROOT", "TARGET", "-", "-", "-", "QUALITY"],
        },
    ),
    (
        #0       1  2       3 
        "inspire me with something",
        {
            "heads": [0, 0, 3, 0],
            "deps": ["ROOT", "TARGET", "-", "-", "-"],
        },
    ),
    (
        # 0      1 
        "sing something",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "-"],
        },
    ),
    (
        # 0    1  2 
        "sing to me",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "-", "TARGET"],
        },
    ),
    (
        # 0     1  2 
        "peform a song",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "-", "OBJ"],
        },
    ),
    (
        # 0     1  2 
        "intone to me",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "-", "TARGET"],
        },
    ),
    (
        # 0   1  2      3        4  5
        "can you sing something for me",
        {
            "heads": [1, 2, 2, 2, 5, 2],
            "deps": ["STATE", "TARGET", "ROOT", "-", "-", "TARGET"],
        },
    ),
    (
        # 0    1  2 
        "chant to me",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "-", "TARGET"],
        },
    ),
    (
        # 0 
        "goodbye",
        {
            "heads": [0],
            "deps": ["ROOT"],
        },
    ),
    (
        # 0 
        "farewell",
        {
            "heads": [0],
            "deps": ["ROOT"],
        },
    ),
    (
        # 0    1  2  3 
        "so long my friend",
        {
            "heads": [0, 0, 3, 0],
            "deps": ["-", "ROOT", "-","TARGET"],
        },
    ),
    (
        # 0    1  2  3 
        "good night",
        {
            "heads": [1, 1],
            "deps": ["QUALITY", "ROOT"],
        },
    ),
    (
        # 0   1  2  3 
        "have a good night",
        {
            "heads": [2, 2, 3, 3],
            "deps": ["-", "-", "QUALITY", "ROOT"],
        },
    ),
    (
        # 0    1  2  3 
        "bye",
        {
            "heads": [0],
            "deps": ["ROOT"],
        },
    ),
    (
        # 0    1  2   
        "see you later",
        {
            "heads": [1, 2, 2],
            "deps": ["-", "TARGET", "ROOT"],
        },
    ),
    (
        # 0    1  2   
        "see you soon",
        {
            "heads": [1, 2, 2],
            "deps": ["-", "TARGET", "ROOT"],
        },
    ),
    (
        # 0    1  2   3
        "see you next time",
        {
            "heads": [1, 2, 2, 2],
            "deps": ["-", "TARGET", "ROOT", "TIME"],
        },
    ),
    (
        # 0    1  2  
        "see you tomorrow",
        {
            "heads": [1, 2, 2],
            "deps": ["-", "TARGET", "ROOT"],
        },
    ),

    # (
    #     #0    1 2    3    4     5
    #     "find a cafe with great wifi",
    #     {
    #         "heads": [0, 2, 0, 5, 5, 2],  # index of token head
    #         "deps": ["ROOT", "-", "PLACE", "-", "QUALITY", "ATTRIBUTE"],
    #     },
    # ),
    # (
    #     #0    1 2     3    4   5
    #     "find a hotel near the beach",
    #     {
    #         "heads": [0, 2, 0, 5, 5, 2],
    #         "deps": ["ROOT", "-", "PLACE", "QUALITY", "-", "ATTRIBUTE"],
    #     },
    # ),
    # (
    #     "find me the closest gym that's open late",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 6, 4, 6, 6],
    #         "deps": [
    #             "ROOT",
    #             "-",
    #             "-",
    #             "QUALITY",
    #             "PLACE",
    #             "-",
    #             "-",
    #             "ATTRIBUTE",
    #             "TIME",
    #         ],
    #     },
    # ),
    # (
    #     "show me the cheapest store that sells flowers",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 4, 4, 4],  # attach "flowers" to store!
    #         "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "-", "PRODUCT"],
    #     },
    # ),
    # (
    #     "find a nice restaurant in london",
    #     {
    #         "heads": [0, 3, 3, 0, 3, 3],
    #         "deps": ["ROOT", "-", "QUALITY", "PLACE", "-", "LOCATION"],
    #     },
    # ),
    # (
    #     "show me the coolest hostel in berlin",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 4, 4],
    #         "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "LOCATION"],
    #     },
    # ),
    # (
    #     "find a good italian restaurant near work",
    #     {
    #         "heads": [0, 4, 4, 4, 0, 4, 5],
    #         "deps": [
    #             "ROOT",
    #             "-",
    #             "QUALITY",
    #             "ATTRIBUTE",
    #             "PLACE",
    #             "ATTRIBUTE",
    #             "LOCATION",
    #         ],
    #     },
    # ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    nlp_en = English()
    sentencizer = nlp_en.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec", "sentencizer"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            # print("Losses", losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


def test_model(nlp):
    texts = [
        "hello bot",
        "hello there",
        "hi good morning",
        "hey bot",
        "Hello",
        "HI THERE",

        "how are you doing bot",
        "how do you do",
        "how do you feel",

        "how is the weather",
        "how did the cat get there",
        "how can I find the restroom",

        "hi my name is Steve",

        "hi how are you. sing something",
        "sing me a song all aloud",
        "sing a lullaby",

        "tell a famous quote",
        "say famous phrase",
        "inspire me with a quote",

        "goodbye friend",
        "bye bye",
        "have a good night",
        "see you soon",
        
        # "find a hotel with good wifi",
        # "find me the cheapest gym near work",
        # "show me the best hotel in berlin",
    ]
    greetings = ["hi", "hello", "hey", "morning", "afternoon", "yo"]
    greeting_responses = [
        "Hi!",
        "Hello friendly human.",
        "Hi there!",
        "Hey!",
    ]
    welcome_responses = [
        "Hi there! I'm a bot and you can say hi to me.",
        "Hello! I'm a greeting bot.",
        "Welcome, feel free to say hi to me anytime.",
        "Hey human! I'm a bot, but you can say hi to me and I'll do my best to try and answer.",
    ]
    questions = ["how"]
    targets_self = ["bot", "you", "chatbot"]
    self_state_responses = [
        "I'm doing fine thank you.",
        "Thanks for asking, I'm doing alright.",
        "Right now I'm feeling great! Just a little sleepy.",
    ]
    targets_user = ["me", "I"]
    request = ["tell", "say", "inspire", "can"]
    request_quote = ["quote", "phrase"]
    request_song = ["sing", "chant", "perform", "intone"]
    quotes = [
        "One of the most bittersweets feelings has to be when you realize you're going to miss a moment while you're still living it. By Alissa N",
        "Go into the arts. I'm not kidding... Practicing an art, not matter how well or badly, is a way to make your soul grow... Do it as well you possibly can. You will get an enormous reward. You will have created something. By Kurt Vonnegut",
        "Never discourage anyone who continually make progress, no matter how slow",
        "I think the saddest people always try their hardest to make people happy. Because they know what it's like to feel absolutely worthless and they don't want anybody else to feel like that. Robbin Williams",
        "I spend way too much of my life with the mentality *I can't wait for -this- to be over*. u/500daystolive ",
        "I wish I lived in the moment more and quit constanly hoping for better things",
    ]
    songs = [
       "Jaja ding dong  (ding dong) My love for you is growing wide and long",
       "POTATO MAAAAN, volcanic potato maaaaan",
       "Jaja ding dong (ding dong) I swell and burst when I see what we become",
       "Jaja ding dong (ding dong) Come, come my baby, we can get love on",
       "Jaja ding dong (ding dong) When I see you, I feel like ding-ding dong",
    ]
    goodbyes = [
        "bye",
        "night",
        "farewell",
        "later",
        "goodbye",
        "soon",
        "tomorrow",
    ]
    goodbyes_responses = [
        "See you later!",
        "Goodbye!",
        "Farewell!",
        "See you soon!",
        "Have a nice day!",
    ]

    docs = nlp.pipe(texts)
    sentences = []
    for doc in docs:
        sents = [s for s in doc.sents]
        for span in sents:
            span_list = list(span)
            sen = " ".join([e.text for e in span_list])
            sentences += [sen]


    docs = nlp.pipe(sentences)
    
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != "-"])
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

        responses = []
        # sentences = [s for s in doc.sents]
        # print(sentences)
        
        # for sen in sentences:
                
        # Dependency label dictionary
        label_dict = {t.dep_ : t for t in doc}
        print(f"label_dict: {label_dict}")
        

        # Revisar si el ROOT es un saludo conocido
        if label_dict["ROOT"].text.lower() in greetings:
            # Responder con un saludo aleatorio
            responses += [random.choice(greeting_responses)]

        elif label_dict["ROOT"].text.lower() in request:
            if "OBJ" in label_dict and label_dict["OBJ"].text.lower() in request_quote:
                responses += [random.choice(quotes)]
            else:
                responses += ["I'm sorry, I'm not sure how to answer that."]

        elif label_dict["ROOT"].text.lower() in request_song:
                responses += [random.choice(songs)]

        elif label_dict["ROOT"].text.lower() in goodbyes:
                responses += [random.choice(goodbyes_responses)]

        elif label_dict["ROOT"].text.lower() in questions:
            # Es una pregunta
            # Responder si preguntan cómo estamos
            if "STATE" in label_dict and "TARGET" in label_dict and label_dict["TARGET"].text.lower() in targets_self:
                responses += [random.choice(self_state_responses)]
            else:
                responses += ["I'm sorry, I'm not sure how to answer that."]
        
        else:
            # Responder con un mensaje de bienvenida aleatorio
            responses += [random.choice(welcome_responses)]

        print("Response:")
        print(responses)

        print("\n")
        print("-" * 20)
        print("\n")


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # find a hotel with good wifi
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('hotel', 'PLACE', 'find'),
    #   ('good', 'QUALITY', 'wifi'),
    #   ('wifi', 'ATTRIBUTE', 'hotel')
    # ]
    # find me the cheapest gym near work
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('cheapest', 'QUALITY', 'gym'),
    #   ('gym', 'PLACE', 'find'),
    #   ('near', 'ATTRIBUTE', 'gym'),
    #   ('work', 'LOCATION', 'near')
    # ]
    # show me the best hotel in berlin
    # [
    #   ('show', 'ROOT', 'show'),
    #   ('best', 'QUALITY', 'hotel'),
    #   ('hotel', 'PLACE', 'show'),
    #   ('berlin', 'LOCATION', 'hotel')
    # ]