import spacy
import random

greetings = ["hi", "hello", "hey", "morning", "afternoon", "yo"]
greetings_responses = [
    "Hi!",
    "Hello friendly human",
    "Hi there!",
    "Hey!",
]
welcome_responses = [
    "Hi there! I'm a bot and you can say hi to me",
    "Hello!, I'm a greeting bot",
    "Welcome, feel free to say hi to me anytime",
    "Hey human! I'm a bot, but you can say hi to me and I'll do my best to try and answer",
]
questions =["how"]
targets_self = ["bot", "you", "chatbot"]
self_state_responses = [
    "I'm doing fine, thank you",
    "Thanks for asking, I'm doing alright",
    "Right know I'm feeling great! Just a little sleepy",
]
targets_user = ["me", "I"]
request = ["tell", "say", "inspire", "can"]
request_quote = ["quote", "phrase"]
request_song = ["sing", "chant", "perform", "intone"]
quotes = [
    "One of the most bittersweets feelings has to be when you realize you're going to miss a moment while you're still living it. By Alissa N",
    "Go into the arts. I'm not kidding... Practicing an art, not matter how well or badly, is a way to make your soul grow... Do it as well you possibly can. You will get an enormous reward. You will have created something. By Kurt Vonnegut",
    "Never discourage anyone who continually makes progress, no matter how slow",
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

class AI():
    
    def __init__(self):
        self.nlp = spacy.load('model')
#Sending a message to AI
    def message(self, msg):
        if not msg:
            return None
        doc = self.nlp(msg)
        sentences = []
        sents = [s for s in doc.sents]
        for span in sents:
            span_list = list(span)
            sen = " ".join([e.text for e in span_list])
            sentences += [sen]

        docs = self.nlp.pipe(sentences)
        responses = []

        for doc in docs:
                    
            # Dependency label dictionary
            label_dict = {t.dep_ : t for t in doc}
            print(f"label_dict: {label_dict}")
            

            # Revisar si el ROOT es un saludo conocido
            if label_dict["ROOT"].text.lower() in greetings:
                # Responder con un saludo aleatorio
                responses += [random.choice(greetings_responses)]

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

        return ' '.join(responses)


