from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = stopwords.words("english")
stopwords.append('kind')


class Utils:
    def init_explanation_bank_lemmatizer(self):

        lemmatization_file = open("../data/lemmatization-en.txt")
        self.lemmas = {}
        # saving lemmas
        for line in lemmatization_file:
            self.lemmas[line.split("\t")[1].lower().replace("\n", "")] = line.split("\t")[0].lower()
        return self.lemmas
    def explanation_bank_lemmatize(self, string: str):
        if self.lemmas is None:
            self.init_explanation_bank_lemmatizer()
        temp = []
        for word in string.split(" "):
            if word.lower() in self.lemmas:
                temp.append(self.lemmas[word.lower()])
            else:
                temp.append(word.lower())
        return " ".join(temp)
    @staticmethod
    def clean_fact_for_overlaps(fact_explanation):
        fact = []
        for key in fact_explanation:
            if "FILL" in key or "SKIP" in key or fact_explanation[key] is None:
                continue
            else:
                fact.append(" ".join(str(fact_explanation[key]).split(";")))
        return " ".join(fact)
    def recognize_entities(self, string):
        entities = []
        temp = []
        for word in word_tokenize(string):
            if not word.lower() in stopwords:
                temp.append(word.lower())
        tokenized_string = word_tokenize(" ".join(temp))
        head_index = 0
        word_index = 0
        for word in tokenized_string:
            check_index = len(tokenized_string)
            final_entity = ""
            if word_index > head_index:
                head_index = word_index
            while check_index > head_index:
                if len(wordnet.synsets("_".join(tokenized_string[head_index:check_index]))) > 0:
                    final_entity = self.explanation_bank_lemmatize(" ".join(tokenized_string[head_index:check_index]))
                    entities.append(final_entity)
                    break
                check_index -= 1
            head_index = check_index
            word_index += 1
        return entities


utils = Utils()
utils.init_explanation_bank_lemmatizer()

