from portion import closedopen
from ufal.morphodita import Forms, TokenRanges, Tokenizer

class MorphoDiTaTokenizer:
    def __init__(self, lang:str ="cs"):
        lang = lang.lower()
        assert lang in ["cs", "en", "generic", "vertical"]
        if lang == "cs":
            self.tokenizer = Tokenizer.newCzechTokenizer()
        elif lang == "en":
            self.tokenizer = Tokenizer.newCzechTokenizer()
        elif lang == "generic":
            self.tokenizer = Tokenizer.newGenericTokenizer()
        elif lang == "vertical":
            self.tokenizer = Tokenizer.newVerticalTokenizer()
        self.forms = Forms()
        self.tokens = TokenRanges()


    def tokenizeSentences(self, text: str, spans: bool=False):
        self.tokenizer.setText(text)
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            first = self.tokens[0].start
            last = self.tokens[-1].start + self.tokens[-1].length
            if spans:
                yield text[first:last], closedopen(first, last)
            else:
                yield text[first:last]

    def tokenizeWords(self, text: str, spans: bool=False):
        self.tokenizer.setText(text)
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            for form, token in zip(self.forms, self.tokens):
                if spans:
                    first = token.start
                    last = token.start + token.length
                    yield form, closedopen(first, last)
                else:
                    yield form

def detokenize(txt):
    # TFIDF to transformer (de)tokenization fixes
    txt = txt.replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
    txt = txt.replace("`` ", '"').replace(" ''", '"').replace(" '", "'")
    txt = txt.replace("-LRB- ", "(").replace("-RRB-", ")")
    return txt

def detokenize2(txt):
    # updated detokenize, most models are not trained with this...
    txt = txt.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
    txt = txt.replace("`` ", '"').replace(" ''", '"').replace(" '", "'")
    txt = txt.replace("-LRB- ", "(").replace("-RRB-", ")")
    txt = txt.replace("( ", "(").replace(" )", ")")
    return txt