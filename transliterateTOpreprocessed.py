import re
from nltk.tokenize import word_tokenize

f1 = open("data/hinglish_data/SemEval20/Hinglish_train_14k_split_conll_transliterated.txt","r")
f = open("data/hinglish_data/SemEval20/Hinglish_train_14k_split_conll_transliterated_preprocessedv1.txt","w")

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

line = f1.readline()

while(line):
    deemojify = deEmojify(line.split("\t")[1])

    tokenized = word_tokenize(deemojify)
    detokenized = ""
    i = 0
    while (i < len(tokenized)):
        if tokenized[i] == "@":
            i += 2
            continue
        if tokenized[i] == "https":
            break
        detokenized += tokenized[i]
        detokenized += " "
        i += 1
    if detokenized == "":
        line = f1.readline()
        continue

    f.write(line.split("\t")[0])
    f.writelines("\t")
    f.write(detokenized)
    f.writelines("\t")
    f.write(line.split("\t")[2])
    f.writelines("\t")
    f.write(line.split("\t")[3])
    line = f1.readline()
