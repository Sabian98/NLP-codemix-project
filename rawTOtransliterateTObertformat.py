import requests

f1 = open("data/hinglish_data/SemEval20/Hinglish_test_unlabeled_conll_updated.txt","r")
f2 = open("data/hinglish_data/SemEval20/Hinglish_test_labels.txt","r")                  # test file
f = open("data/hinglish_data/SemEval20/Hinglish_test_labeled_conll_updated_transliterated.txt","a+")

count = 0
word_list = []
hindi_list = []
sentence_id = 0
sentiment_label = 0
transliterated_word_list = []
transliterated_sentence = ""

line = f1.readline()
label = f2.readline()

while(line):
    if(line=="\n"):
        if(len(word_list)):
            count+=1
            if (count%1 == 0):
                transliterated_word_list += [str(sentence_id), "\t"] + word_list[:-1]
                transliterated_word_list += ["\t", str(sentiment_label), "\t", str(sentiment_label), "\n" ]
                dummy_list=[]
                for elem in hindi_list:
                    if len(elem)>8:
                        dummy_list.append(elem[:8])
                    else:
                        dummy_list.append(elem)
                hindi_list = dummy_list
                hindi_string = " ".join(hindi_list)
                URL = "https://www.google.com/inputtools/request?text="+str(hindi_string)+"&ime=transliteration_en_hi&num=5&cp=0&cs=0&ie=utf-8&oe=utf-8&app=jsapi&uv"
                PARAMS = {} 
                r = requests.get(url = URL, params = PARAMS) 
                data = r.json() 
                hindi_translation = data[1][0][1][0]
                hindi_translation_list = hindi_translation.split(" ")

                j=0
                for i in range(len(transliterated_word_list)):
                    if(transliterated_word_list[i]=="qwertyuiopasdfghjklzxcvbnm"):
                        transliterated_word_list[i] = hindi_translation_list[j]
                        j+=1
                transliterated_sentence = "".join(transliterated_word_list)

                f.write(transliterated_sentence)
                transliterated_word_list=[]
            hindi_list =[]
            word_list=[]

    else:
        array = line.split("\t")
        if(array[0]=='meta'):
            # sentence_id = array[1]                                                      # train & val file
            sentence_id = array[1][:-1]                                                 # test file
            sentiment_label_map = {"negative":0, "positive":2, "neutral":1}
            # sentiment_label = sentiment_label_map[array[2][:-1]]                        #train & val file
            sentiment_label = sentiment_label_map[label.split(",")[1][:-1]]             # test file
            label = f2.readline()                                                       # test file
        else:
            if(array[1][:-1]=="Hin"):
                hindi_list.append(array[0])
                word_list += ["qwertyuiopasdfghjklzxcvbnm", " "]
            else:
                word_list += [array[0], " "]
    line = f1.readline()

f.write(transliterated_sentence)