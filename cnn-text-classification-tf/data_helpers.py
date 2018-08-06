import re
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("data") if isfile(join("data", f))]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


for mf in onlyfiles:
    if not mf.startswith("."):
        dataset = list(open("data/"+mf, "r").readlines())
        dataset = [s.strip() for s in dataset[1:]]
        dataset = [clean_str(sent) for sent in dataset]
        x_text = [s.split(" ") for s in dataset]
        dataset = [s[0:] for s in x_text]
        dataset_clean = []
        for s in dataset:
            dataset_clean.append( [' '.join(ww for ww in s)])
         
        for s in dataset_clean:
            for w in s:
                f = open('corpus/'+mf,'a').write(w + '\n')