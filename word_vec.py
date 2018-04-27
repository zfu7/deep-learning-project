import numpy as np
import torchwordemb

def loadGloveModel(gloveFile):
    # print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

gloveFile = '../glove/glove.twitter.27B.50d.txt'
model = loadGloveModel(gloveFile)
print(model['apple'])
print(len(model))

# vocab, vec = torchwordemb.load_word2vec_text(gloveFile)
# print(vec.size())
# print(vec[ vec.vocab["apple"] ] )
