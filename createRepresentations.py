import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import math
from sklearn.decomposition import TruncatedSVD


dimension = 300
train_pickle = "data_pickle"

train_path = 'train-test-data/HAM-Train-Test.txt'

word2vec_path = 'model.bin'
doc2vec_path = 'doc2vec.bin'


# call this func to set up the DSs
def readCorpus(path, train_pickle):

    with open(path) as f:

        lines = f.readlines()

        sentences = []

        label = []

        doc_id = 1

        TF_IDF = {}

    for line in lines:

        line = line.rstrip()

        token = line.split('@@@@@@@@@@')

        label.append(token[0])

        text = token[1]

        tokens = text.split(' ')

        tokens.remove('')

        for word in tokens:

            if (word in TF_IDF):

                if doc_id in TF_IDF[word]:

                    tmp = TF_IDF[word][doc_id]
                    TF_IDF[word][doc_id] = (tmp + 1)

                else:
                    TF_IDF[word][doc_id] = 1

            else:
                TF_IDF[word] = {}
                TF_IDF[word][doc_id] = 1

        sentences.append(tokens)

        doc_id += 1

    print('TF created')

    doc_mat = (createDocVecMatrix(TF_IDF, (doc_id - 1)))

    print('Doc_Mat created')

    word2vecModel(sentences, word2vec_path)

    print('Word_vec created')

    doc2vec(sentences, doc2vec_path)

    print('doc_vec created')


    with open(train_pickle, "wb") as f:

        pickle.dump((sentences, label, TF_IDF, (doc_id - 1), doc_mat), f)

    return


# creates doc-word matrix and reduces its dimension by SVD
def createDocVecMatrix(TF_IDF, documents_count):

    words_count = len(TF_IDF)

    word_doc = np.zeros((words_count, documents_count))

    index = 0

    for word in TF_IDF:

        for doc in TF_IDF[word]:

            word_doc[index, doc - 1] = TF_IDF[word][doc]

        index += 1

    svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)

    svd.fit(word_doc)

    print(np.shape(svd.components_))

    with open('SVD_mat_doc', "wb") as f:

        pickle.dump(svd.components_, f)

    return


# creates word-to-vec model of given sentences
def word2vecModel(sentences, word2vec_path):

    # train model
    model = Word2Vec(sentences, min_count=1, size=dimension, sg=1)

    # save model
    model.save(word2vec_path)

    return


# loads Models
def loadModel():

    m_word2vec = Word2Vec.load(word2vec_path)

    m_doc2vec = Doc2Vec.load(doc2vec_path)

    return m_word2vec, m_doc2vec


# creates doc-to-vec model
def doc2vec(sentences, doc2vec_path):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]

    model = Doc2Vec(documents, vector_size=dimension, window=2, min_count=1, workers=4, dm=0)

    model.save(doc2vec_path)

    return


# returns TF-IDF of given word in given doc
def getTF_IDF(word, doc, TF_IDF):

    tmp = 0

    if word in TF_IDF:

        if doc in TF_IDF[word]:
            tmp = TF_IDF[word][doc] * math.log(documents_count / len(TF_IDF[word]))

    return tmp


# builds representation of given sentence by averaging rep of its words
def word2vecMeanRep(sentence):

    if (len(sentence) > 0):
        tmp = m_word2vec.wv[sentence[0]]

    for i in range(1, len(sentence)):
        tmp = np.add(tmp, m_word2vec.wv[sentence[i]])

    return tmp / len(sentence)


# builds weighted representation of given sentence, TF-IDF is used as weight of each word
def word2vecTF_IDFRep(sentence, doc_ID, TF_IDF):

    weight = 0

    if (len(sentence) > 0):

        weight += getTF_IDF(sentence[0], doc_ID, TF_IDF)

        tmp = getTF_IDF(sentence[0], doc_ID, TF_IDF) * m_word2vec.wv[sentence[0]]


    for i in range(1, len(sentence)):
        weight += getTF_IDF(sentence[i], doc_ID, TF_IDF)

        tmp = np.add(tmp, getTF_IDF(sentence[i], doc_ID, TF_IDF) * m_word2vec.wv[sentence[i]])

    return tmp / weight


# builds doc-to-vec representation of given sentence
def doc2vecRep(sentence, model):

    return model.infer_vector(sentence)


# buils doc-to-vec representation of given sentence by using word-doc matrix
def doc2wordRep(doc):

    return doc_mat[:, doc]


# creates SVD representation of given sentences
def createSVDRep(sentences, index):

    rep = []
    id = index

    for sentence in sentences:
        rep.append(doc2wordRep(id))
        id += 1

    return rep


# creates representation of given sentences by averaging reps of its tokens
def createWord2vecMeanRep(sentences):

    word2vecMeanReps = []

    for sentence in sentences:
        word2vecMeanReps.append(word2vecMeanRep(sentence))

    return word2vecMeanReps


def createWord2VecTF_IDFRep(sentences, index):

    rep = []
    id = index

    for sentence in sentences:
        rep.append(word2vecTF_IDFRep(sentence, id, TF_IDF))
        id += 1

    return rep


# creates doc-to-vec representation of given sentences by using doc-to-vec model
def createDoc2VecRep(sentences):

    doc2vecReps = []

    for sentence in sentences:
        doc2vecReps.append(doc2vecRep(sentence, m_doc2vec))

    return doc2vecReps


# readCorpus(train_path, train_pickle)

with open(train_pickle, "rb") as f:
    sentences, label, TF_IDF, documents_count, _ = pickle.load(f)


with open('SVD_mat_doc', "rb") as f:
    doc_mat = pickle.load(f)

train_data = []
train_label = []
test_data = []
test_label = []

for i in range(0, 7740):

    train_data.append(sentences[i])
    train_label.append(label[i])

for i in range(7740, 8600):

    test_data.append(sentences[i])
    test_label.append(label[i])


m_word2vec, m_doc2vec = loadModel()

# print(np.shape(doc_mat))
#
# word2vecMeanReps_train = createWord2vecMeanRep(train_data)
#
# doc2vecReps_train = createDoc2VecRep(train_data)
#
# word2vecTFRep_train = createWord2VecTF_IDFRep(train_data, 1)
#
# svd_rep_train = createSVDRep(train_data, 0)
#
# word2vecMeanReps_test = createWord2vecMeanRep(test_data)
#
# doc2vecReps_test = createDoc2VecRep(test_data)
#
# word2vecTFRep_test = createWord2VecTF_IDFRep(test_data, 7740)
#
# svd_rep_test = createSVDRep(test_data, 7739)
#
# with open('representations', "wb") as f:
#
#     pickle.dump((word2vecMeanReps_train, doc2vecReps_train, word2vecTFRep_train, svd_rep_train,
#                  word2vecMeanReps_test, doc2vecReps_test, word2vecTFRep_test, svd_rep_test), f)


with open('representations', "rb") as f:
    word2vecMeanReps_train, doc2vecReps_train, word2vecTFRep_train, svd_rep_train, word2vecMeanReps_test, doc2vecReps_test, word2vecTFRep_test, svd_rep_test = pickle.load(f)

