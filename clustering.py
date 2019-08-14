from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from createRepresentations import train_label, test_label, word2vecMeanReps_train, doc2vecReps_train, \
    word2vecTFRep_train, svd_rep_train, word2vecMeanReps_test, doc2vecReps_test, word2vecTFRep_test, \
    svd_rep_test, test_label
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics

cluster_count = 5

# this func is used for clustering the train data and assigning the test data to centroids, and evaluating
def cluster(data_train, label_train, data_test, label_test):

    kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(data_train)

    pred = (kmeans.labels_)

    new_label = []

    assign = {}

    index = 0

    for word in label_train:

        if word in assign:

            if pred[index] in assign[word]:
                tmp = assign[word][pred[index]] +1
                assign[word][pred[index]] = tmp

            else:
                assign[word][pred[index]] = 1
        else:
            assign[word] = {}
            assign[word][pred[index]] = 1

        index +=1

    for cat in assign:
        tmp = 0
        for word in assign[cat]:
            tmp += assign[cat][word]
        for word in assign[cat]:
            assign[cat][word] = assign[cat][word]/tmp
    #
    new_index={}

    # print(assign)
    #
    for cat in assign:

        val = 0
        tmp_index = 0

        for index in assign[cat]:

            if(assign[cat][index] >= val):

                val = assign[cat][index]
                tmp_index = index

        new_index[cat] = tmp_index



    pred_test = kmeans.predict(data_test)

    test_new_label = []

    for word in label_test:
        test_new_label.append(new_index[word])

    for word in label_train:
        new_label.append(new_index[word])

    print('train NMI:')
    print(normalized_mutual_info_score(new_label, pred))

    print('Test NMI:')
    print(normalized_mutual_info_score(test_new_label, pred_test))

    print('Train V-measure: ')
    print(metrics.v_measure_score(new_label, pred))

    print('Test V-measure: ')
    print(metrics.v_measure_score(test_new_label, pred_test))

    print('Train Accuracy: ')
    print(metrics.accuracy_score(new_label, pred))

    print('Test Accuracy: ')
    print(metrics.accuracy_score(test_new_label, pred_test))

    print('Train report:')
    print(metrics.classification_report(new_label, pred))

    print('Test report:')
    print(metrics.classification_report(test_new_label, pred_test))


# cluster(svd_rep_train, train_label, svd_rep_test, test_label)
cluster(word2vecMeanReps_train, train_label, word2vecMeanReps_test, test_label)
cluster(word2vecTFRep_train, train_label, word2vecTFRep_test, test_label)
cluster(doc2vecReps_train, train_label, doc2vecReps_test, test_label)
