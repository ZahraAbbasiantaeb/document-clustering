# document-clustering
<p>In this project representation of each document is obtained in four different ways:</p>

<p>1) averaging word2vec representations of document's words. </p>
<p>2) TF-IDF weighted averaging of word2vec representations of document's words. </p>
<p>3) Doc2vec Representation. </p>
<p>4) Reducing dimensionality of doc-word matrix, which is filled with TF, by PCA. </p>

Then documents are clustered using Kmeans algorithm. Accomplished using Scikit, NLTK, and Gensim Libraries in python.
