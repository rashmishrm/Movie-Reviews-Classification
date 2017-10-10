import os
import re
import copy
import random
import numpy as np
import scipy as sp
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from stemming.porter2 import stem


def getPositiveNegativeWords():
    with open("negative-words.txt", "r") as fh:
        negative_words = fh.readlines()
    with open("positive-words.txt", "r") as fh:
        positive_words = fh.readlines()

    pos_list = []
    for s in positive_words:
        s = s.replace("\n", "")
        pos_list.append(s)

    pos_list1 = [stem(d) for d in pos_list]
    pos_list.extend(pos_list1)

    neg_list = []
    for s in negative_words:
        s = s.replace("\n", "")
        neg_list.append(s)

    neg_list1 = [stem(d) for d in neg_list]
    neg_list.extend(neg_list1)

    return pos_list, neg_list


def getDocs(filename="train.dat"):
    # open docs file and read its lines
    with open(filename, "r") as fh:
        train_lines = fh.readlines()

    pos_list, neg_list = getPositiveNegativeWords()

    if (filename == "train.dat"):
        train_labels = [l.split()[0] for l in train_lines]
        train_doc = [re.sub(r'[^\w]', ' ', l[2:].lower()).split() for l in train_lines]
        train_rev = filterLen(train_doc, 4)
        train_documents = stemDoc(train_rev)
        for t in train_documents:
            pos_count = 0
            neg_count = 0
            new_term = "ann-neutral"

            for d in t:
                if (d in pos_list):
                    pos_count += 1
                elif (d in neg_list):
                    neg_count += 1
            if (neg_count > pos_count):
                new_term = "ann-neg"
            elif (neg_count < pos_count):
                new_term = "ann-pos"

            new_list = kmers(t)
            t.append(new_term)

            t.extend(new_list)


    return train_labels, train_documents, pos_list, neg_list


def splitTrainData(train_docs, train_labels):
    train_docs1 = train_docs[0:len(train_docs) / 2]
    train_labels1 = train_labels[0:len(train_labels) / 2]

    test_docs = train_docs[len(train_docs) / 2:]
    test_labels = train_labels[len(train_labels) / 2:]

    return train_docs1, train_labels1, test_docs, test_labels


def getTestDocs(filename="test.dat"):
    # open docs file and read its lines
    pos_list, neg_list = getPositiveNegativeWords()
    with open(filename, "r") as fh:
        test_lines = fh.readlines()

    test_doc = [re.sub(r'[^\w]', ' ', l[2:].lower()).split() for l in test_lines]
    test_rev = filterLen(test_doc, 4)
    test_documents = stemDoc(test_rev)

    for t in test_documents:
        pos_count = 0
        neg_count = 0
        new_term = "ann-neutral"
        for d in t:
            if (d in pos_list):
                pos_count += 1
            elif (d in neg_list):
                neg_count += 1
        if (neg_count > pos_count):
            new_term = "ann-neg"
        elif (neg_count < pos_count):
            new_term = "ann-pos"
        new_list = kmers(t)
        t.append(new_term)
        t.extend(new_list)


    return test_documents


def club_words(input, n=2):
    for i in xrange(len(input) - (n - 1)):
        yield input[i:i + n]


def kmers(input):
    changed = []
    for first, second in club_words(input, 2):
        st = first + " " + second
        changed.append(st)
    return changed


def filterLen(docs, minlen):
    """ filter out terms that are too short.
    docs is a list of lists, each inner list is a document represented as a
    list of words minlen is the minimum length of the word to keep
	filters stop words too]
    """
    with open("stop-words.txt", "r") as fh:
        stop_words = fh.readlines()

    st_list = []
    for s in stop_words:
        s = s.replace("\n", "")
        st_list.append(s)

    return [[t for t in d if (len(t) >= minlen and t not in st_list)] for d in docs]


def stemDoc(docs):
    """ automatically removes suffixes (and in some cases prefixes) in order to
    find the root word or stem of a given word
    """

    return [[stem(t) for t in d] for d in docs]


def build_matrix(docs, pos_list, neg_list, weight=2):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """

    nrows = len(docs)
    idx = {}
    tid = 2
    nnz = 0
    idx["ann-neg"] = 0
    idx["ann-pos"] = 1

    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)

    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows + 1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        # print cnt
        keys = list(k for k, _ in cnt.most_common())
        # print keys
        l = len(keys)
        for j, k in enumerate(keys):
            ind[j + n] = idx[k]
            val[j + n] = cnt[k]
            if (k in pos_list or k in neg_list):
                val[j + n] = val[j + n] * weight
            if (k == 'ann-neg' or k == 'ann-pos'):
                val[j + n] = val[j + n] * 15
        ptr[i + 1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()

    return mat, idx


def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        if i > 1:
            df[i] += 1
    # inverse document frequency
    for k, v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(2, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat


def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i + 1]):
            rsum += val[j] ** 2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0 / np.sqrt(rsum)
        for j in range(ptr[i], ptr[i + 1]):
            val[j] *= rsum

    if copy is True:
        return mat


def getNeighborsWhole(distances, train_labels, train_no, test_no, k):
    test_labels = []
    minimum=train_no
    maximum=train_no + test_no
    for index in range(minimum, maximum):
        #fetch all similarity for test vector denoted by index
        similarity = distances[index, :train_no].toarray().tolist()[0]

        #combining simlarity and labels
        zipped_sim_labels = zip(similarity, train_labels, range(len(train_labels)))

        #sorting labels
        sorted_zipped_sim_labels = sorted(zipped_sim_labels, key=lambda (val, k, l): val, reverse=True)
        tmp = 0

        for j in range(k):
            if sorted_zipped_sim_labels[j][0] > 0:
                tmp += int(sorted_zipped_sim_labels[j][1])
        if tmp == 0:
            # get nearest one in case of tie
            tmp = sorted_zipped_sim_labels[0][1]
        if tmp > 0:
            test_labels.append('+1')
        else:
            test_labels.append('-1')

    return test_labels


def evaluations():
    # 1) get training documents
    print "1. Reading document and preprocessing"
    train_labels, train_documents, pos_list, neg_list = getDocs()

    # 2) split training into test and train data
    print "2. Splitting into train and test"

    train_documents, train_labels, test_documents, test_labels = splitTrainData(train_documents, train_labels)
    train_doc_no = len(train_documents)

    train_documents.extend(test_documents)
    print "3. Building CSR Matrix"

    # 3) build csr matrix using negative and positive list
    csr_mat, word_dict = build_matrix(train_documents, pos_list, neg_list)
    # 4 ) idf matrix

    print "4. Building IDF Matrix"

    mat1 = csr_idf(csr_mat, copy=True)

    # 5) Normalize matrix
    print "5. Normalized"

    mat = csr_l2normalize(mat1, copy=True)

    # 6) Find cosine Similarity
    print "6 Calculating cosine similarity"
    distances = cosine_similarity(mat, dense_output=False)

    # 7)
    k_list = [17, 33, 150, 349, 399, 449]
    #k_list = [2]
    print "7 For Different K performing N neighbor"

    for f in k_list:

        print "calculating for K=" + str(f)

        clspr = getNeighborsWhole(distances, train_labels, train_doc_no, len(test_documents), f)

        acc = 0.0
        for i in range(len(test_labels)):
            if test_labels[i] == clspr[i]:
                acc += 1
            acc /= len(test_labels)
        print "K= "+str(f) + "---> Accuracy: " + str(acc)


def test_phase():
    print "1. Reading document and preprocessing"

    train_labels, train_documents, pos_list, neg_list = getDocs()
    train_docs_no=len(train_documents)
    print "2. Reading test document"

    test_documents = getTestDocs("test.dat")
    train_documents.extend(test_documents)
    print "3. Creating CSR matrix"

    csr_mat, word_dict = build_matrix(train_documents, pos_list, neg_list)
    print "4. Creating IDF matrix"

    mat1 = csr_idf(csr_mat, copy=True)
    print "5. Creating normalized matrix"

    mat = csr_l2normalize(mat1, copy=True)
    print "6. Creating cosine similarity"

    distances = cosine_similarity(mat, dense_output=False)
    print "6. Calculating best value k="

    filename = 'format.dat'
    output_file = open(filename, 'w')
    clspr = getNeighborsWhole(distances, train_labels, train_docs_no, len(test_documents), 449)

    for e in clspr:
        output_file.write(e + "\n")

    output_file.close()

    return "Done"


# Call below method while evaluating

print "Evaluating models...."
# uncomment for evaluations
evaluations()

print "Testing on test.dat file...."

# For actual test phase.
test_phase()
