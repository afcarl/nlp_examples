from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy
import numpy as np

# Load all categories from the training set
categories = None

#remove  headers, signatures, and quoting to avoid overfitting
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

categories = data_train.target_names    # for case categories == None

vectorizer = TfidfVectorizer(
    #tokenizer=tokenize, #provide a tokenizer if you want to
    sublinear_tf=True,
    use_idf=True,
    max_df=0.5,
    min_df = 10, #words must appear at least min_df times
    stop_words='english')

X_train = vectorizer.fit_transform(data_train.data)

#Step 1: Compute SVD
svd_output = scipy.sparse.linalg.svds(X_train, k=200, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True)
U,d,Vt = svd_output

#warning: the most significant singular values/vectors are at k = 199

word_names = vectorizer.get_feature_names()

class WordMappings:
    def item_to_index(self, item):
        #item is a word like jesus
        return vectorizer.vocabulary_[item]

    def index_to_item(self, index):
        global word_names
        return word_names[index]

word_mappings=WordMappings()
word_mappings.item_to_index('jesus')
word_mappings.index_to_item(5329)


class DocumentMappings:
    def item_to_index(self, item):
        return item

    def index_to_item(self, index):
        global categories
        cat_index = data_train.target[index]
        filename = data_train.filenames[index]
        return (categories[cat_index], filename)

doc_mappings = DocumentMappings()

def find_related_items(data_matrix, mappings, query):
    query_index = mappings.item_to_index(query)
    query = data_matrix[query_index, :]
    if type(data_matrix)==np.ndarray:
        dot_products = np.dot(query, data_matrix.T)
    else:
        dot_products = np.dot(query, data_matrix.T).toarray()[0,:]
    return [(mappings.index_to_item(i), dot_products[i]) for i in np.argsort(dot_products)[::-1]] # descending

def normalize_rows(matrix):
    #axis = 1 should normalize by row
    return normalize(matrix, norm='l2', axis=1)

normalized_Vt_T= normalize_rows(Vt.T)


def multiply_by_S(mat):
    global d
    Sinv = np.diag(d/(d + 1.0))
    return mat.dot(Sinv)


def find_and_show_results(method, mappings, mat, query):
    print '--------------------------------------------------'
    related_words = find_related_items(mat, mappings, query)
    for result in related_words[0:10]:
        print method, result

#Original matrix with normalized rows
X_train_norm = normalize_rows(X_train.T)
find_and_show_results('orig-norm-words', word_mappings, X_train_norm, 'jesus')

#Matrix after SVD and re-normalizing
normalized_Vt_T_mult_by_S = normalize_rows(multiply_by_S(Vt.T))
find_and_show_results('svd-norm-words', word_mappings, normalized_Vt_T_mult_by_S, 'jesus')

find_and_show_results('orig-doc', doc_mappings, X_train, 10)

U_mult_by_S = multiply_by_S(U)
find_and_show_results('svd-doc', doc_mappings, U_mult_by_S, 10)

norm_U_mult_by_S = normalize_rows(U_mult_by_S)

find_and_show_results('svd-doc-norm', doc_mappings, norm_U_mult_by_S, 10)


def print_clusters(matrix, mappings, num_clusters):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(matrix)
    cluster_labels = list(kmeans.predict(matrix))
    print kmeans.score(matrix)
    #cluster_scores = list(kmeans.score(matrix))
    from collections import defaultdict
    clusters = defaultdict(list)
    for (item_index, cluster_id) in zip(range(0, len(cluster_labels)), cluster_labels):
        clusters[cluster_id].append((item_index, 0.0))

    for (cluster_id, item_indices) in clusters.iteritems():
        sorted_item_indices = sorted(item_indices, key=lambda(index,score):score)
        print 'cluster', cluster_id
        print '----------------'
        for (item_index,score) in sorted_item_indices[0:20]:
            print mappings.index_to_item(item_index)

print_clusters(norm_U_mult_by_S, doc_mappings, 20)
print_clusters(normalized_Vt_T_mult_by_S, word_mappings, 20)


def word_selection(matrix, num_clusters, num_selected_words):
    from sklearn.cluster import KMeans
    from sklearn.feature_selection import SelectKBest, chi2
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(matrix)
    cluster_labels = list(kmeans.predict(matrix))
    ch2 = SelectKBest(chi2, k=num_selected_words)
    ch2.fit(X_train, cluster_labels)
    selected_words = [vectorizer.get_feature_names()[i] for i in ch2.get_support(indices=True)]
    print selected_words

word_selection(norm_U_mult_by_S, 40, 100)

"""
Output you should get:

--------------------------------------------------
orig-norm-words (u'jesus', 1.0000000000000002)
orig-norm-words (u'christ', 0.38325027850611992)
orig-norm-words (u'god', 0.30135440304020045)
orig-norm-words (u'christians', 0.25603170579354101)
orig-norm-words (u'sin', 0.25564600158669282)
orig-norm-words (u'teachings', 0.22645675657558478)
orig-norm-words (u'bible', 0.21848941851003195)
orig-norm-words (u'apostles', 0.20748800158208561)
orig-norm-words (u'kingdom', 0.20346074699459432)
orig-norm-words (u'disciples', 0.20149152699088543)
--------------------------------------------------
svd-norm-words (u'jesus', 1.0000000000000002)
svd-norm-words (u'christ', 0.86009590097280519)
svd-norm-words (u'disciples', 0.79547702474809434)
svd-norm-words (u'sins', 0.7670644944157452)
svd-norm-words (u'apostles', 0.7500950717108521)
svd-norm-words (u'kingdom', 0.74135533850601321)
svd-norm-words (u'messiah', 0.72948008455743663)
svd-norm-words (u'preached', 0.72003283192442791)
svd-norm-words (u'sin', 0.70366203172709318)
svd-norm-words (u'apostle', 0.69997779338040655)
--------------------------------------------------
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104091'), 1.0)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104933'), 0.25451159165587312)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/103156'), 0.21650250034425009)
orig-doc (('rec.autos', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.autos/101559'), 0.18531709027473151)
orig-doc (('rec.autos', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.autos/103292'), 0.17072745749547422)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104478'), 0.15905304172850643)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104655'), 0.14735828729102751)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104593'), 0.13734798161738124)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104410'), 0.13459105455924186)
orig-doc (('comp.sys.mac.hardware', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/comp.sys.mac.hardware/50546'), 0.1318787044403357)
--------------------------------------------------
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104091'), 0.0080242731141985831)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104338'), 0.0052533193653471122)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104512'), 0.0049817987323623738)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104636'), 0.0049667650320133171)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104667'), 0.0048383548006953291)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104668'), 0.0048383548006953291)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104478'), 0.0045451341603956327)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/105131'), 0.0045142197613318247)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104314'), 0.0043080853827732191)
orig-doc (('rec.motorcycles', '/home/dsr/scikit_learn_data/20news_home/20news-bydate-train/rec.motorcycles/104342'), 0.0042294930396086844)


"""
"""
Sample ouput of related words

neighbors of word jesus
0)jesus	1.0
1)christ	0.860119992825
2)disciples	0.795417656395
3)sins	0.767120306733
4)apostles	0.750180696093
5)kingdom	0.741343430201
6)messiah	0.729399236861
7)preached	0.720041229162
8)sin	0.70373303307
9)apostle	0.700112381087
-----------------------------
neighbors of word congress
0)congress	1.0
1)republican	0.729622493518
2)unemployment	0.703090731305
3)senate	0.699977217363
4)president	0.692364680451
5)stimulus	0.69106135376
6)clinton	0.686730404507
7)deficit	0.648808789729
8)initiative	0.644703183239
9)senator	0.641546835912
-----------------------------
neighbors of word tax
0)tax	1.0
1)vat	0.884753461508
2)income	0.853078608173
3)taxes	0.825410583617
4)revenue	0.749018425882
5)pay	0.722507251148
6)debt	0.70858594808
7)deficit	0.696701691953
8)unemployment	0.635297955259
9)paying	0.620214443173
-----------------------------
neighbors of word clinton
0)clinton	1.0
1)bush	0.852998368066
2)administration	0.789054528513
3)president	0.75689257196
4)democrats	0.747821631005
5)republican	0.699718582673
6)republicans	0.696396350506
7)congress	0.686730404507
8)deficit	0.686573069729
9)initiative	0.680064682161
-----------------------------
neighbors of word encryption
0)encryption	1.0
1)privacy	0.783796886374
2)secure	0.736865638024
3)wiretap	0.723013504433
4)scheme	0.718645164207
5)encrypted	0.712629136872
6)eff	0.706641398501
7)decrypt	0.703532050056
8)unbreakable	0.701735176845
9)conversations	0.694501678421
-----------------------------
neighbors of word government
0)government	1.0
1)libertarian	0.706619845974
2)corporations	0.651774901415
3)regulation	0.611673078513
4)libertarians	0.603999365891
5)governments	0.590575660261
6)rulers	0.53317596645
7)elimination	0.532251821693
8)citizens	0.531399350449
9)coercion	0.525797341549
-----------------------------
neighbors of word api
0)api	1.0
1)toolkit	0.607635515232
2)microsoft	0.568793686279
3)compiler	0.529889072609
4)ms	0.491191585888
5)interviews	0.483072214589
6)framework	0.471057278975
7)nt	0.453136021878
8)library	0.451632591401
9)toolkits	0.449313298937
-----------------------------
neighbors of word apple
0)apple	1.0
1)duo	0.656905501382
2)laserwriter	0.631306791861
3)dock	0.594643671072
4)quicktime	0.580775531159
5)230	0.545001125044
6)macs	0.53547036628
7)optional	0.527728555439
8)c650	0.526263895619
9)powerbook	0.500702411004
"""