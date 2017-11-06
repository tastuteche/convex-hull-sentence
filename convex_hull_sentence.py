import numpy as np
import matplotlib.pyplot as plt
b_dir = './glove-global-vectors-for-word-representation/'
from sklearn.manifold import TSNE

# preparing embedding index
embeddings_index = {}
word_glove_list = []
with open(b_dir + 'glove.6B.200d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        word_glove_list.append(word)


sentence1 = "Sir Ken Robinson makes an entertaining and profoundly moving case for creating an education system that nurtures (rather than undermines) creativity."
sentence2 = "With the same humor and humanity he exuded in 'An Inconvenient Truth, ' Al Gore spells out 15 ways that individuals can address climate change immediately, from buying a hybrid to inventing a new, hotter brand name for global warming."
sentence_list = [sentence1, sentence2]


word_all = set()

a = np.arange(len(embeddings_index))
np.random.shuffle(a)
word_all.update([word_glove_list[i] for i in a[:5000]])

for sent in sentence_list:
    for word in sent.split(' '):
        word = word.lower()
        if word in embeddings_index:
            word_all.add(word)

tokens = []
labels = []
for word in word_all:
    tokens.append(embeddings_index[word])
    labels.append(word)

tsne_model = TSNE(perplexity=40, n_components=2,
                  init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)
dic_point = {}
for i, p in enumerate(new_values):
    dic_point[labels[i]] = p


# x, y = zip(*new_values)

# for i in range(len(x)):
#     plt.scatter(x[i], y[i])
#     plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2),
#                  textcoords='offset points', ha='right', va='bottom')
# plt.show()
from scipy.spatial import ConvexHull

# http://matplotlib.org/examples/color/colormaps_reference.html
import seaborn as sns
colors = sns.mpl_palette("Dark2", len(sentence_list))

from shapely.geometry import box, Polygon


def sent_plot(i):
    sent = sentence_list[i]
    all_w = list(
        set([w for w in sent.lower().split(' ') if w in embeddings_index]))
    points = np.array([dic_point[w] for w in all_w])
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], 'o', c=colors[i])
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.plot(points[hull.vertices, 0],
             points[hull.vertices, 1], '--', lw=2, c=colors[i])
    plt.plot(points[hull.vertices[0], 0],
             points[hull.vertices[0], 1], 'o', c=colors[i])
    for j, label in enumerate(all_w):
        plt.annotate(label, xy=(points[j][0], points[j][1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', color=colors[i])
    polygon_shape = Polygon(points[hull.vertices])
    return polygon_shape
    # plt.show()


p0 = sent_plot(0)
p1 = sent_plot(1)
plt.savefig('convex_sentence.png', dpi=200)
plt.clf()
plt.cla()
plt.close()

p0.intersection(p1).area
# 13930.357679770113
p0.area
# 15788.895608835395
p1.area
# 19109.439874825217
