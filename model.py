import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as smp
from PIL import Image
import time
import string
from nltk.corpus import stopwords

# # of words in vocab
# V = 25
# # of topics
# T = 10
# # of documents
# D = 500

# parameter for the per document topic distribution 
# alpha = np.ones(T)
# parameter for the per topic word distribution
# beta = np.ones(V)
# Dirichlet prior on the per document distribution over the topics:
# low alpha means document will likely contain a mixture of a few topics,
# while high alpha means document will likely contain a mixture of most
# of the topics
# theta = pm.Container([pm.Dirichlet("theta_%s" % i, theta=alpha) for i in range(D)])
# Dirichlet prior on the per topic distribution over the words:
# low beta means topic will likely contain a mixture of a few words from the vocab,
# while high beta means document will likely contain a mixture of most
# of the words from the vocab
# phi = pm.Container([pm.Dirichlet("theta_%s" % i, theta=beta) for i in range(T)])

#z = pm.Categorical()

#w = pm.Categorical()


# Toy example code

# First, create the topics
# gen_topic = Image.new("L", (5,5))
# topics = []
# for i in range(gen_topic.width):
#     for j in range(gen_topic.height):
#         gen_topic.putpixel((i,j),255)
#     topics.append(gen_topic)    
#     gen_topic = Image.new("L", (5,5))
# gen_topic = Image.new("L", (5,5))
# for i in range(gen_topic.height):
#     for j in range(gen_topic.width):
#         gen_topic.putpixel((j,i),255)
#     topics.append(gen_topic)    
#     gen_topic = Image.new("L", (5,5))

# generate dataset
# generate global word_vec and doc_vec 
# word_vec: word index in vocab for each word  
# doc_vec: doc index for each word
# dataset = []
# dataset_size = 500
# samples_per_image = 100
# word_vec = []
# doc_vec = []
# for i in range(dataset_size):
#     # generate dirichlet prior over the topics
#     theta = np.random.dirichlet(alpha)
#     gen_image = Image.new("L", (5,5))
#     for _ in range(samples_per_image):
#         # sample topic from theta
#         topic_ind = np.random.choice(T, 1, p=theta)[0]
#         # sample pixel from topic
#         if topic_ind < 5:
#             col = topic_ind
#             row = np.random.choice(5, 1)[0]
#             # sampled_pixels.append((row, col))
#         else:
#             row = topic_ind - 5
#             col = np.random.choice(5, 1)[0]
#             # sampled_pixels.append((row, col))
#         gen_image.putpixel((col,row),gen_image.getpixel((col,row))+30)
#         # generating word_vec
#         word_vec.append(row*5 + col)
#         # generating doc_vec
#         doc_vec.append(i)
#     dataset.append(gen_image)

# generate word_vec and doc_vec for real document corpus
def generate_vecs_from_file(file):
    file = open(file, "r+")
    abs_list = file.read().replace('\n', '').replace('Abstract', '').split("DOC DONE!")
    new_abs_list = []
    exclude = set(string.punctuation)
    en_stops = set(stopwords.words('english'))
    for abstract in abs_list:
        per_abs = []
        abstract = ''.join(ch.lower() for ch in abstract if ch not in exclude)
        abstract_split = abstract.split(" ")
        for i in abstract_split:
            if i.isalpha() and i.lower() not in en_stops:
                per_abs.append(i)
        new_abs_list.append(" ".join(per_abs))
    #alph_abs_list = " ".join([i for i in abs_list if i.isalpha()])
    # print(new_abs_list)
    dict_word_count_abstracts = {}
    for i, abstract in enumerate(new_abs_list):
        abstract_split = abstract.split(" ")
        for word in abstract_split:
            if word not in dict_word_count_abstracts:
                dict_word_count_abstracts[word] = set([i])
            else:
                dict_word_count_abstracts[word].add(i)
    final_abs_list= []
    for abstract in new_abs_list:
        abstract_split = abstract.split(" ")
        per_abs = []
        for word in abstract_split:
            if len(dict_word_count_abstracts[word]) >= 5:
                per_abs.append(word)
        final_abs_list.append(" ".join(per_abs))
    filtered_vocab = {k:dict_word_count_abstracts[k] for k in dict_word_count_abstracts if len(dict_word_count_abstracts[k]) >= 5}
    # with open("dict.txt", "w+") as f:
    #     f.write(str(filtered_vocab))
    V = len(filtered_vocab.keys())
    D = len(final_abs_list)
    filtered_vocab_keys = filtered_vocab.keys()
    vocab = {}
    reverse_vocab = {}
    for i, word in enumerate(filtered_vocab_keys):
        vocab[word] = i
        reverse_vocab[i] = word
    word_vec = []
    doc_vec = []
    final_abs_list = final_abs_list[:-1]
    for i, abstract in enumerate(final_abs_list):
        abs_split = abstract.split(" ")
        for word in abs_split:
            word_vec.append(vocab[word])
            doc_vec.append(i)
    return V, D, word_vec, doc_vec, reverse_vocab

generate_vecs_from_file("abstracts.txt")

# word_vec: word indices
# doc_vec: document indices
def gibbsSampler(alpha, beta, word_vec, doc_vec, num_topics, num_docs, num_vocab, doc_topic_counts, topic_word_counts, topic_assignments, no_iteration):
    if no_iteration == 0:
        for i in range(len(word_vec)):
            # select random topic for each word
            random_topic_assignment = np.random.choice(num_topics, 1)[0]
            doc_topic_counts[doc_vec[i]][random_topic_assignment] += 1
            topic_word_counts[random_topic_assignment][word_vec[i]] += 1
            topic_assignments.append(random_topic_assignment)
        # create initial random assignments, keep track of how many words
        # were assigned to each topic, how many times each topic was selected
        # in each document
        return topic_word_counts, doc_topic_counts, topic_assignments
    else:
        new_doc_topic_counts = [[0 for i in range(num_topics)] for j in range(num_docs)]
        new_topic_word_counts = [[0 for i in range(num_vocab)] for j in range(num_topics)]
        new_topic_assignments = []
        # print len(word_vec)
        np_twc = np.array(topic_word_counts)
        np_twc_sum = np.sum(np_twc, axis=1)
        np_dtc = np.array(doc_topic_counts)
        for i in range(len(word_vec)):
            # if i % 100 == 0:
                # print i
            topic_assignment = topic_assignments[i]
            doc_topic_counts[doc_vec[i]][topic_assignment] -= 1
            topic_word_counts[topic_assignment][word_vec[i]] -= 1
            # generate counts for sampling
            # start = time.time()
            den_doc_topic = np.sum(np_dtc[doc_vec[i]]) + num_topics*alpha - 1
            # print time.time() - start
            # np_twc = np.array(topic_word_counts)
            # print np_dtc, np_twc
            topic_probs = []
            for t in range(num_topics):
                num_doc_topic_t = np_dtc[doc_vec[i]][t] + alpha
                if t == topic_assignment:
                    num_doc_topic_t -= 1
                den_topic_word = np_twc_sum[t] + num_topics*beta
                if t == topic_assignment:
                    den_topic_word -= 1
                num_topic_word_v = topic_word_counts[t][word_vec[i]] + beta
                topic_probs.append((num_doc_topic_t/den_doc_topic) * (num_topic_word_v/den_topic_word))
            # print topic_probs
            topic_probs = [j/sum(topic_probs) for j in topic_probs]
            new_topic_assignment = np.random.choice(num_topics, 1, p=topic_probs)[0]
            # print new_topic_assignment
            new_topic_assignments.append(new_topic_assignment)
            new_doc_topic_counts[doc_vec[i]][new_topic_assignment] += 1
            new_topic_word_counts[new_topic_assignment][word_vec[i]] += 1
            doc_topic_counts[doc_vec[i]][topic_assignment] += 1
            topic_word_counts[topic_assignment][word_vec[i]] += 1
        return new_topic_word_counts, new_doc_topic_counts, new_topic_assignments

def dict_to_image(dict_topic_assignments, no_iteration):
    indices = []
    topics = []
    for i in range(V):
        indices.append((i/5, i - i/5 * 5))
    for topic in dict_topic_assignments:
        gen_topic = Image.new("L", (5,5))
        for i, index in enumerate(indices):
            gen_topic.putpixel((index[1], index[0]), dict_topic_assignments[topic][i])
        topics.append(gen_topic)
    widths, heights = zip(*(i.size for i in topics))

    total_width = sum(widths) + 27
    max_height = max(heights)

    new_im = Image.new('L', (total_width, max_height))

    x_offset = 0
    offset_im  = Image.new("L", (3,5))
    for i in range(offset_im.width):
        for j in range(offset_im.height):
            offset_im.putpixel((i,j), 70)
    for im in topics:
        new_im.paste(im, (x_offset,0))
        new_im.paste(offset_im, (x_offset+5,0))
        x_offset += im.size[0] + 3

    new_im.show()
    # file_name = "gibbs_sampling_iter_%s_toy_paper.png" % no_iteration 
    # new_im.save(file_name)

def calc_log_likelihood(topic_word_counts, beta):
    # res = T * math.log(math.gamma(V*beta)/(math.gamma(beta)**V))
    res = T * (math.lgamma(V*beta) - V*math.lgamma(beta))
    np_twc = np.array(topic_word_counts)
    np_twc_sum = np.sum(np_twc, axis=1)
    # print res

    for t in range(T):
        for v in range(V):
            res += math.lgamma(np_twc[t][v] + beta)
        res -= math.lgamma(np_twc_sum[t] + V*beta)
    
    return res

def gen_most_probable_words_per_topic(twc, reverse_vocab, beta):
    topic_words_most_probable = []
    for topic_num, topic in enumerate(twc):
        new_topic = [(i,topic[i]) for i in range(len(topic))]
        sorted_topic = sorted(new_topic, key=lambda x:x[1])[-12:]
        # print("sorted: ", sorted_probs)
        print ([reverse_vocab[i[0]] for i in sorted_topic])
        topic_words_most_probable.append([reverse_vocab[i[0]] for i in sorted_topic])
    return topic_words_most_probable

    
V, D, word_vec, doc_vec, reverse_vocab = generate_vecs_from_file("abstracts.txt")
T = 10
alpha = 10.0/T
beta = 0.1
dtc = [[0 for i in range(T)] for j in range(D)]
twc = [[0 for i in range(V)] for j in range(T)]
ta = []
log_likelihoods = []
no_iterations = 1001
iteration_checks = [0,1,2,5,10,20,50,100,150,200,300,500,1000]
start = time.time()
for i in range(no_iterations):
    twc, dtc, ta = gibbsSampler(alpha, beta, word_vec, doc_vec, T, D, V, dtc, twc, ta, i)
    if i in iteration_checks:
        log_likelihood = calc_log_likelihood(twc, beta)
        log_likelihoods.append(log_likelihood)
        print(log_likelihood)
        if i in [0, 1, 10, 200, 300, 500, 1000]:
            gen_most_probable_words_per_topic(twc, reverse_vocab, beta)
        # dict_to_image({i:twc[i] for i in range(T)}, i) 
print(log_likelihoods)
# print("duration: ", time.time() - start)
# plt.plot(iteration_checks, log_likelihoods)
# plt.show()
# Gibbs sampling gave me an assignment of topics to each word