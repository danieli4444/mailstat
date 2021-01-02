
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import matplotlib.cm as cm
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mailparser
import os
import re

import pprint


PLAIN_TEXT_MAILS_PATH = "talpinet_email_text\\"
EMAIL_FOLDER_PATH = "talpinet_email_messages\\"
LOGS_FOLDER= "logs\\"

file_list = os.listdir(PLAIN_TEXT_MAILS_PATH)
file_path = EMAIL_FOLDER_PATH + file_list[5]

heb_stopwords = ["על","את","לא","של"]
eng_stopwords = ["cv", "www", "io","guy","ment","linkedin",
                 "computer","attached", 'com', 'google',
                 'groups', 'group','unsubscribe', 'https', 'email',
                 'gmail','msgid','googlegroups', 'send','visit',
                 'message', 'web', 'view', 'discussion', 'received',
                 'stop', 'emails','receiving','subscribed','40mail',
                 'looking','experience', 'data', 'team', 'product',
                 'company','years']
mystopwords_list = heb_stopwords + eng_stopwords
mystopwords_list.extend(ENGLISH_STOP_WORDS)


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def get_all_mails_content_list():
    file_list = os.listdir(PLAIN_TEXT_MAILS_PATH)
    mail_content_list = []
    for file in file_list:
        with open(PLAIN_TEXT_MAILS_PATH + file, 'r', encoding='utf-8') as fp:
            body = fp.read()
            google_spam_string_place = body.find("You received this message because you are subscribed to the Google")
            email_title_string = body.find("[talpinet]",google_spam_string_place)
            cleaned_body = body[:google_spam_string_place] + body[email_title_string:]
            cleaned_body = re.sub("\S*\d\S*", " ", cleaned_body)
            cleaned_body = re.sub(r'http\S+', '', cleaned_body)  # removes URLs with http
            cleaned_body = re.sub(r'www\S+', '', cleaned_body)  # removes URLs with www
            mail_content_list.append(cleaned_body)
    print(" gathered all mails content!")
    return mail_content_list

def get_all_mails_body_list():
    file_list = os.listdir(EMAIL_FOLDER_PATH)
    mails_body_list = []
    for file in file_list:
        cur_email = mailparser.parse_from_file(EMAIL_FOLDER_PATH + file)
        mails_body_list.append(cur_email.body)
    print("parsed all mails body!")
    return mails_body_list

def retreive_mail(file_name):
    mail = mailparser.parse_from_file(EMAIL_FOLDER_PATH + file_name)
    return mail


def get_sorted_bag_of_words(mails_body_list):
    cv = CountVectorizer(stop_words=mystopwords_list, min_df=2)
    word_count_vector = cv.fit_transform(mails_body_list)
    bag_of_words = word_count_vector.sum(axis=0)
    print(bag_of_words.shape)
    words_freq = [(word, bag_of_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq

def perform_tf_idf_analysis(mails_body_list):
    vectorizer = TfidfVectorizer(stop_words=mystopwords_list, max_df=0.50, min_df=2)
    X = vectorizer.fit_transform(mails_body_list)
    #X_dense = X.todense()
    #coords = PCA(n_components=2).fit_transform(X_dense)
    #plt.scatter(coords[:, 0], coords[:, 1], c='m')
    #plt.show()
    features = vectorizer.get_feature_names()
    #s1 = "top tf-idf scores:\n" + str(top_feats_in_doc(X, features, 1, 50))
    #s2 = "top tf mean scores:\n" + str(top_mean_feats(X, features))
    #print(s1)
    #print(s2)
    tf_idf = pd.DataFrame(data=X.toarray(),columns=vectorizer.get_feature_names())
    final_df = tf_idf
    print("{} rows".format(final_df.shape[0]))
    final_df.T.nlargest(5, 0)
    return final_df,features

def basic_text_processing():
    mails_body_list = get_all_mails_content_list()

    # show bag of words analysis:
    #bow = get_sorted_bag_of_words(mails_body_list)
    #print(" top 100 words in all whole corpus:")
    #pprint.pprint(bow[:100])

    # show tf means and tf-idf analysis:
    perform_tf_idf_analysis(mails_body_list)


def run_KMeans(max_k, data):
    max_k += 1
    kmeans_results = dict()
    for k in range(2, max_k):
        kmeans = cluster.KMeans(n_clusters = k
                               , init = 'k-means++'
                               , n_init = 10
                               , tol = 0.0001
                               , n_jobs = -1
                               , random_state = 1
                               , algorithm = 'full')

        kmeans_results.update( {k : kmeans.fit(data)} )
    return kmeans_results

def printAvg(avg_dict):
    for avg in sorted(avg_dict.keys(), reverse=True):
        print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))


def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

    ax1.axvline(x=silhouette_avg, color="red",
                linestyle="--")  # The vertical line for average silhouette score of all the values
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')

    y_lower = 10
    sample_silhouette_values = silhouette_samples(df, kmeans_labels)  # Compute the silhouette scores for each sample
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                 str(i))  # Label the silhouette plots with their cluster numbers at the middle
        y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
    plt.show()


def silhouette(kmeans_dict, df, plot=False):
    df = df.to_numpy()
    avg_dict = dict()
    for n_clusters, kmeans in kmeans_dict.items():
        kmeans_labels = kmeans.predict(df)
        silhouette_avg = silhouette_score(df, kmeans_labels)  # Average Score for all Samples
        avg_dict.update({silhouette_avg: n_clusters})

        if (plot): plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)

def get_top_features_cluster(tf_idf_array, features,prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.show()

# Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict

def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

def perform_clustering():
    mails_body_list = get_all_mails_content_list()
    final_df,features = perform_tf_idf_analysis(mails_body_list)
    print(final_df.T.nlargest(5, 0))

    k = 8
    kmeans_results = run_KMeans(k, final_df)

    best_result = 8
    kmeans = kmeans_results.get(best_result)

    final_df_array = final_df.to_numpy()
    prediction = kmeans.predict(final_df)
    n_feats = 20
    dfs = get_top_features_cluster(final_df_array, features, prediction, n_feats)
    plotWords(dfs, 13)
    centroids = pd.DataFrame(kmeans.cluster_centers_)
    centroids.columns = final_df.columns
    generateWordClouds(centroids)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #mail = retreive_mail('msg_13893.msg')
    #print(mail.body)
    #basic_text_processing()
    perform_clustering()
    print("Finished!")


