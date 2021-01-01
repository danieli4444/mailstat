
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mailparser
import os
import pprint


TALPINET_MAILS_FOLDER_PATH = "talpinet_email_text\\"
TALPINET_EMAIL_MESSAGE_FOLDER = "talpinet_email_messages\\"
LOGS_FOLDER= "logs\\"

file_list = os.listdir(TALPINET_EMAIL_MESSAGE_FOLDER)
file_path = TALPINET_EMAIL_MESSAGE_FOLDER + file_list[5]

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
    file_list = os.listdir(TALPINET_MAILS_FOLDER_PATH)
    mail_content_list = []
    for file in file_list:
        with open(TALPINET_MAILS_FOLDER_PATH + file, 'r', encoding='utf-8') as fp:
            body = fp.read()
            mail_content_list.append(body)
    print(" gathered all mails content!")
    return mail_content_list

def get_all_mails_body_list():
    file_list = os.listdir(TALPINET_EMAIL_MESSAGE_FOLDER)
    mails_body_list = []
    for file in file_list:
        cur_email = mailparser.parse_from_file(TALPINET_EMAIL_MESSAGE_FOLDER + file)
        mails_body_list.append(cur_email.body)
    print("parsed all mails body!")
    return mails_body_list

def retreive_mail(file_name):
    mail = mailparser.parse_from_file(TALPINET_EMAIL_MESSAGE_FOLDER + file_name)
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
    vect = TfidfVectorizer(stop_words=mystopwords_list, max_df=0.50, min_df=2)
    X = vect.fit_transform(mails_body_list)
    #X_dense = X.todense()
    #coords = PCA(n_components=2).fit_transform(X_dense)
    #plt.scatter(coords[:, 0], coords[:, 1], c='m')
    #plt.show()
    features = vect.get_feature_names()
    s1 = "top tf-idf scores:\n" + str(top_feats_in_doc(X, features, 1, 50))
    s2 = "top tf mean scores:\n" + str(top_mean_feats(X, features))
    print(s1)
    print(s2)

def basic_text_processing():
    mails_body_list = get_all_mails_content_list()
    # show bag of words analysis:
    bow = get_sorted_bag_of_words(mails_body_list)
    print(" top 100 words in all whole corpus:")
    pprint.pprint(bow[:100])
    # show tf means and tf-idf analysis:
    perform_tf_idf_analysis(mails_body_list)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #mail = retreive_mail('msg_13893.msg')
    #print(mail.body)
    basic_text_processing()
    print("Finished!")


