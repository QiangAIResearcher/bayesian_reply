# -*- coding: utf-8 -*-
"""
This file contains preprocessing routines to convert RumourEval data
into the format of branchLSTM input: it splits conversation trees into
branches and extracts features from tweets including average of word2vec and
extra features (specified in
https://www.aclweb.org/anthology/S/S17/S17-2083.pdf) and concatenates them.
Assumes that data is in the same folder as the script.
Dataset: http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools

Saves processed data in saved_data folder
"""
import os
import numpy as np
import json
import gensim
import nltk
from nltk.corpus import stopwords
import re

EOS_TOKEN = "_eos_"

# process tweet into features

def getW2vCosineSimilarity(words, wordssrc):
    global model
    words2 = []
    for word in words:
        if word in model.vocab:  # change to model.wv.vocab
            words2.append(word)
    wordssrc2 = []
    for word in wordssrc:
        if word in model.vocab:  # change to model.wv.vocab
            wordssrc2.append(word)

    if len(words2) > 0 and len(wordssrc2) > 0:
        return model.n_similarity(words2, wordssrc2)
    return 0.


def cleantweet(tweettext):
    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-zA-Z]{2}))+/(?:[-\w.]|(?:%[\da-zA-Z]{2}))+', tweettext)
    for url in urls:
        tweettext = tweettext.replace(url, "picpicpic")

    return tweettext

def str_to_wordlist(tweettext, remove_stopwords=False):
    #  Remove non-letters
    # NOTE: Is it helpful or not to remove non-letters?
    tweettext = cleantweet(tweettext)
    str_text = re.sub("[^a-zA-Z]", " ", tweettext)
    # Convert words to lower case and split them
    # words = str_text.lower().split()
    words = nltk.word_tokenize(str_text.lower())
    # Optionally remove stop words (false by default)
    # NOTE: generic list of stop words, should i remove them or not?
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    # 5. Return a list of words
    return (words)


def loadW2vModel():
    # LOAD PRETRAINED MODEL
    global model
    print("Loading the Word2Vec model")
    model = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join('GoogleNews-vectors-negative300.bin'), binary=True)
    print("Model Loaded!")


def text2fea(text):
    features = []
    text = cleantweet(text)
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
                                       '', text.lower()))
    # issourcetw = int(tw['in_reply_to_screen_name'] == None)
    # hasqmark = 0
    # if tw['text'].find('?') >= 0:
    #     hasqmark = 1
    # hasemark = 0
    # if tw['text'].find('!') >= 0:
    #     hasemark = 1
    # hasperiod = 0
    # if tw['text'].find('.') >= 0:
    #     hasperiod = 0
    # hasurl = 0
    # if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
    #     hasurl = 1
    # haspic = 0
    # if (tw['text'].find('picpicpic') >= 0) or (
    #             tw['text'].find('pic.twitter.com') >= 0) or (
    #             tw['text'].find('instagr.am') >= 0):
    #     haspic = 1
    #
    # hasnegation = 0
    # negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
    #                  'neither', 'nor', 'nowhere', 'hardly',
    #                  'scarcely', 'barely', 'don', 'isn', 'wasn',
    #                  'shouldn', 'wouldn', 'couldn', 'doesn']
    # for negationword in negationwords:
    #     if negationword in tokens:
    #         hasnegation += 1
    #
    # charcount = len(tw['text'])
    # wordcount = len(nltk.word_tokenize(re.sub(r'([^\s\w]|_)+',
    #                                           '',
    #                                           tw['text'].lower())))
    #
    # swearwords = []
    # with open('badwords.txt', 'r') as f:
    #     for line in f:
    #         swearwords.append(line.strip().lower())
    #
    # hasswearwords = 0
    # for token in tokens:
    #     if token in swearwords:
    #         hasswearwords += 1
    # uppers = [l for l in tw['text'] if l.isupper()]
    # capitalratio = len(uppers) / len(tw['text'])
    #
    # # %%
    avgw2v = text2vec(text, avg=True)
    # features = [charcount, wordcount, issourcetw, hasqmark, hasemark,
    #             hasperiod, hasurl, haspic, hasnegation, hasswearwords,
    #             capitalratio]
    # features.extend(avgw2v)
    # features = np.asarray(features, dtype=np.float32)
    return features


def text2vec(text, avg=True):
    global model
    # num_features = 300
    temp_rep = [] # np.zeros(num_features)
    wordlist = str_to_wordlist(text, remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep.append(model[wordlist[w]])
    # if avg:
    #     vec = temp_rep / len(wordlist)
    # else:
    #     # sum
    #     vec = temp_rep
    # return vec

    return temp_rep

def convert_a_label(label):
    if label == "support":
        return (1)
    elif label == "deny":
        return (0)
    elif label == "comment":
        return (2)
    elif label == "query":
        return (3)
    else:
        print(label)

def convert_b_label(label):
    if label == "true":
        return (1)
    elif label == "false":
        return (0)
    elif label == "unverified":
        return (2)
    else:
        print(label)

def load_true_labels(dataset_name):

    traindev_path = os.path.join("semeval_2017task8", "semeval2017-task8-dataset", "traindev")
    data_files = {"a_dev": os.path.join(traindev_path, "rumoureval-subtaskA-dev.json"),
                  "a_train": os.path.join(traindev_path, "rumoureval-subtaskA-train.json"),
                  "a_test": "subtaska.json",
                  "b_dev": os.path.join(traindev_path, "rumoureval-subtaskB-dev.json"),
                  "b_train": os.path.join(traindev_path, "rumoureval-subtaskB-train.json"),
                  "b_test": "subtaskb.json",
                  }

    # Load the dictionary containing the tweets and labels from the .json file
    with open(data_files[dataset_name]) as f:
        for line in f:
            tweet_label_dict = json.loads(line)

    return tweet_label_dict

def extract_conversation(path,tweetid_text,tweetid_owner,userid_tweetid,commentid_srcid):

    folds = sorted(os.listdir(path))
    newfolds = [i for i in folds if i[0] != '.']
    folds = newfolds

    for nfold, fold in enumerate(folds):
        path_to_tweets = os.path.join(path, fold)
        tweet_data = sorted(os.listdir(path_to_tweets))
        newfolds = [i for i in tweet_data if i[0] != '.']
        tweet_data = newfolds

        for foldr in tweet_data:
            ## deal with the source tweet
            path_src = path_to_tweets + '/' + foldr + '/source-tweet'
            files_t = sorted(os.listdir(path_src))
            with open(os.path.join(path_src, files_t[0])) as f:
                for line in f:
                    src = json.loads(line)
                    srcid = src['id_str']
                    user_id = src['user']['id_str']
                    tweetid_owner[srcid] = user_id
                    tweetid_text[srcid] = src['text']

            ## deal with replies
            path_repl = path_to_tweets + '/' + foldr + '/replies'
            files_t = sorted(os.listdir(path_repl))
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            for repl_file in files_t:
                with open(os.path.join(path_repl, repl_file)) as f:
                    for line in f:
                        tw = json.loads(line)
                        replyid = tw['id_str']
                        tweetid_text[replyid] = tw['text']
                        user_id = tw['user']['id_str']
                        tweetid_owner[replyid] = user_id

                        if user_id not in userid_tweetid.keys():# a new user
                            userid_tweetid[user_id] = [srcid,replyid]
                        elif srcid not in userid_tweetid[user_id]: # an old user but a new source tweet
                            userid_tweetid[user_id].extend([srcid,replyid])
                        else:
                            userid_tweetid[user_id].append(replyid) # an old user and an old source tweet

                        commentid_srcid[replyid] = srcid
    # print(len(tweetid_text.keys()),len(tweetid_owner.keys()),len(userid_tweetid.keys()),len(commentid_srcid.keys()))
    return tweetid_text, tweetid_owner, userid_tweetid,commentid_srcid

def load_dataset():

    # Load train and dev data
    path_to_train_dev = os.path.join('semeval_2017task8', 'semeval2017-task8-dataset/rumoureval-data')
    path_to_test = os.path.join('semeval_2017task8', 'semeval2017-task8-test-data')

    tweetid_text = {}  # each tweetid corresponds to a tweet
    tweetid_owner = {}  # each tweet is owned by one user
    userid_tweetid = {}  # each user may reply/comment many tweets
    commentid_srcid = {} # each comment corresponds to a source tweet


    tweetid_text, tweetid_owner, userid_tweetid,commentid_srcid = \
        extract_conversation(path_to_train_dev,tweetid_text, tweetid_owner,userid_tweetid,commentid_srcid)
    tweetid_text, tweetid_owner, userid_tweetid,commentid_srcid = \
        extract_conversation(path_to_test,tweetid_text, tweetid_owner,userid_tweetid,commentid_srcid)

    return tweetid_text, tweetid_owner, userid_tweetid,commentid_srcid


def preprocess_data():
    # Load tweetid and labels
    a_train = load_true_labels("a_train")
    a_dev = load_true_labels("a_dev")
    a_test = load_true_labels("a_test")

    dataset = {}
    dataset['train'] = a_train.keys()
    dataset['dev'] = a_dev.keys()
    dataset['test'] = a_test.keys()

    b_train = load_true_labels("b_train")
    b_dev = load_true_labels("b_dev")
    b_test = load_true_labels("b_test")

    tweetid_text, tweetid_owner, userid_tweetid, commentid_srcid = load_dataset()

    all_users = list(userid_tweetid.keys())

    path_to_saved_data = 'saved_data'
    all_text = os.path.join(path_to_saved_data, 'all_text' + ".txt")
    all_file = open(all_text, "w")
    whichset = ['train', 'dev', 'test']
    for sset in whichset:
        claim_list = []
        comment_list = []
        user_list = []
        alabel_list = []
        blabel_list = []
        count_true = 0
        count_false = 0

        # save .txt files
        path_to_saved_data = 'saved_data'
        filename = os.path.join(path_to_saved_data, str(sset) + ".txt")
        file = open(filename, "w")

        for tweetid in dataset[sset]:
            if tweetid in commentid_srcid.keys():# a comment tweet
                comment = tweetid_text[tweetid]
                claimid = commentid_srcid[tweetid]
                claim = tweetid_text[claimid]
                userid = tweetid_owner[tweetid]
                claimids_user = userid_tweetid[userid]
                claims_user = [tweetid_text[id] for id in claimids_user]

                # wordlist1 = str_to_wordlist(claim, remove_stopwords=False)
                # wordlist2 = str_to_wordlist(comment, remove_stopwords=False)
                # wordstr1 = " ".join(wordlist1)+" " + EOS_TOKEN + " "
                # wordstr2 = " ".join(wordlist2) + " " + EOS_TOKEN + " "
                # file.write(wordstr1 + wordstr2)
                # all_file.write(wordstr1 + wordstr2)

                clmVec = text2vec(claim)
                comVec = text2vec(comment)

                userid_idx = all_users.index(userid)

                if sset == 'train':
                    a_label = a_train[tweetid]
                    b_label = b_train[claimid]
                elif sset == 'dev':
                    a_label = a_dev[tweetid]
                    b_label = b_dev[claimid]
                elif sset == 'test':
                    a_label = a_test[tweetid]
                    b_label = b_test[claimid]
                a_label = convert_a_label(a_label)
                b_label = convert_b_label(b_label)

                if len(clmVec) >0 and len(comVec)>0 and b_label < 2 and a_label < 2 :
                    if sset != 'test':
                        if b_label == 0: #true
                            count_true += 1
                        else: # false
                            count_false += 1
                    else:
                        if b_label == 0: #true
                            count_true += 1
                        else: # false
                            count_false += 1
                    claim_list.append(clmVec)
                    comment_list.append(comVec)
                    # claim_list.append(claim)
                    user_list.append(userid_idx)
                    alabel_list.append(a_label)
                    blabel_list.append(b_label)

                # elif tweetid in tweetid_owner.keys(): # source tweet
                #     claim = tweetid_text[tweetid]
                #     commentids = [id for id in commentid_srcid.keys() if commentid_srcid[id]==tweetid]
                #     comments = [tweetid_text[id] for id in commentids]
                #     users = [tweetid_owner[id] for id in commentids]
                # else:
                #     print("Wrong tweetid: ",tweetid)
                #     # del "580323060533764097": "true" from subtaskB-dev
                #     # del "580323060533764097": "support" from subtaskA-dev
        file.close()
        print("{} dataset has {} true claims and {} false claims".format(sset,count_true,count_false))

        # path_to_save_sets = os.path.join(path_to_saved_data, sset)
        # if not os.path.exists(path_to_save_sets):
        #     os.makedirs(path_to_save_sets)

        # save to files
        np.save(os.path.join(path_to_saved_data, str(sset)+'_claim'), claim_list)
        np.save(os.path.join(path_to_saved_data, str(sset)+'_comment'), comment_list)
        np.save(os.path.join(path_to_saved_data, str(sset)+'_userid'), user_list)
        np.save(os.path.join(path_to_saved_data, str(sset)+'_alabel'), alabel_list)
        np.save(os.path.join(path_to_saved_data, str(sset)+'_blabel'), blabel_list)
        # np.save(os.path.join(path_to_save_sets, 'all_count'), all_count)
    all_file.close()

if __name__ == "__main__":
    # Import NLTK data
    nltk_data_location = os.path.dirname(os.path.realpath(__file__))
    nltk.download('punkt', download_dir=nltk_data_location)

    loadW2vModel()

    # tweetid_text, tweetid_owner, userid_tweetid = load_dataset()

    preprocess_data()

    print("Done!")