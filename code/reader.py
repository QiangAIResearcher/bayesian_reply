import os
import itertools

from utils import *
from random import shuffle

EOS_TOKEN = "_eos_"

class TextReader(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.word2embed_dim = 300

    self.train_claim, self.train_comment,self.train_alabel, self.train_blabel, self.train_user =\
      self._file_to_vec('train')
    self.dev_claim, self.dev_comment, self.dev_alabel, self.dev_blabel, self.dev_user = \
      self._file_to_vec('dev')
    self.test_claim, self.test_comment, self.test_alabel, self.test_blabel, self.test_user = \
      self._file_to_vec('test')
    self.num_user = 3703

    self.train_claim = list(self.train_claim) + list(self.dev_claim)
    self.train_comment = list(self.train_comment) + list(self.dev_comment)
    self.train_alabel = list(self.train_alabel) + list(self.dev_alabel)
    self.train_blabel = list(self.train_blabel) + list(self.dev_blabel)
    self.train_user = list(self.train_user) + list(self.dev_user)


  def _file_to_vec(self,data_type):
    if data_type == 'dev':
      data_type = 'train'
    claim = np.load(os.path.join(self.data_path, str(data_type) + '_claim3.npy'))
    comment = np.load(os.path.join(self.data_path, str(data_type) + '_comment3.npy'))
    alabel = np.load(os.path.join(self.data_path, str(data_type) + '_alabel3.npy'))
    blabel = np.load(os.path.join(self.data_path, str(data_type) + '_blabel3.npy'))
    user = np.load(os.path.join(self.data_path, str(data_type) + '_userid3.npy'))
    return claim, comment, alabel, blabel, user

  def get_data_from_type(self, data_type):
    if data_type == "train":
      return self.train_claim, self.train_comment,self.train_alabel,self.train_blabel,self.train_user
    elif data_type == "dev":
      return self.dev_claim, self.dev_comment,self.dev_alabel,self.dev_blabel,self.dev_user
    elif data_type == "test":
      return self.test_claim, self.test_comment,self.test_alabel,self.test_blabel,self.test_user
    else:
      raise Exception(" [!] Unkown data type %s: %s" % data_type)


  def random(self, data_type="train",random_bool = True):
    claim, comment, alabel, blabel,user = self.get_data_from_type(data_type)
    indices = list(range(len(claim)))
    if random_bool == True:
      shuffle(indices)

    data1 = [claim[idx] for idx in indices]
    data2 = [comment[idx] for idx in indices]
    stance_label = [alabel[idx] for idx in indices]
    veracity_label = [blabel[idx] for idx in indices]
    # userid = [user[idx] for idx in indices]
    return data1,data2,stance_label,veracity_label,user

