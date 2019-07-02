import time
import numpy as np
import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pack_padded_sequence

class claim_reply(nn.Module):
  """Variational Document Model"""
  def __init__(self,reader,batch_size=32, nb_lstm_layers=1, h_dim=30,var_dim = 10, word_embed_dim=300,
               blabel_size = 2, alabel_size=2,num_user = 3703,user_embed_dim=30,sample_number=20):
    super(claim_reply, self).__init__()
    self.reader = reader
    self.batch_size = batch_size
    self.nb_lstm_layers = nb_lstm_layers
    self.h_dim = h_dim
    self.word_embed_dim = word_embed_dim
    self.var_dim = var_dim
    self.num_user = num_user
    self.user_embed_dim = user_embed_dim
    self.alabel_size = alabel_size
    self.blabel_size = blabel_size
    self.sample_number = sample_number


    self.lstm1 = nn.LSTM(self.word_embed_dim, self.h_dim,num_layers=1,bidirectional=True)
    self.lstm2 = nn.LSTM(self.word_embed_dim, self.h_dim, num_layers=1,bidirectional=True)

    self.fc1 = nn.Linear(self.h_dim, self.var_dim,bias=True)
    self.fc2_1 = nn.Linear(self.var_dim, self.var_dim,bias=True)
    self.fc2_2 = nn.Linear(self.var_dim, self.var_dim,bias=True)

    self.fc3 = nn.Linear(self.h_dim, self.var_dim,bias=True)
    self.fc4_1 = nn.Linear(self.var_dim, self.var_dim, bias=True)
    self.fc4_2 = nn.Linear(self.var_dim, self.var_dim, bias=True)

    self.fc5 = nn.Linear(1, blabel_size, bias=False)
    self.fc6 = nn.Linear(self.h_dim + blabel_size, self.var_dim, bias=True)
    self.fc7_1 = nn.Linear(self.var_dim, self.var_dim, bias=True)
    self.fc7_2 = nn.Linear(self.var_dim, self.var_dim, bias=True)

    # self.fc8 = nn.Linear(1, alabel_size, bias=False)
    self.fc9 = nn.Linear(self.h_dim + blabel_size, self.var_dim)
    self.fc10_1 = nn.Linear(self.var_dim, self.var_dim)
    self.fc10_2 = nn.Linear(self.var_dim, self.var_dim)

    self.fc11 = nn.Linear(self.var_dim+self.h_dim, blabel_size, bias=True)
    self.fc12 = nn.Linear(2 * self.var_dim, alabel_size, bias=True)

    self.user_embedding = nn.Embedding(self.num_user,self.user_embed_dim)

    for name, param in self.lstm1.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.001)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)
    for name, param in self.lstm2.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.001)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

  def repara_Gaussian(self,mu,logvar):
    if self.training:
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def encode(self,state_1, y):
    lambda_theta = torch.tanh(self.fc1(state_1))
    self.mu_theta = self.fc2_1(lambda_theta)
    self.log_sigma_theta = self.fc2_2(lambda_theta)

    if self.training:
      y = y.float()
      sy = self.fc5(y.view(-1, 1))
      gamma = torch.cat((state_1, sy), dim=1)
      lambda_phi = torch.tanh(self.fc6(gamma))
      self.mu_phi = self.fc7_1(lambda_phi)
      self.log_sigma_phi = self.fc7_2(lambda_phi)

      return self.mu_phi,self.log_sigma_phi

    else:
      return self.mu_theta,self.log_sigma_theta

  def decode(self, mu, log_sigma):
    for i in range(self.sample_number):
      if i == 0:
        h = self.repara_Gaussian(mu, log_sigma)
      else:
        h = h+self.repara_Gaussian(mu, log_sigma)
    return h/self.sample_number

  def forward(self,x1,x2,len1,len2, y):
    ##### preprocess the claim and the comment
    ## padding zeros to the maximum length
    max_len1 = max(len1)
    temp_emb1 = np.zeros((len(len1), max_len1, self.word_embed_dim))
    for i, x_len in enumerate(len1):
      sentence = x1[i]
      temp_emb1[i,-x_len:,:] = sentence
      len1[i] = max_len1

    max_len2 = max(len2)
    temp_emb2 = np.zeros((len(len2), max_len2, self.word_embed_dim))
    for i, x_len in enumerate(len2):
      sentence = x2[i]
      temp_emb2[i, -x_len:, :] = sentence
      len2[i] = max_len2

    ## to variable
    x1 = Variable(torch.FloatTensor(temp_emb1))
    x2 = Variable(torch.FloatTensor(temp_emb2))

    ## pack
    x1_pack = pack_padded_sequence(x1, len1, batch_first=True)
    x2_pack = pack_padded_sequence(x2, len2, batch_first=True)

    #### encode and decode
    ## LSTM
    output_1_pack, hidden_state_1 = self.lstm1(x1_pack, None)
    output_2_pack, hidden_state_2 = self.lstm2(x2_pack, None)

    ## unpack state
    state_1 = hidden_state_1[0][-1, :, :]
    state_2 = hidden_state_2[0][-1, :, :]

    ## encode
    mu, logvar = self.encode(state_1, y)
    h1 = self.decode(mu, logvar)

    hh = torch.cat((h1,state_2),dim=1)
    prob_y = F.softmax(self.fc11(hh), dim=1)

    return prob_y

  def loss_function(self, output_b, blabels):

    kl = gau_kl(p_mu=self.mu_phi, p_ln_var=self.log_sigma_phi, q_mu=self.mu_theta, q_ln_var=self.log_sigma_theta)

    # log likelihood of p(y)
    blabels = blabels.float()
    pred_true = output_b[:, 0]
    pred_false = output_b[:, 1]
    log_likely_y = torch.sum(torch.mul(torch.log(pred_true), blabels), dim=0) + \
                   torch.sum(torch.mul(torch.log(pred_false), 1 - blabels), dim=0)

    obj = log_likely_y - kl
    loss = -obj
    assert (loss > 0)
    return loss


def gau_kl(p_mu, p_ln_var, q_mu, q_ln_var):
  """
  Kullback-Liebler divergence from Gaussian p_mu,p_ln_var to Gaussian q_mu,q_ln_var.
  Diagonal covariances are assumed.  Divergence is expressed in nats.
  """
  if (len(p_mu.shape) == 2):
    axis = 1
  else:
    axis = 0

  p_var = torch.exp(p_ln_var)
  q_var = torch.exp(q_ln_var)
  # Determinants of diagonal covariances p_var, q_var
  dp_var = torch.prod(p_var, dim=axis)
  dq_var = torch.prod(q_var, dim=axis)
  # Inverse of diagonal covariance q_var
  iq_var = 1. / q_var
  # Difference between means p_mu, q_mu
  diff = q_mu - p_mu

  aaa = torch.log(dq_var / dp_var)  # log |\Sigma_q| / |\Sigma_p|
  bbb = torch.sum(iq_var * p_var, dim=axis)  # + tr(\Sigma_q^{-1} * \Sigma_p)
  ccc = torch.sum(diff * iq_var * diff, dim=axis)  # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
  ddd = p_mu.shape[-1]  # - N
  kl_diver = torch.sum(0.5 * (aaa + bbb + ccc - ddd))

  return kl_diver