#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  krocki 9/21/20
#

import pickle, gzip
import numpy as np
import math
import time
import argparse
import matplotlib

# save to file
matplotlib.use('Agg')
import matplotlib.pyplot as plt

norm = lambda x: np.linalg.norm(x)
normalize = lambda x: x / norm(x)
distance = lambda w, v: norm(w-v)
cosine = lambda w, v: (np.dot(w, v)/(norm(w)*norm(v)))[0]

def debug_print(x, s):
  print('\n-----')
  print(s + '.shape')
  print(x.shape)
  print('')
  print("\t" + str(x).replace('\n','\n\t'))
  print('=====')

def random_batch(x, b):
  r = np.random.randint(0, high=x.shape[0], size=B)
  # returns x.shape [b, 784]
  return x[r, :], r

def save_arr(a, filename):
  plt.matshow(a)
  plt.savefig(filename)

def show_imgs(xs, filename=None, labels=None):

  #x.shape: [N, 28, 28]
  num = xs.shape[0]

  # try to make it square
  num_sqrt = round(math.sqrt(num))
  x_dim = num // num_sqrt
  extra_row = 0 if num == x_dim*num_sqrt else 1
  y_dim = num // x_dim + extra_row

  fig, ax = plt.subplots(y_dim, x_dim, squeeze=False)

  for y in range(y_dim):
    for x in range(x_dim):

      idx = x+y*x_dim

      if idx<num:
        label = labels[idx] if labels.any() else idx
        ax[y,x].matshow(xs[idx].reshape(28,28))
        ax[y,x].set_title('{}'.format(label),
          fontdict={
            'fontsize': 6,
            'fontweight': 'medium'
          })

        # hide tick labels
        ax[y,x].get_xaxis().set_visible(False)
        ax[y,x].get_yaxis().set_visible(False)

      else:
        ax[y,x].axis("off") # don't plot idx>num

  #fig.tight_layout()

  plt.subplots_adjust(top=1.4)

  if filename:
    plt.savefig(filename, bbox_inches='tight')

  else: plt.show()

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-b', '--batchsize', type=int, default=16, help='# of keys')
  parser.add_argument('-q', '--queries', type=int, default=16, help='# of queries')
  parser.add_argument('-n', '--disable-norm', action='store_true',  help='dont normalize vectors')
  parser.add_argument('-r', '--rand', action='store_true',  help='random vectors')
  parser.add_argument('-s', '--save', action='store_true',  help='save imgs')
  args = parser.parse_args()

  nk = args.batchsize
  nq = args.queries
  save_png = args.save
  normalize_inputs = not args.disable_norm

  f = gzip.open('./MLP-Python/mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(
    f, encoding='bytes')
  f.close()

  if args.rand:
    ki = np.random.randint(0, high=x.shape[0], size=nk)
    qi = np.random.randint(0, high=x.shape[0], size=nq)
  else:
    ki = np.arange(nk)
    qi = np.arange(90, 90+nq) # some overlap

  K = train[0][ki, :].astype(np.float32)
  Q = train[0][qi, :].astype(np.float32)

  if len(K.shape) < 2:
    K = np.expand_dims(K, axis=0)
  if len(Q.shape) < 2:
    Q = np.expand_dims(Q, axis=0)

  if save_png:
    show_imgs(K, 'K.png', ki)
    show_imgs(Q, 'Q.png', qi)

  if normalize_inputs:
    for i in range(Q.shape[0]):
      Q[i] = normalize(Q[i])
    for i in range(K.shape[0]):
      K[i] = normalize(K[i])

  np.set_printoptions(
    formatter={'float': '{: 0.3f}'.format},
    threshold=np.inf)

  # Q = [num_queries, d]
  # K = [num_keys, d]

  # Z = [num_queries, num_keys]
  # [... x d] * [d x ....]

  t0 = time.time()
  Z = np.dot(Q, K.T)
  t1 = time.time()

  debug_print(Z, 'Z')

  idxs = np.argsort(Z, axis=1)

  z = np.zeros_like(Z)
  for ii in range(Z.shape[0]): z[ii, :] = Z[ii, idxs[ii]]

  top_k = 3
  maxvals, maxidxs = z[:, -top_k:], idxs[:, -top_k:]

  print(maxidxs)
  debug_print(maxvals, 'maxvals')

  print('dot(Q,K.T): t={:6.3f} ms'.format(1000.0 * (t1-t0)))

  if save_png:
    save_arr(z, 'z_sorted.png')
    save_arr(Z, 'Z.png')

  # softmax
  # [nq, ] -> [nq, 1]
  #m0 = np.expand_dims(maxvals[:, -1], axis=1)
  #z0 = Z - m0 # [nq, nk] - [nq, 1]
  #p0 = np.exp(z0)
  #s0 = np.sum(p0, axis=1)
  ## [nq, ] -> [nq, 1]
  #s0 = np.expand_dims(s0, axis=1)
  ## normalized
  #ps = p0 / s0

  #debug_print(m0, 'm0')
  #debug_print(z0, 'z0')
  #debug_print(p0, 'p0')
  #debug_print(s0, 's0')
  #debug_print(ps, 'ps')
