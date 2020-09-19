#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Angular Locality Sensitive Hashing

import pprint
import numpy as np
import argparse
import time

# x - inputs
# n - num of input buckets
# h - num of hashes

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

def cosine_sim(v,w):
  return np.dot(v,w)/(np.linalg.norm(v) * np.linalg.norm(w))

def debug_print(x, s):
  print('\n-----')
  print(s + '.shape')
  print(x.shape)
  print('')
  # print(x)
  # indentation
  print("\t" + str(x).replace('\n','\n\t'))
  print('=====')

def lsh_hyperplane(x, h):

  p = np.random.randn(x.shape[-1], h).astype(np.float32)
  projections = np.dot(x, p)

  debug_print(p, 'p')
  debug_print(projections, 'projections')

  return (projections >= 0).astype(np.int32)

# x: inputs
# b: number of buckets
# h: number of hashes

def lsh_angular(x, b, h):

  assert b % 2 == 0
  rotations_shape = (x.shape[-1], h, b // 2)

  rr = np.random.randn(*rotations_shape).astype(np.float32)
  debug_print(rr, 'rr')
  rr = np.reshape(rr, [-1, h * b // 2])
  debug_print(rr, 'rr_reshaped')

  rv = np.dot(x, rr)
  debug_print(rv, 'rv')

  rv = np.reshape(rv, [-1, h, b // 2])
  debug_print(rv, 'reshape rv')

  rv = np.transpose(rv, (1, 0, 2))
  debug_print(rv, 'transpose rv')

  rv = np.concatenate([rv, -rv], axis=-1)
  debug_print(rv, 'concatenate')

  buckets = np.argmax(rv, axis=-1).astype(np.int32)
  debug_print(buckets, 'buckets')

  return buckets

def matching_buckets(x, y):
  return np.sum(x==y)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-N', type=int, default=3,  help='number of input vectors')
  parser.add_argument('-D', type=int, default=2,  help='input inner dim')
  parser.add_argument('-H', type=int, default=4,  help='number of hashes')
  parser.add_argument('-B', type=int, default=4,  help='number of buckets')

  args = parser.parse_args()
  N, D, H, B = args.N, args.D, args.H, args.B

  keys = np.random.uniform(size=(N, D)).astype(np.float32)
  query = np.random.uniform(size=(1, D)).astype(np.float32)

  vecs = np.concatenate([keys, query])
  hashes = lsh_angular(vecs, B, H).T
  h_keys, h_query = hashes[:-1], hashes[-1]
  debug_print(h_keys, 'h_keys')
  debug_print(h_query, 'h_query')

  sim = np.array([cosine_sim(query, v) for v in keys])

  debug_print(keys, 'keys')
  debug_print(query, 'query')
  #print('hyperplane')
  #hashes0 = lsh_hyperplane(vecs, O)
  #debug_print(hashes0, 'hashes hyperplane')
  scores = np.expand_dims(np.array([matching_buckets(x, h_query) for x in h_keys]), axis=1)

  debug_print(scores, 'scores')
  debug_print(sim, 'cosine sim')
  debug_print(np.concatenate([sim.T, scores.T]).T, 'csim, hsim')

