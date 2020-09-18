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

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def debug_print(x, s):
  print('\n-----')
  print(s + '.shape')
  print(x.shape)
  print('')
  # print(x)
  # indentation
  print("\t" + str(x).replace('\n','\n\t'))
  print('=====')

def lsh_angular(x, n, h):

  rot_size = n
  n_buckets = n
  rotations_shape = (x.shape[-1], h, rot_size // 2)

  print('lsh_angular():\nx={},\nn={}, h={}'.format(x, n, h))
  print('rotations_shape')
  print(rotations_shape)
  debug_print(x, 'x')

  r_rotations = np.random.randn(*rotations_shape).astype(np.float32)
  r_rotations = np.reshape(r_rotations, [-1, h * (rot_size // 2)])
  debug_print(r_rotations, 'r_rotations')

  rotated_vecs = np.dot(x, r_rotations)
  debug_print(rotated_vecs, 'rotated_vecs')

  rotated_vecs = np.reshape(rotated_vecs, [-1, h, rot_size // 2])
  debug_print(rotated_vecs, 'reshape rotated_vecs')

  rotated_vecs = np.transpose(rotated_vecs, (1, 0, 2))
  debug_print(rotated_vecs, 'transpose rotated_vecs')

  rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
  debug_print(rotated_vecs, 'concatenate')

  buckets = np.argmax(rotated_vecs, axis=-1).astype(np.int32)
  debug_print(buckets, 'buckets')

  return r_rotations

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-N', type=int, default=5, help='input size')
  parser.add_argument('-d', type=int, default=8, help='input dim')
  parser.add_argument('-I', type=int, default=4, help='number of buckets in')
  parser.add_argument('-O', type=int, default=3, help='number of buckets')

  args = parser.parse_args()
  N, d, I, O = args.N, args.d, args.I, args.O

  vecs = np.random.randn(N, d)

  hashes = lsh_angular(vecs, I, O)
