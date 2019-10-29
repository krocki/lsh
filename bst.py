#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" krocki @ 10/29/19 """

class BSTNode:
  val,left,right=None,None,None
  def __init__(self, val):
    self.val=val

def insert(root, val):
  if root == None:
    root=BSTNode(val)
  else:
    if root.val==val:
      print('duplicate value')
    else:
      if val<root.val:
        root.left=insert(root.left,val)
      if val>root.val:
        root.right=insert(root.right,val)

  return root

def printtree(root):
  if root:
    printtree(root.left)
    print(root.val)
    printtree(root.right)

if __name__ == "__main__":
  root=None
  root=insert(root, 9)
  root=insert(root, 7)
  root=insert(root, 4)
  root=insert(root, 6)
  root=insert(root, 1)
  root=insert(root, 5)
  root=insert(root, 8)
  root=insert(root, 3)
  root=insert(root, 2)
  print('hopefully this in ascending order')
  printtree(root)
