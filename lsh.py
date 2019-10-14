import pprint
import numpy as np
import matplotlib.pyplot as plt

# inner dim
d=2
# hash size
k=3
# count
N=100

pp = pprint.PrettyPrinter(indent=4)

def cosine_sim(v,w):
  return np.dot(v,w)/(np.linalg.norm(v) * np.linalg.norm(w))

def bool2str(x):
  return ''.join(x.astype(int).astype('str'))

vecs = np.random.randn(N,d)
rs = np.random.randn(k,d)

print("vecs")
print(vecs)

print("rs")
print(rs)

prods = np.dot(vecs, rs.T)
print("prods")
print(prods)

hashes = prods > 0.0
print("hashes")
str_hashes = [bool2str(x) for x in hashes]

d = {}
for s,x in zip(str_hashes, vecs):
  if s in d: d[s].append(x)
  else: d[s] = [x]

pp.pprint(d)

xs = []
ys = []
zs = []

for i, z in enumerate(d):
  vals = d[z]
  color = i
  print(vals, color)
  for v in vals:
    xs.append(v[0])
    ys.append(v[1])
    zs.append(i)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, axisbg="1.0")
ax.scatter(xs, ys, c=zs, alpha=0.8, edgecolors='none', s=30)
plt.title('points')
plt.show()

#for i in range(0, N):
#  for j in range(0, N):
#    csim = cosine_sim(vecs[i], vecs[j])
#    hsim = np.sum(hashes[i] == hashes[j])
#    print("{:2d}, {:2d}, cos={:6.3f}, hsim={:2d}".format(i,j,csim,hsim))
