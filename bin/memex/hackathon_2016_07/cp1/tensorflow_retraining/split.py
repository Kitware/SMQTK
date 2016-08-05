import shutil
import pickle
import os
import itertools

parts = {
  'test': 'test',
  'train1': 'train',
  'train2': 'train',
  'train3': 'validate'
}

suffix = 'shas.pickle'

data_dir = 'cp1data'
image_dir = 'subset'

size = 5000

for group in ['pos', 'neg']:
  for prefix, part in parts.iteritems():
    shas = pickle.load(open('_'.join([prefix, group, suffix])))
    # for sha in itertools.islice(shas, size):
    for sha in shas:
      sha_bin = sha[:3]
      src = os.path.join(data_dir, group, sha_bin, sha)
      dst = os.path.join(image_dir, group, part + '_' + sha + '.jpg')
      shutil.copyfile(src, dst)
