#!/usr/bin/env python
"""
Record of how train/test images were split out of development phone/ad images
and labels.
"""
import csv
import json

phone2shas = json.load(open('ad-images.map.phone2shas.json'))
phone2label = json.load(open('ad-images.map.phone2label.json'))

phones_pos = [p for p in phone2label if phone2label[p] == 'positive']
phones_neg = [p for p in phone2label if phone2label[p] == 'negative']

# Split pos/neg phone numbers and image shas based on phone number child image counts
phone_pos_train = set(
    p
    for i, (p, shas) in enumerate(sorted([(p, phone2shas[p]) for p in phones_pos], key=lambda e: len(e[1])))
    if i % 3 != 2
)
phone_pos_test = set(
    p
    for i, (p, shas) in enumerate(sorted([(p, phone2shas[p]) for p in phones_pos], key=lambda e: len(e[1])))
    if i % 3 == 2
)
phone_neg_train = set(
    p
    for i, (p, shas) in enumerate(sorted([(p, phone2shas[p]) for p in phones_neg], key=lambda e: len(e[1])))
    if i % 3 != 2
)
phone_neg_test = set(
    p
    for i, (p, shas) in enumerate(sorted([(p, phone2shas[p]) for p in phones_neg], key=lambda e: len(e[1])))
    if i % 3 == 2
)

pos_train_shas = set(sha for p in phone_pos_train for sha in phone2shas[p])
neg_train_shas = set(sha for p in phone_neg_train for sha in phone2shas[p])
pos_test_shas = set(sha for p in phone_pos_test for sha in phone2shas[p])
neg_test_shas = set(sha for p in phone_neg_test for sha in phone2shas[p])

# Report statistics
print
print "Pos phone numbers:", len(phones_pos)
print "  for train:", len(phone_pos_train)
print "             %d SHA1 values" % len(pos_train_shas)
print "  for test :", len(phone_pos_test)
print "             %d SHA1 values" % len(pos_test_shas)
print
print "Neg phone numbers:", len(phones_neg)
print "  for train:", len(phone_neg_train)
print "             %d SHA1 values" % len(neg_train_shas)
print "  for test :", len(phone_neg_test)
print "             %d SHA1 values" % len(neg_test_shas)
print
print "Pos/Neg set SHA intersection:", len( (pos_train_shas | pos_test_shas) & (neg_train_shas | neg_test_shas) )
print "Pos train/test SHA intersection:", len( pos_train_shas & pos_test_shas )
print "Neg train/test SHA intersection:", len( neg_train_shas & neg_test_shas )
print

# Combined SHA1 values and labels to create test/train CSV files for each experiment 1 thru 4
# 1. Use the sha values associated to each phone number as is
with open('ad-images.method1.train.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_train_shas:
        w.writerow([sha, 'positive'])
    for sha in neg_train_shas:
        w.writerow([sha, 'negative'])
with open('ad-images.method1.test.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_test_shas:
        w.writerow([sha, 'positive'])
    for sha in neg_test_shas:
        w.writerow([sha, 'negative'])

# 2. Discard sha values that intersect with other categories
with open('ad-images.method2.train.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_train_shas.difference( neg_train_shas | pos_test_shas | neg_test_shas ):
        w.writerow([sha, 'positive'])
    for sha in neg_train_shas.difference( pos_train_shas | pos_test_shas | neg_test_shas ):
        w.writerow([sha, 'negative'])
with open('ad-images.method2.test.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_test_shas.difference( neg_test_shas | pos_train_shas | neg_train_shas ):
        w.writerow([sha, 'positive'])
    for sha in neg_test_shas.difference( pos_test_shas | pos_train_shas | neg_train_shas ):
        w.writerow([sha, 'negative'])

# 3. Keep intersecting images in test set
with open('ad-images.method3.train.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_train_shas.difference( neg_train_shas | pos_test_shas | neg_test_shas ):
        w.writerow([sha, 'positive'])
    for sha in neg_train_shas.difference( pos_train_shas | pos_test_shas | neg_test_shas ):
        w.writerow([sha, 'negative'])
with open('ad-images.method3.test.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_test_shas.difference( neg_train_shas | neg_test_shas ):
        w.writerow([sha, 'positive'])
    for sha in neg_test_shas.difference( pos_train_shas | pos_test_shas ):
        w.writerow([sha, 'negative'])

# 4. Keep intersecting images in train set
with open('ad-images.method4.train.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_train_shas.difference( neg_train_shas | neg_test_shas ):
        w.writerow([sha, 'positive'])
    for sha in neg_train_shas.difference( pos_train_shas | pos_test_shas ):
        w.writerow([sha, 'negative'])
with open('ad-images.method4.test.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for sha in pos_test_shas.difference( pos_train_shas | neg_train_shas | neg_test_shas ):
        w.writerow([sha, 'positive'])
    for sha in neg_test_shas.difference( neg_train_shas | pos_train_shas | pos_test_shas ):
        w.writerow([sha, 'negative'])
