import csv
import json

score_file = 'run16_sha_scores.csv'
matches_file = 'cp1eval/CP1_clusters_ads_images.csv'

matches = list(csv.reader(open(matches_file)))[1:]
scores = list(csv.reader(open(score_file)))[1:]

cluster_map = {}

for match in matches:
    images = cluster_map.get(match[0], [])
    images.append(match[2])
    cluster_map[match[0]] = images

score_map = {}

for score in scores:
    score_map[score[0].split('/')[1]] = float(score[2])

cluster_scores = []

for cluster, images in cluster_map.iteritems():
    cur_scores = []
    for image in images:
        if image in score_map:
            cur_scores.append(score_map[image])
    if len(cur_scores) > 0:
        # agg_score = sum(cur_scores) / len(cur_scores)
        agg_score = max(cur_scores)
        cluster_scores.append({'cluster_id': cluster, 'score': agg_score})

for cluster_score in cluster_scores:
    print json.dumps(cluster_score)
