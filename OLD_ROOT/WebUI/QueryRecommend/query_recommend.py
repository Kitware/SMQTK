"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
import os
import numpy as np
import json

thispath = os.path.dirname(os.path.abspath(__file__))

# commonly used words in event descriptions
additional_stop_words = ['event', 'name', 'explication', 'evidential', 'description', 'scene',
                         'objects', 'people', 'activities', 'audio']

# zero-shot queries for E006 ~ E015
queries = dict()
queries['E000'] = ''
queries['E006'] = 'sc.person sc.enclosed_area sc.electric_or_indoor_lighting sc.has_audience sc.congregating' \
                  ' ob.light_source ob.person'
queries['E007'] = 'sc.transporting sc.manmade sc.using_tools sc.asphalt ob.round_shape ob.car'
queries['E008'] = 'sc.congregating sc.has_audience ob.person sc.pavement' \
                  ' ob.large_group_of_people ob.crowd ob.small_group_of_people ob.railing ob.floor'
queries['E009'] = 'sc.dirty sc.natural_light sc.natural ob.large_open_area sc.sunny sc.trees' \
                  ' ob.truck ob.car ob.large_open_area ob.outdoor'
queries['E010'] = 'sc.working sc.dirty sc.enclosed_area'
queries['E011'] = 'sc.enclosed_area sc.wood_not_part_of_tree sc.electric_or_indoor_lighting'
queries['E012'] = 'sc.congregating sc.has_audience sc.asphalt sc.pavement' \
                  ' ob.person ob.large_group_of_people ob.tree ob.sports_venue ob.crowd' \
                  ' ob.small_group_of_people ob.railing ob.floor'
queries['E013'] = 'sc.asphalt sc.trees sc.natural_light sc.open_area' \
                  ' ob.large_open_area ob.tree ob.rectangular_shape ob.door'
queries['E014'] = 'sc.using_tools sc.working sc.learning ob.round_shape'
queries['E015'] = 'sc.person sc.enclosed_area sc.electric_or_indoor_lighting'
queries['E021'] = 'sc.trees sc.vegetation sc.natural sc.open_area sc.pavement sc.asphalt sc.natural_light' \
                  ' ob.tree ob.large_open_area ob.cloud ob.outdoor ob.sports_venue ob.sky ob.truck '
queries['E022'] = 'sc.learning sc.working sc.enclosed_area sc.dirty sc.using_tools sc.electric_or_indoor_lighting'
queries['E023'] = 'sc.asphalt sc.pavement sc.clouds' \
                  ' ob.cloud ob.small_group_of_people ob.floor ob.sports_venue ob.railing'
queries['E024'] = 'sc.transporting sc.asphalt sc.trees sc.pavement ob.rectangular_shape ob.door'
queries['E025'] = 'sc.person ob.small_group_of_people ob.vertical_pattern'
queries['E026'] = 'sc.wood_not_part_of_tree sc.enclosed_area sc.working sc.using_tools sc.dirty' \
                  ' ob.door ob.vertical_pattern ob.rectangular_shape ob.railing '
queries['E027'] = 'sc.natural sc.dirty sc.open_area sc.trees sc.natural_light' \
                  ' ob.large_group_of_people ob.tree ob.outdoor ob.vertical_pattern ob.crowd ob.person '
queries['E028'] = 'sc.person sc.has_audience sc.enclosed_area ob.rectangular_shape ob.crowd'
queries['E029'] = 'sc.sunny sc.still_water sc.open_area sc.pavement sc.trees sc.manmade sc.asphalt' \
                  ' ob.large_open_area ob.sports_venue ob.outdoor ob.horizontal_pattern'
queries['E030'] = 'sc.using_tools sc.working sc.dirty ob.railing ob.floor ob.face'


def read_words(_words):
    words = []
    with open(_words, 'r') as fid_stop_words:
        for line in fid_stop_words:
            if line[-1]=='\n':
                line = line[:-1]
            if line != '':
                words.append(line)
    return words


def preprocess(string, stop_words=None, special_char=None):
    if stop_words is None:
        _stop = thispath + '/stop_words.txt'
        stop_words = read_words(_stop)
    if special_char is None:
        _special = thispath + '/special_characters.txt'
        special_char = read_words(_special)

    string = string.lower()
    string = string.replace('\n', ' ')
    string = string.replace('\t', ' ')
    for schar in special_char:
        string = string.replace(schar.decode("utf8"), '')

    words = string.split(' ')
    words_out = []
    for w in words:
        if not (w in stop_words) and len(w) > 0:
            words_out.append(w)
    return words_out


def generate_bow(string, dictionary):
    bow = np.zeros(len(dictionary))
    words = preprocess(string)
    for w in words:
        try:
            bow[dictionary[w]] += 1
        except KeyError:
            # A word doesn't exist in the dictionary, so ignore it.
            continue
    if np.sum(bow) > 0:
        bow /= np.sum(bow)
    return bow


def build_dictionary():
    _stop = thispath + '/stop_words.txt'
    _special = thispath + '/special_characters.txt'

    stop_words = read_words(_stop) + additional_stop_words
    special_char = read_words(_special)

    words = []
    for eid in range(6, 16) + range(21, 31):
        string = ""
        with open('./eventtexts/E%03d.txt' % eid, 'r') as fid_event:
            for line in fid_event:
                string += line
        words += preprocess(string, stop_words, special_char)
    words = sorted(list(set(words)))
    dictionary = dict()
    for idx, w in enumerate(words):
        dictionary[w] = idx
    np.save('dictionary_event_description.npy', dictionary)


def generate_event_bow():
    dictionary = np.load(thispath + '/dictionary_event_description.npy').item()
    for eid in range(6, 16) + range(21, 31):
        string = ""
        with open(thispath + '/eventtexts/E%03d.txt' % eid, 'r') as fid_event:
            for line in fid_event:
                string += line
        bow_eid = generate_bow(string, dictionary)
        np.save(thispath + '/eventbow/E%03d.npy' % eid, bow_eid)


def recommend_query(string):
    '''
    Return zero-shot queries based on event description
    @param string: Event description in a string format
    @return: Queries in a string format
    '''
    dictionary = np.load(thispath + '/dictionary_event_description.npy').item()
    bow = generate_bow(string, dictionary)
    min_dist = 1
    detected_eid = 0    # if description matching fails, it will return an empty query.
    for eid in range(6, 16) + range(21, 31):
        bow_eid = np.load(thispath + '/eventbow/E%03d.npy' % eid)
        dist = np.sqrt(np.sum((bow - bow_eid)**2))
        if min_dist > dist:
            min_dist = dist
            detected_eid = eid
    return queries['E%03d' % detected_eid]


if __name__ == '__main__':
    # build_dictionary()
    # generate_event_bow()
    string = 'AExplication: Bikes are normally ridden with a person sitting down on ' \
             'seat and holding onto the handlebars and steering with their hands. ' \
             'Tricks consist of difficult ways of riding the bike, such as on ' \
             'one wheel, steering with feet or standing on the seat; or intentional ' \
             'motions made with the bike that are not simply slowing down/stopping ' \
             'the bike, propelling it forward, or steering the bike as it'
    q = recommend_query(string)
    print q
