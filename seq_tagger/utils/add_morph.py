# -*- coding: utf-8 -*-
#
#   Analyses JSON format data, adding layers of analyses up to the level
#   of morphological analysis (incl).
#   Saves results as EstNLTK 1.6.0 JSON export files (recommended), or
#   as pickle Text objects.
#   This script is for processing large corpora -- Koondkorpus and
#   etTenTen -- with EstNLTK.
#
#   Developed and tested under Python's version:  3.5.4
#                              EstNLTK version:   1.6.0_beta
#
#

import sys, codecs
import os, os.path
import json
import argparse

from datetime import datetime
from datetime import timedelta

from estnltk import Text


skip_existing = True  # skip existing files (continue analysis where it stopped previously)
skip_saving = False  # skip saving (for debugging purposes)
record_sentence_fixes = True  # record types of sentence postcorrections (for debugging purposes)
add_metadata = True  # add metadata to Text
add_syntax_ignore = True  # add layer 'syntax_ignore'
add_gt_morph_analysis = True  # add layer 'gt_morph_analysis'

input_ext = '.txt'  # extension of input files
corpus_type = 'ettenten'  # 'koond' or 'ettenten'
output_format = 'json'  # 'json' or 'pickle'

skip_list = []  # files to be skipped (for debugging purposes)


# =======  Helpful utils

def write_error_log(fnm, err):
    ''' Writes information about a processing error into
        file "__errors.txt".
    '''
    err_file = "__errors.txt"
    if not os.path.exists(err_file):
        with open(err_file, 'w', encoding='utf-8') as f:
            pass
    with open(err_file, 'a', encoding='utf-8') as f:
        f.write('{} :'.format(datetime.now()) + ' ' + str(fnm) + ' : ')
        f.write("{0}".format(err) + '\n\n')


def load_in_file_names(fnm):
    ''' Loads names of the input files from a text file.
        Each name should be on a separate line.
    '''
    filenames = []
    with open(fnm, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                filenames.append(line)
    return filenames


# =======  Parse input arguments

arg_parser = argparse.ArgumentParser(description= \
                                         '''  Analyses JSON format text data with EstNLTK 1.6.0, adding layers of 
                                              analyses up to the level of morphological analysis (incl).
                                              Saves results as EstNLTK JSON export files (recommended), or 
                                              as pickle Text objects.
                                              This script is for processing large corpora -- Koondkorpus and 
                                              etTenTen -- with EstNLTK.
                                         ''')
arg_parser.add_argument('in_dir', default=None, \
                        help='the directory containing input files; ' + \
                             'Input files should be JSON files, in binary ' + \
                             'format, containing only ASCII symbols (non-ASCII ' + \
                             'characters escaped). Analysable text should be ' + \
                             'under the key "text", and other keys may contain ' + \
                             'metadata about the text. Basically, the input files ' + \
                             'should be JSON files created by EstNLTK 1.4.1 method ' + \
                             'write_document (see ' + \
                             'https://github.com/estnltk/estnltk/blob/1.4.1/estnltk/corpus.py#L80' + \
                             ' for details)'
                        )
arg_parser.add_argument('--in_files', default=None, \
                        help='a text file containing names of the input ' + \
                             'files (files from in_dir) that should be ' + \
                             'analysed. File names should be separated by newlines. ' + \
                             'Use this argument to specify a subset of files to be ' + \
                             'analysed while parallelizing the analysis process. ' + \
                             'You can use the script "split_large_corpus_files_into_subsets.py" ' + \
                             'for splitting the input corpus into subsets of files.')
arg_parser.add_argument('out_dir', default=None, \
                        help='the output directory where the results ' + \
                             'of analysis (one output file per each input file) ' + \
                             'will be written.')
arg_parser.add_argument('--koond', default=False,
                        help='If set, then expects that the input files come ' + \
                             'from Koondkorpus (and applies Koondkorpus-specific meta-' + \
                             'data acquisition methods). Otherwise, assumes that input ' + \
                             'files come from etTenTen.', \
                        action='store_true')
arg_parser.add_argument('--pickle', default=False,
                        help='If set, then changes the output format from json (which is ' + \
                             'default) to pickle. Note that pickle files take up more space ' + \
                             'and therefore the recommended format for processing large ' + \
                             'corpora is json.', \
                        action='store_true')

args = arg_parser.parse_args()
in_dir = args.in_dir if os.path.isdir(args.in_dir) else None
out_dir = args.out_dir if os.path.isdir(args.out_dir) else None
in_files = args.in_files if args.in_files and os.path.isfile(args.in_files) else None
if args.in_files and not os.path.isfile(args.in_files):
    print(' Unable to load input from', in_files, '...')
output_format = 'pickle' if args.pickle == True else 'json'
corpus_type = 'koond' if args.koond == True else 'ettenten'

if out_dir and in_dir:
    assert corpus_type and corpus_type.lower() in ['koond', 'ettenten']
    assert output_format and output_format.lower() in ['json', 'pickle']
    # =======  Collect input files
    print(' Type of the input corpus: ', corpus_type)
    if not in_files:
        all_files = os.listdir(in_dir)
    else:
        print(' Loading input from', in_files, '...')
        all_files = load_in_file_names(in_files)
    print('*' * 70)
    print(' Found', len(all_files), ' files.')
    # =======  Initialize
    startTime = datetime.now()
    elapsed = 0
    errors = 0
    skipped = 0
    processed = 0
    new_files = 0
    output = open("etn2-ud2.conllu", "w+")
    for in_file_name in all_files:
        fnm = os.path.join(in_dir, in_file_name)
        # skip dirs and non-input files
        if os.path.isdir(fnm):
            continue
        if not fnm.endswith(input_ext):
            continue
        if in_file_name in skip_list:
            continue
        # construct output file, and check whether it already exists
        out_file_name_pckl = in_file_name.replace('.txt', '.pickle')
        out_file_name_json = in_file_name.replace('.txt', '.json')
        ofnm_pckl = os.path.join(out_dir, out_file_name_pckl)
        ofnm_json = os.path.join(out_dir, out_file_name_json)
        if skip_existing:
            if os.path.exists(ofnm_pckl):
                print('(!) Skipping existing file:', ofnm_pckl)
                processed += 1
                skipped += 1
                continue
            if os.path.exists(ofnm_json):
                print('(!) Skipping existing file:', ofnm_json)
                processed += 1
                skipped += 1
                continue
        # Load input text from JSON
        text_dict = None
        with codecs.open(fnm, 'rb', 'ascii') as f:
            text_dict = json.loads(f.read())
        if 'text' in text_dict:
            text = Text(text_dict['text'])
            if output_format == 'pickle':
                print(processed, '->', ofnm_pckl)
            else:
                print(processed, '->', ofnm_json)
            for sentence_string in text.sentence_texts:
                sentence = Text(sentence_string)
                if len(sentence.words) <= 1:
                    continue
                for i, w in enumerate(sentence.words):
                    word = w['text']
                    lemma = sentence.lemmas[i] if sentence.lemmas[i] != "" else "_"
                    form = sentence.forms[i] if sentence.forms[i] != "" else "_"
                    postag = sentence.postags[i] if sentence.postags[i] != "" else "_"
                    if "|" in form:
                        form = form.split("|")[0]
                    if "|" in postag:
                        postag = postag.split("|")[0]
                    line = []
                    line.append(str(i + 1))
                    line.append(word)
                    line.append(lemma)
                    line.append(postag)
                    line.append(postag)
                    line.append(form)
                    line.append("_")
                    line.append("_")
                    line.append("_")
                    line.append("_")
                    print('\t'.join([str(x) for x in line]), file=output)
                print(file=output)

else:
    arg_parser.print_help()



