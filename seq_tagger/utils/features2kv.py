"""
Usage: python morph_features2kv.py [--skip-tag] <feature_table_file>
"""

import sys

def read_feature_table(feature_table_file):
    feature_value2key_dict = {}
    for ln in open(feature_table_file):
        key, value = ln.rstrip().split(",")
        if value in feature_value2key_dict:
            assert feature_value2key_dict[value] == key
        else:
            feature_value2key_dict[value] = key
    return feature_value2key_dict

if len(sys.argv) == 3:
    assert sys.argv[1] == "--skip-tag"
    skip_tag = True
    feature_table_file = sys.argv[2]
elif len(sys.argv) == 2:
    skip_tag = False
    feature_table_file = sys.argv[1]

feature_value2key_dict = read_feature_table(feature_table_file)

def preprocess_analysis(anl):
    if "|?" in anl:
        anl = anl.replace("|?", "")
    return anl

def process_analysis(anal_str):
    anal_proc = preprocess_analysis(anal_str)
    if anal_proc == "_J_|sub|crd":
        return "POS=SCONJ"
    
    feature_kv_list = []
    for feature_value in anal_proc.split("|"):
       if feature_value not in feature_value2key_dict:
           raise ValueError("Feature '%s' not in table. Orig. analysis: '%s'; Processed analysis: '%s'" % (feature_value, anal_str, anal_proc))
       feature_key = feature_value2key_dict[feature_value]
       feature_kv = "%s=%s" % (feature_key, feature_value)
       feature_kv_list.append(feature_kv)
    a = "|".join(feature_kv_list)
    return a

for ln in sys.stdin:
    ln = ln.strip()
    if ln == "":
        print()
    else:
        items = ln.split("\t")
        token, analyses = items[0], items[1:]
        if skip_tag:
            analyses =  [analyses[0]] + [process_analysis(a) for a in analyses[1:]]
        else:
            analyses = [process_analysis(a) for a in analyses]
        print(token, *analyses, sep="\t")
       