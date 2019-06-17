from estnltk import synthesize, Text

print(synthesize('kass', 'sg p'))

# input = open("../data/ud-treebanks-v2.1/UD_EstonianNltk2/etn2-ud-dev.conllu", "r")
input = open("./et_ud_estonian.test.pred.postest.final.conllu", "r")
output = open("combining,stanford.estnltk.conllu", "w+")
newLineSplit = []
for line in input:
    if "# " in line:
        print(line, file=output)
        continue
    splits = line.split('\t')
    if len(splits) < 10:
        print("", file=output)
        continue
    print(splits[2] + " " + splits[5])

    morph = ' '.join([x.split("=")[1] if "=" in x else x for x in splits[5].split("|")])
    morph_split = morph.split(" ")
    if len(morph_split) > 1 and (morph_split[1] == "sg" or morph_split[1] == "pl"):
        morph = morph_split[1] + " " + morph_split[0]

    syn = synthesize(splits[2], morph, partofspeech=splits[4])
    word = splits[2]
    print(syn)
    if len(syn) > 0 and syn[0] != "_":
        word = syn[0]

    tenth = splits[9].strip("\n")

    newLineSplit = []
    newLineSplit.append(splits[0])
    newLineSplit.append(word)
    newLineSplit.append(splits[2])
    newLineSplit.append(splits[3])
    newLineSplit.append(splits[4])
    newLineSplit.append(splits[5])
    newLineSplit.append(splits[6])
    newLineSplit.append(splits[7])
    newLineSplit.append(splits[8])
    newLineSplit.append(tenth)
    print('\t'.join([str(x) for x in newLineSplit]), file=output)




