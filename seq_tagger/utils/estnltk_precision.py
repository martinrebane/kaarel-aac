from estnltk import synthesize, Text

print(synthesize('kass', 'sg p'))

# input = open("../data/ud-treebanks-v2.1/UD_EstonianNltk2/etn2-ud-dev.conllu", "r")
input = open("./et_ud_estonian.dev.in.conllu", "r")

feats_set = set()

total = 0
plausible = 0
correct = 0

for line in input:
    if "# " in line:
        continue
    splits = line.split('\t')

    if len(splits) == 10:
        lemma = splits[2]
        pos = splits[4]
        word = splits[1]
        feats = splits[5].split('|')

        feats_arr = []
        if len(feats) > 0 and "_" not in feats:
            for f in feats:
                feats_arr.append(f.split("=")[1])
            feats_str = ' '.join([str(x) for x in feats_arr])
            feats_set.add(feats_str)

            synthesized_lemmas = synthesize(lemma, feats_str, partofspeech=pos)
            if len(synthesized_lemmas) > 0:
                synthesized_lemma = synthesized_lemmas[0]
            else:
                synthesized_lemma = lemma
            if word.lower() in [x.lower() for x in synthesized_lemmas] or synthesized_lemma.lower() == word.lower():
                correct += 1
            elif feats_str != "":
                # print(synthesized_lemma + " = " + word + " / lemma=" + lemma + " / morph=" + feats_str)
                # print(synthesized_lemmas)
                plausible += 1
            # else:
            #     print(synthesized_lemma + " = " + word + " / lemma=" + lemma + " / morph=" + feats_str)
            #     print(synthesized_lemmas)
            total += 1
        elif "_" in feats:
            if lemma.lower() == word.lower():
                correct += 1
            else:
                print("lemma=" + lemma + " / should=" + word.lower())

            total += 1
        else:
            print('wat')

print(feats_set)

print(100.0 * correct / total)
print(100.0 * (correct + plausible) / total)