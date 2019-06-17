from estnltk import synthesize, Text
notFoundMorphs = set()

print(synthesize('ettekanne', 'sg n'))

def syn(connlString, word, includeMorph=True):
    ret = ""
    if word == "<unk>":
        return word
    morphs = connlString.split("|")
    pos_tag = ""
    if len(morphs) > 0:
        pos_tag = morphs[0]
    finalMorph = ""
    secondBestMorph = ""
    finalMorph = ' '.join([str(x).split("=")[1] if "=" in str(x) else str(x) for x in morphs[1:]])
    # print(finalMorph)
    # for morph in morphs[1:]:
    #     if morph != "" and "_" not in morph:
    #         finalMorph = morph
    #         break
    #     elif morph != "" and "_" in morph:
    #         secondBestMorph = morph
    #     else:
    #         secondBestMorph = morph
    #
    # if finalMorph == "":
    #     finalMorph = secondBestMorph
    # if finalMorph == "":
    #     ret = word
    words = synthesize(word, finalMorph, partofspeech=pos_tag)
    if len(words) > 0:
        ret = "|".join([str(x) for x in words])
    else:
        ret = word
    if includeMorph:
        return ret + " (" + finalMorph + ")[" + pos_tag + "]"
    else:
        return ret.split("|")[0] if '|' in ret else ret
