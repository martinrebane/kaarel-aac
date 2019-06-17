from estnltk import Text
import regex

# sets = ["train"]

ud = {
    'ab': 'Case=ab',
    'abl': 'Case=abl',
    'ad': 'Case=ad',
    'adt': 'Case=adt',
    'all': 'Case=all',
    'el': 'Case=el',
    'es': 'Case=es',
    'g': 'Case=g',
    'ill': 'Case=ill',
    'in': 'Case=in',
    'kom': 'Case=kom',
    'n': 'Case=n',
    'p': 'Case=p',
    'ter': 'Case=ter',
    'tr': 'Case=tr',
    'b': 'VerbForm=b',
    'd': 'VerbForm=d',
    'da': 'VerbForm=da',
    'des': 'VerbForm=des',
    'ge': 'VerbForm=ge',
    'gem': 'VerbForm=gem',
    'gu': 'VerbForm=gu',
    'ks': 'VerbForm=ks',
    'ksid': 'VerbForm=ksid',
    'ksime': 'VerbForm=ksime',
    'ksin': 'VerbForm=ksin',
    'ksite': 'VerbForm=ksite',
    'ma': 'VerbForm=ma',
    'maks': 'VerbForm=maks',
    'mas': 'VerbForm=mas',
    'mast': 'VerbForm=mast',
    'mata': 'VerbForm=mata',
    'me': 'VerbForm=me',
    'V|n': 'VerbForm=n',
    'nud': 'VerbForm=nud',
    'nuks': 'VerbForm=nuks',
    'nuksid': 'VerbForm=nuksid',
    'nuksime': 'VerbForm=nuksime',
    'nuksin': 'VerbForm=nuksin',
    'nuksite': 'VerbForm=nuksite',
    'nuvat': 'VerbForm=nuvat',
    'o': 'VerbForm=o',
    's': 'VerbForm=s',
    'sid': 'VerbForm=sid',
    'sime': 'VerbForm=sime',
    'sin': 'VerbForm=sin',
    'site': 'VerbForm=site',
    'ta': 'VerbForm=ta',
    'tagu': 'VerbForm=tagu',
    'taks': 'VerbForm=taks',
    'takse': 'VerbForm=takse',
    'tama': 'VerbForm=tama',
    'tav': 'VerbForm=tav',
    'tavat': 'VerbForm=tavat',
    'te': 'VerbForm=te',
    'ti': 'VerbForm=ti',
    'tud': 'VerbForm=tud',
    'tuks': 'VerbForm=tuks',
    'tuvat': 'VerbForm=tuvat',
    'v': 'VerbForm=v',
    'vad': 'VerbForm=vad',
    'vat': 'VerbForm=vat',
    'neg': 'Polarity=neg',
    'pl': 'Number=pl',
    'sg': 'Number=sg',
}
mode = "all"
sets = ["dev","test"]
# output = open("7et_ud_estonian.train.in.conllu", "w+")

for set in sets:
    input = open("../data/ud-treebanks-v2.1/UD_Estonian/et-ud-" + set + ".conllu", "r")
    # input = open("./et-common_crawl-000.conllu", "r")
    output = open("et_ud_estonian." + set + ".in.conllu", "w+")

    text = Text('')
    i = 0

    whole_sentence = []
    write_whole_sentence = True
    for line in input:

        if "# sent_id" in line:
            continue
        if '# text' in line:
            if write_whole_sentence:
                print('\n'.join([str(x) for x in whole_sentence]), file=output)
            whole_sentence = []
            write_whole_sentence = True
            m = regex.search('# text = (.*)', line)
            text_string = m.group(1)
            regex_string = '([.\"\”\“\)\%\;\?\(\:\,\!\]\…\»]{2}|[0-9][.]|o\.|-,|t\.)'
            m2 = regex.findall(regex_string, text_string)
            for replaceable in m2:
                text_string = text_string.replace(replaceable, replaceable[0] + ' ' + replaceable[1])

            m2 = regex.findall(regex_string, text_string)
            while len(m2) > 0:
                for replaceable in m2:
                    text_string = text_string.replace(replaceable, replaceable[0] + ' ' + replaceable[1])
                m2 = regex.findall(regex_string, text_string)

            text = Text(text_string.replace('le.', 'le .').replace('ni.', 'ni .').replace('ne.', 'ne .').replace('el.', 'el .')
                        .replace('C.', 'C .').replace('aastased.', 'aastased .').replace('kordsed.', 'kordsed .')
                        .replace('ks.', 'ks .').replace('ga.', 'ga .').replace('h.', 'h .').replace('-"', '- "')
                        .replace('i:', 'i :').replace('P.', 'P .').replace('(E', '( E'))
        elif not write_whole_sentence:
            continue


        split = line.split("\t")
        newLineSplit = []
        if len(split) == 10:
            i = int(split[0]) - 1
            # print(i)
            # print(split[0])
            # print(len(text.lemmas))
            # print(len(text.words))
            # print(text.words[i]['text'])
            if i >= len(text.forms):
                write_whole_sentence = False
                continue

            nud_tud = False
            adj_nud_tud = False
            # FEATS
            form = text.forms[i]
            if form == "":
                form = "_"
            if split[1] == "seotud":
                print(split)

            if "|" in form:
                forms = form.split("|")
                if "nud" in forms and split[3] == "VERB":
                    nud_tud = True
                    form = "nud"
                elif "tud" in forms and split[3] == "VERB":
                    nud_tud = True
                    form = "tud"
                elif ("tud" in forms or "nud" in forms) and len(forms) > 1:
                    form = forms[0]
                    adj_nud_tud = True
                else:
                    form = forms[0]

            # POSTAGS
            postag = text.postags[i]
            postags = []
            if postag == "":
                postag = "_"
            if "|" in postag:
                postags = postag.split("|")
                if nud_tud and "V" in postags:
                    postag = "V"
                else:
                    postag = postags[0]

            # FEATST TO UFEATS
            forms = []
            for f in form.split(" "):
                if f == "_" or f == "?" or f == "":
                    forms.append("_")
                    continue
                if postag + "|" + form == "V|n":
                    forms.append(ud[postag + "|" + form])
                else:
                    forms.append(ud[f])
            form = '|'.join([str(x) for x in forms])
            if form == "":
                form = "_"

            # LEMMA
            lemma = text.lemmas[i]
            if len(lemma) > 1 and "|" in lemma:
                lemmas = lemma.split("|")
                if adj_nud_tud and len(lemmas) > 1:
                    lemma = lemmas[1]
                else:
                    lemma = lemmas[0]
            if lemma == "":
                write_whole_sentence = False
                continue

            # MISC
            tenth = split[9].strip("\n")
            word = text.words[i]['text']

            if mode == "all":
                newLineSplit.append(split[0])
                newLineSplit.append(word)
                newLineSplit.append(lemma)
                newLineSplit.append(split[3])
                newLineSplit.append(postag)
                newLineSplit.append(form)
                newLineSplit.append(split[6])
                newLineSplit.append(split[7])
                newLineSplit.append(split[8])
                newLineSplit.append(tenth)
            elif mode == "in":
                newLineSplit.append(split[0])
                newLineSplit.append("_")
                newLineSplit.append(lemma)
                # newLineSplit.append("_")
                # newLineSplit.append("_")
                # newLineSplit.append("_")
                newLineSplit.append(split[3])
                newLineSplit.append(postag)
                newLineSplit.append(form)
                newLineSplit.append(split[6])
                newLineSplit.append(split[7])
                newLineSplit.append(split[8])
                newLineSplit.append(tenth)
            elif mode == "mtrain":
                if lemma == "" or form == "":
                    print(i + 1)
                    print(lemma)
                    print(form)
                    write_whole_sentence = False
                    continue
                newLineSplit.append(i+1)
                newLineSplit.append(lemma)
                newLineSplit.append(form)
            elif mode == "mtest":
                newLineSplit.append(lemma)
            if (mode == "mtrain" or mode == "mtest") and i == 0:
                whole_sentence.append("")
            whole_sentence.append('\t'.join([str(x) for x in newLineSplit]))

        else:
            if mode != "mtrain" and mode != "mtest":
                whole_sentence.append(line.strip("\n"))
    print('\n'.join([str(x) for x in whole_sentence]), file=output)
output.close()
