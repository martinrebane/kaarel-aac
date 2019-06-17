gold = open("./et_ud_estonian.test.in.conllu", "r")
output = open("et_ud_estonian.test.gold.lower.conllu", "w+")

i = 0
j = 0
for line in gold:
    if "# text" in line or "# newdoc" in line:
        print(line.strip("\n"), file=output)
        continue
    gold_split = line.split("\t")
    if len(gold_split) > 2:
        newLineSplit = []
        newLineSplit.append(gold_split[0])
        newLineSplit.append(gold_split[1])

        newLineSplit.append(gold_split[2].lower())
        newLineSplit.append(gold_split[3])
        newLineSplit.append(gold_split[4])
        newLineSplit.append(gold_split[5])
        newLineSplit.append(gold_split[6])
        newLineSplit.append(gold_split[7])
        newLineSplit.append(gold_split[8])
        newLineSplit.append(gold_split[9].strip("\n"))
        print('\t'.join([str(x) for x in newLineSplit]), file=output)
    else:
        print(file=output)




