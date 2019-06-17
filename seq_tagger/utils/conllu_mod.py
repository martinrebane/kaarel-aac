gold = open("./et_ud_estonian.test.in.conllu", "r")
system_file = open("./test.conllu", "r")
output = open("fpos.test.in.conllu", "w+")

system = system_file.readlines()

i = 0
j = 0
for line in gold:
    if "# text" in line or "# newdoc" in line:
        continue
    print(i)
    gold_split = line.split("\t")
    system_split = system[i].split("\t")
    print(system_split)
    if len(gold_split) > 2 and len(system_split) > 2:
        newLineSplit = []
        newLineSplit.append(gold_split[0])
        newLineSplit.append("_")
        newLineSplit.append(gold_split[2])
        newLineSplit.append("_")
        newLineSplit.append(system_split[2])
        newLineSplit.append(system_split[3].strip("\n"))
        newLineSplit.append(gold_split[6])
        newLineSplit.append(gold_split[7])
        newLineSplit.append(gold_split[8])
        newLineSplit.append(gold_split[9].strip("\n"))
        print('\t'.join([str(x) for x in newLineSplit]), file=output)
    else:
        print(file=output)
    i += 1




