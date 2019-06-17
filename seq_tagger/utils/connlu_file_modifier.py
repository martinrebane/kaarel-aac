sets = ["pred"]

for set in sets:
    input = open("../../marmot/text.out.txt3", "r")

    output = open("et_ud_estonian." + set + ".in.conllu", "w+")

    for line in input:
        # if "#" in line:
        #     print(line, file=output, end="")
        #     continue
        split = line.split("\t")
        newLineSplit = []
        if len(split) == 8:
            newLineSplit.append(split[0])
            newLineSplit.append(split[2])
            newLineSplit.append(split[1])
            newLineSplit.append(split[3])
            newLineSplit.append(split[4])
            newLineSplit.append(split[5])
            newLineSplit.append("1")
            newLineSplit.append("_")
            newLineSplit.append("_")
            newLineSplit.append("_")
            if split[0] == "1":
                print(file=output)
            print('\t'.join([str(x) for x in newLineSplit]), file=output, end="\n")

    output.close()
