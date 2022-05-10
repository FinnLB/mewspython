import csv
import os
import sys

import sklearn
import pandas


def find_file(name, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == name:
                return str(os.path.join(root, file))


def remove_preceding_zeros(name: str):
    for i in range(len(name)):
        if name[i] != "0":
            return name[i:len(name)]


reader = csv.reader(open("MEWS_data/TRACE_Datensatz_transposed_220308.csv", "r"), delimiter="\t")
writer = csv.writer(open("data.csv", "w"))
head = reader.__next__()
head.append("doc")
writer.writerow(head)
for row in reader:
    doc_id = row[0]
    efile = find_file(remove_preceding_zeros(doc_id.replace("X", "_")) + ".txt", "MEWS_data/MEWS_Essays")
    if efile is not None:
        with open(efile, "r") as essay_file:
            row.append(essay_file.read())
            writer.writerow(row)
    else:
        print(doc_id)

if __name__ == '__main__':
    pass
