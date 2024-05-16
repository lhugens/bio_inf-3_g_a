import sklearn 
import pandas

def file_to_seqs(filepath):
    file = open(filepath)
    lines = "".join(file.readlines())
    sequences = []
    seq = ""
    for line in lines.splitlines():
        if line.startswith(">sp"):
            sequences.append(seq)
            seq = ""
        else: 
            seq += line
    return sequences[1:]

print(file_to_seqs("./sequences/globin.fasta"))
print(file_to_seqs("./sequences/zincfinger.fasta"))