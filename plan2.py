import readline
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import time

class DR:
    def __init__(self):
        self.transcript_name = []
        self.genome = []
        self.tr_name = []
        self.tr_point = []
        self.reads_point = []
        self.assigned = []
        self.tr_len = []
        self.kmer = 11
        self.table = {'A':0, 'a':0, 'C':1, 'c':1, 'G':2, 'g':2, 'T':3, 't':3}
    def getDecoys(self, filename):
        with open(filename, 'r') as dcy:
            tmp = dcy.readline()[:-1]
            while tmp:
                self.tr_name.append(tmp)
                tmp = dcy.readline()
    def getGentrome(self, filename):
        with open(filename, 'r') as gentrome:
            print("getGentrome()")
            tmp = gentrome.readline()
            tr = []
            dim = np.zeros(4**self.kmer)
            while tmp:
                if tmp[0] == '>':
                    tmp = gentrome.readline()
                    tr = ''.join(tr)
                    self.tr_len.append(len(tr)/1000)
                    for i in range(len(tr)-self.kmer+1):
                        if 'N' in tr[i:i+self.kmer]:
                            continue
                        index = self.table[tr[i]]*16 + self.table[tr[i+1]]*4 + self.table[tr[i+2]]
                        dim[index] += 1
                    self.tr_point.append(dim)
                    tr = []
                    dim = np.zeros(4**self.kmer)
                    continue
                else:
                    tr.append(tmp[:-1])
                    tmp = gentrome.readline()

            tr = ''.join(tr)
            self.tr_len.append(len(tr)/1000)
            for i in range(len(tr)-self.kmer+1):
                if 'N' in tr[i:i+self.kmer]:
                        continue
                index = self.table[tr[i]]*16 + self.table[tr[i+1]]*4 + self.table[tr[i+2]]
                dim[index] += 1
            self.tr_point.append(dim)

            del self.tr_point[0]
            del self.tr_len[0]
            self.assigned = np.zeros(len(self.tr_point))

    def getFA(self):
        for filename in os.listdir("/mammoth/flux_simulator_data/Drosophila_melanogaster.BDGP6.80_refgenome"):
            with open(os.path.join("/mammoth/flux_simulator_data/Drosophila_melanogaster.BDGP6.80_refgenome", filename), 'r') as fa:
                print("getFA():", filename)
                fa_header = fa.readline()
                tmp = fa.readline()
                tmp = tmp[:-1]
                line = []
                line.append(tmp)
                while tmp:
                    tmp = fa.readline()
                    tmp = tmp[:-1]
                    line.append(tmp)
                line = ''.join(line)
                self.genome.append(line)
    def getGFF3(self):
        cnt = 0
        for filename in os.listdir("/mammoth/flux_simulator_data/Drosophila_melanogaster.BDGP6.80_reftranscript"):
            with open(os.path.join("/mammoth/flux_simulator_data/Drosophila_melanogaster.BDGP6.80_reftranscript", filename), 'r') as gff3:
                print("getGFF3():", filename)
                dim = np.zeros(4**self.kmer)
                string = []
                tmp = gff3.readline()
                while tmp:
                    li = tmp.split()
                    if li[2] == "transcript":
                        string = ''.join(string)
                        self.tr_len.append(len(string)/1000)
                        for i in range(len(string)-self.kmer+1):
                            if 'N' in self.genome[cnt][i:i+self.kmer]:
                                continue
                            index = self.table[self.genome[cnt][i]]*16 + self.table[self.genome[cnt][i+1]]*4 + self.table[self.genome[cnt][i+2]]
                            assert(index >=0 and index<=63)
                            dim[index] += 1
                        self.tr_point.append(dim)
                        dim = np.zeros(4**self.kmer)
                        string = []
                    elif li[2] == "exon":
                        string.append(self.genome[cnt][int(li[3]) : int(li[4])+1])
                    else:
                        pass
                    tmp = gff3.readline()
            cnt += 1 
    def getReads(self, filename):
        with open(filename, 'r') as reads:
            print("getReads():", filename)
            re = reads.readline()[:-1]
            while re:
                if re[0] == '>':
                    re = reads.readline()[:-1]
                dim = np.zeros(4**self.kmer)
                for i in range(len(re)-self.kmer+1):
                    if 'N' in re[i:i+self.kmer]:
                        continue
                    index = self.table[re[i]]*16 + self.table[re[i+1]]*4 + self.table[re[i+2]]
                    assert(index >=0 and index<=63)
                    dim[index] += 1
                self.reads_point.append(dim)
                re = reads.readline()
                re = re[:-1]
    def reduction(self):
        print("reduction()")
        x = np.concatenate((self.tr_point, self.reads_point))
        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(x)
        tr = np.array(principalComponents[:len(self.tr_point)])
        reads = np.array(principalComponents[len(self.tr_point):])
        return tr, reads

    def calcDistance(self, tr, reads):
        max_reads = 10**4
        for i in range(len(reads) // max_reads + 1):
            np.add.at(self.assigned, np.argmin(np.linalg.norm(tr[None, :, :] - reads[i*max_reads:(i+1)*max_reads, None, :], axis=-1), axis=-1), 1)

    def calcTPM(self):
        self.denominator = np.sum(np.array(self.assigned*10**-6)/np.array(self.tr_len))
        self.assigned = self.assigned / self.tr_len / self.denominator

if __name__ == "__main__":
    start = time.time()
    dr = DR()
    dr.getGentrome("/mammoth/flux_simulator_data/droso_gentrome.fa")
    dr.getReads("/mammoth/flux_simulator_data/droso_1millionreads/droso_1million.fasta")
    tr, reads = dr.reduction()
    dr.calcDistance(tr, reads)
    dr.calcTPM()
    dr.getDecoys("/mammoth/flux_simulator_data/decoys.txt")
    dr.transcript_name = dict(zip(dr.tr_name, dr.assigned))
    with open("/mammoth/flux_simulator_data/droso_1millionreads/plan2_TPM.txt", 'w') as f:
        keys = list(dr.transcript_name.keys())
        values = list(dr.transcript_name.values())
        for i in range(len(keys)):
            mes = str(values[i]) + "\n"
            f.write(mes)
    end = time.time()
    print("it takes (sec)", end - start)