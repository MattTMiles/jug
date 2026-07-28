import numpy as np
import glob as glob

tims=glob.glob("*.tim")
list=[]
for tim in tims:
        timlines=open(tim).readlines()
        for line in timlines:
                line=line.strip('\n').split()
                for i in range(len(line)):
                        if(line[i]=='-sys' or ('NANOGrav' in tim and line[i]=='-f') or ('dr1dr2' in tim and line[i]=='-q')):
                                list.append('sed -i "s/'+line[i]+' '+line[i+1]+'/ -group '+line[i+1]+' '+line[i]+' '+line[i+1]+'/g" '+tim)


list=np.unique(list)
f=open("sedlist", "w")
for i in range(len(list)):
        f.write(list[i]+'\n')
f.close()
