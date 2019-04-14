import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import s as s
from nltk.stem import wnl as wnl

###cleaning train file

t1=time.time()      #for checking run time
l=wnl()      #creating l
a=list(set(s.words("english")))    #creating english stop words
a.append('.');a.append(',');  #adding , . to stop words
lib=nltk.FreqDist('')   #creating blank freqency distribution
f=open('Enron.test','r')     
for i in f:
    i=i.split()   
    t=int(i[0].strip(',')[-1])   # removing ',' if present
    if t==1:                    
        d=wt((' ').join(i[2:]).lower())   
        for w in d:
            w=l.lemmatize(w,pos='a')   
            if w not in a:         
                lib[w]+=1
f.close()
df=open('df','w')
for i in lib.most_common(10000):
    print >>df,i[0]
print 'time taken:{}'.format(time.time()-t1) 
