import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords as st
from nltk.stem import WordNetLemmatizer

###creating enron test file

t1=time.time()      #for checking run time
l=WordNetLemmatizer()      #creating l
s=list(set(st.words("english")))    #creating english stop words
s.append('.');s.append(',');  #adding , . to stop words
t1=open('Enron.train','r')
t2=open('enrontestfile','w')
lib=nltk.FreqDist('')   #creating blank freqency distribution
for i in t1:
    i=i.split()   
    t=i[0].strip(',')
    t2.write(t)
    t2.write(': ') 
    data=wt((' ').join(i[2:]).lower())
    lib=nltk.FreqDist('')
    for w in data:
        w=l.lemmatize(w,pos='a')   
        if w not in s:
            lib[w]+=1
    for i in lib.most_common(len(lib)):     #not to write same word again
        t2.write(i[0])
        t2.write(' ')        #seperating by space in final file
    print >>t2
print 'time taken:{}'.format(time.time()-t1)
