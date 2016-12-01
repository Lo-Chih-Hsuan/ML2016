import os, sys
import string
import nltk
nltk.download('punkt')
nltk.download("stopwords")

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

from nltk import stem
#ps = PorterStemmer()
rxstem = stem.RegexpStemmer('er$|a$|as$|az$')
snowball = stem.snowball.EnglishStemmer()

if __name__ == '__main__':
    current_path= sys.argv[1]
    #current_path= os.getcwd()
    fout= open(sys.argv[2], 'w')
    test= open('check_index.csv', 'r')
    
    lines= []
    for line in open(os.path.join(current_path, 'title_StackOverflow.txt')):
        line= line[:-1]
        line= line.translate(string.maketrans("",""), string.punctuation)
        line= nltk.word_tokenize(line)
        line= [word for word in line if word.decode('utf-8') not in stopwords.words('english')]
        line= [snowball.stem(word.decode('utf-8')) for word in line]
        line= " ".join(line)
        lines.append(line)
    
    ###tfidf####
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf= True)
    X= vectorizer.fit_transform(lines)
    X= normalize(X)
    idf = vectorizer.idf_
    
    print(vectorizer.get_feature_names())
    print(X.shape)

    ###LSA####
    svd = TruncatedSVD(n_components=20)
    X= svd.fit_transform(X) 
    X= normalize(X)

    ###Kmeans###
    kmeans = KMeans(n_clusters=100).fit(X)
    print(kmeans.labels_.shape)
    
    ###test####
    first= False
    for line in test:
        if first != True:
            fout.write('ID,Ans\n')
            first= True
            second= True
            continue
        line= line.split()
        line= line[0].split(',')
        #print(line[0], line[1], line[2])

        if kmeans.labels_[int(line[1])] == kmeans.labels_[int(line[2])]:
            fout.write('{},1\n'.format(line[0]))
        else:
            fout.write('{},0\n'.format(line[0]))
          
            
            
            
