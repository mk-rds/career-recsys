import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import re
import string
import nltk
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


#Stop words present in the library
new_stopwords = ["index","ability", "good", "years", "knowledge", "work","job", "strong","description","requirements","experience","work"]
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(new_stopwords)

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    
    output= [i for i in text if i not in stopwords]
    return output

def Remove(text):
    result = text.translate(text.maketrans("", "", string.punctuation)) #string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    return result

def tokenization(text):
    remover = nltk.RegexpTokenizer(r"\w+")
    tokens = remover.tokenize(text)
    return tokens

def tokenize_stem(series):

    tokenizer =TreebankWordTokenizer()
    stemmer = PorterStemmer()
    series = series.apply(lambda x: x.replace("\n", ' '))
    series = series.apply(lambda x: tokenizer.tokenize(x))
    series = series.apply(lambda x: [stemmer.stem(w) for w in x])
    series = series.apply(lambda x: ' '.join(x))
    return series

def display_topics(model, feature_names, no_top_words, topic_names=None):
    '''
    displays topics and returns list of toppics
    '''

    topic_list = []
    for i, topic in enumerate(model.components_):
        if not topic_names or not topic_names[i]:
            print("\nTopic ", i)
        else:
            print("\nTopic: '",topic_names[i],"'")

        print(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_list.append(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
    return model.components_, topic_list

def return_topics(series, num_topics, no_top_words, model, vectorizer):
    '''
    returns document_topic matrix and topic modeling model
    '''
    #turn job into series
    series = tokenize_stem(series)
    #transform series into corpus
    ex_label = [e[:30]+"..." for e in series]
    #set vectorizer ngrams = (2,2)
    vec = vectorizer(stop_words = 'english')

    doc_word = vec.fit_transform(series)

    #build model
    def_model = model(num_topics)
    def_model = def_model.fit(doc_word)
    doc_topic = def_model.transform(doc_word)
    #print('model components: ', def_model.components_[0].shape)
    #print('doc_topic', doc_topic[0])
    model_components, topic_list = display_topics(def_model, vec.get_feature_names(), no_top_words)
    return def_model.components_, doc_topic, def_model, vec, topic_list#, topics


def load_data():
    '''
    uses the functions above to read in files, model, and return a topic_document dataframe
    '''
    #read in jobs file and get descriptions
    df = pd.read_json('monster_india_latest_jobs_free_dataset.json')
    #df = df[df.keyword!='marketing']
    df=df[['title','description','industry','skills']]
    #df['industry']=df['industry'].astype(str)
    #df['description']=df['description'].str.lower()
    #df['title']=df['title'].str.lower()
    #df['skils']=df['skills'].str.lower()
    #df['industry']=df['industry'].str.lower()

    #jobs_df = pd.DataFrame(zip(df['Job Description'], df['keyword']), columns = ['Description', 'Job'])

    #array, doc, topic_model, vec, topic_list  = return_topics(jobs_df['Description'],20, 10, TruncatedSVD, TfidfVectorizer)

    #topic_df = pd.DataFrame(doc)
    #topic_df.columns = ['Topic ' + str(i+1) for i in range(len(topic_df.columns)) ]

    #topic_df['job'] = jobs_df.Job
    #Topic_DF.to_csv('topic_df.csv')
    return df

def process_data():
    newdf=load_data()
    newdf['industry']=newdf['industry'].astype(str)
    newdf['des_lower']=newdf['description'].str.lower()
    newdf['ttl_lower']=newdf['title'].str.lower()
    newdf['skil_lower']=newdf['skills'].str.lower()
    newdf['ind_lower']=newdf['industry'].str.lower()
    newdf['dictionary'] = newdf[["des_lower","ttl_lower","skil_lower","ind_lower"]].apply(lambda x:"".join(x),axis=1)
    newdf['clean_dict']= newdf['dictionary'].apply(lambda x:tokenization(x))
    #new_stopwords = ["ability", "good", "years", "knowledge", "work","job", "strong","description","requirements","experience","work"]
    new_stopwords = ["ability", "good", "years", "knowledge", "work","job", "strong","description","requirements","experience","work"]
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(new_stopwords)
    newdf['no_stopwords']=newdf['clean_dict'].apply(lambda x:remove_stopwords(x))
    newdf['no_stopwords']=newdf['no_stopwords'].astype(str)

    return newdf

def sortbyindustry(data,industry):
    newdata=data.loc[data['industry'] == industry]
    return newdata


def topindustry():
    df=process_data()
    industrydf=df[['industry']]
    industrydf['industry']=industrydf['industry'].astype(str)
    industrydf=industrydf.groupby(['industry']).size().reset_index(name='counts')
    industrydf = industrydf[industrydf['industry'] != 'Other']
    industrydf = industrydf[industrydf['industry'] != 'NA']
    industrysorted=industrydf.sort_values(by=['counts'],ascending=False)
    industrysorted=industrysorted.head(15)
    fig = px.bar(industrysorted, x=industrysorted.industry, y = industrysorted.counts)
    
    return fig

def industrykeywords(sorteddf):
    #newdata=data.loc[data['industry'] == industry]
    textcount = sorteddf['no_stopwords'].str.lower().str.replace('[^\w\s]','')
    
    textcount = textcount.str.split(expand=True).stack().value_counts().reset_index()
    unstackedtext=textcount.to_string()
    textcount.columns = ['Word', 'Frequency'] 
    toptextcount=textcount.head(30)
    fig = px.bar(toptextcount, x=toptextcount.Word, y = toptextcount.Frequency)

    return fig, unstackedtext

def cosineresult(user_input):
    user_tfidf = tfidf_vectorizer.transform(user_a['job_skills']) ## can be change to job title 
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_job)
    return

def get_recommendation(top, newdf_all, scores):
    
    recommendation = pd.DataFrame(columns = ['ttl_lower', 'des_lower',  'skil_lower','ind_lower','score'])
    count = 0
    for i in top:
        recommendation.at[count, 'ttl_lower']=newdf_all['ttl_lower'][i]
        recommendation.at[count, 'des_lower'] = newdf_all['des_lower'][i]
        recommendation.at[count, 'skil_lower'] = newdf_all['skil_lower'][i]
        recommendation.at[count, 'ind_lower'] = newdf_all['ind_lower'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    recommendation = recommendation[recommendation['score'] <0.7]
    return recommendation

def get_recommendation1(top, newdf_all, scores):
    
    recommendation = pd.DataFrame(columns = ['ttl_lower', 'des_lower',  'skil_lower','ind_lower','score'])
    count = 0
    for i in top:
        recommendation.at[count, 'ttl_lower']=newdf_all['title'][i]
        recommendation.at[count, 'des_lower'] = newdf_all['description'][i]
        recommendation.at[count, 'skil_lower'] = newdf_all['skills'][i]
        recommendation.at[count, 'ind_lower'] = newdf_all['industry'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    


    return recommendation



def predictive_modeling(df):
    '''
    fits, optimizes, and predicts job class based on topic modeling corpus
    '''
    X,y = df.iloc[:,0:-1], df.iloc[:, -1]
    X_tr, X_te, y_tr, y_te = train_test_split(X,y)

    param_grid = {'n_estimators': [100,300, 400, 500, 600], 'max_depth': [3,7,9, 11]}
    # search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    # search.fit(X_tr, y_tr)
    # bp = search.best_params_
    # print(bp)
    #rfc = RandomForestClassifier(n_estimators = bp['n_estimators'], max_depth = bp['max_depth'])
    rfc = RandomForestClassifier(n_estimators = 500, max_depth = 9)
    rfc.fit(X_tr, y_tr)
    print('acc: ', np.mean(cross_val_score(rfc, X_tr, y_tr, scoring = 'accuracy', cv=5)))
    print('test_acc: ', accuracy_score(y_te, rfc.predict(X_te)))
    print(rfc.predict(X_te))
    return rfc

def predict_resume(topic_model, model, resume):
    '''
    transforms a resume based on the topic modeling model and return prediction probabilities per each job class
    '''
    doc = topic_model.transform(resume)
    return model.predict_proba(doc), model.classes_

def get_topic_classification_models():
    jobs_df, model, vec , topic_list= process_data()
    model_1 = predictive_modeling(jobs_df)
    return model, model_1, vec


def main(resume, topic_model, predictor, vec):
    '''
    run code that predicts resume
    '''
    #jobs_df, model, vec , topic_list= process_data()
    #model_1 = predictive_modeling(jobs_df)

    doc = tokenize_stem(resume)
    doc = vec.transform(doc)
    probabilities, classes = predict_resume(topic_model, predictor, doc)
    return classes, probabilities[0]*100

