import streamlit as st
import pandas as pd
import pca_chart as pc
import matplotlib.pyplot as plt
import word_similarity
import re
import plotly.express as px
import process_data as pcd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title('Job Recommender System')
st.header("Overview of dataset")

df=pcd.load_data()
fig=pcd.topindustry()
st.caption("Top 15 job industries from dataset")

st.plotly_chart(fig, use_container_width=True)
#st.dataframe(pd.process_data())



tab1, tab2, tab3 = st.tabs(["By Skills", "By Jobs", "By Job Industry"])

with tab1:
    st.sidebar.markdown("About our dataset")
    st.sidebar.markdown("A dataset of 1489 records was obtained by data scraping from Monster, a global online provider of job seeking, career management and more")
    #st.sidebar.markdown("Show which jobs are similar")
    newdf_all=pcd.process_data()

    user_input = st.text_area("Skills or topics that you are interested in", '')

    user_input = str(user_input)
    user_input = re.sub('[^a-zA-Z0-9\.]', ' ', user_input)
    user_input = user_input.lower()

    

    user_a = pd.DataFrame({'job_skills':[user_input]})
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_job = tfidf_vectorizer.fit_transform((newdf_all['des_lower']))

    user_tfidf = tfidf_vectorizer.transform(user_a['job_skills']) ## can be change to job title 
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_job)
    result1=list(cos_similarity_tfidf)

    top = sorted(range(len(result1)), key=lambda i: result1[i], reverse=True)[:10]
    list_scores = [result1[i][0][0] for i in top]
    
    
    st.dataframe(pcd.get_recommendation(top,newdf_all, list_scores))
    

with tab2:
    st.header("Similar Job Search")
    job_input = st.text_area("Enter job title to search for similar jobs", 'Data Engineer')


    st.write('Jobs similar to', job_input)
    tfidf_job = tfidf_vectorizer.fit_transform((newdf_all['ttl_lower']))
    userjob_tfidf = tfidf_vectorizer.transform([job_input]) ## can be change to job title 
    cos_similarity_tfidf = map(lambda x: cosine_similarity(userjob_tfidf, x),tfidf_job)
    result1=list(cos_similarity_tfidf)
    top = sorted(range(len(result1)), key=lambda i: result1[i], reverse=True)[:10]
    list_scores = [result1[i][0][0] for i in top]
    
    
    st.dataframe(pcd.get_recommendation(top,newdf_all, list_scores))

    

with tab3:
    st.header("Most frequent used words in job descriptions")
    option = st.selectbox(
     'Select Job Industry',
     ('IT/Computers - Software', 'Banking/Accounting/Financial Services', 'Recruitment/Staffing/RPO',
     'Internet/E-commerce','IT/Computers - Hardware &amp; Networking','ITES/BPO','Hotels/Hospitality/Restaurant',
     'Telecom'))
    st.write('You selected:', option)
    #put option as argument of industrywords def

    newdata=pcd.process_data()
    sorteddf=pcd.sortbyindustry(newdata,option)
    
    fig,unstackedtext=pcd.industrykeywords(sorteddf)
    st.plotly_chart(fig, use_container_width=True)
    wordcloud=WordCloud().generate(unstackedtext)
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    #recommended 





   


def plot_clusters():
    st.markdown('This chart uses PCA to show you where you fit among the different job archetypes.')
    X_train, pca_train, y_train, y_vals, pca_model = pc.create_clusters()
    for i, val in enumerate(y_train.unique()):
        y_train = y_train.apply(lambda x: i if x == val else x)
    example = user_input
    doc = pc.transform_user_resume(pca_model, example)

    pc.plot_PCA_2D(pca_train, y_train, y_vals, doc)
    st.pyplot()

#plot_clusters()