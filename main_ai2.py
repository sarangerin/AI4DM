# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:31:04 2022

@author: Erin
"""

if __name__ == '__main__':
    
    import os
    import sys
    import pickle
    import pandas as pd
    from collections import Counter
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    os.chdir(directory)    

    
    sys.path.append(directory+'/submodule')

    directory = 'C:/Users/Erin/Documents/Erin Carrer/Dissertation/Codes/aifordm2/'
    #directory_ = directory + '/input/'

    data = pd.DataFrame()
    
    for file in os.listdir(directory) :
        if '.csv' in file :
            temp_data = pd.read_csv(directory + file, skiprows=4)
            temp_data['db'] = file.split('_')[0]        
            data = pd.concat([data, temp_data]).reset_index(drop = 1)
            del data['명칭(원문)']
        
    #%% 1. Data pre-processing
    
    # data_ = pd.DataFrame()
    
    # total n = 1145
    
    data = data[['번호','명칭','요약', '출원인대표명', '국제특허분류', '공통특허분류', '출원일','독립 청구항수', '전체 청구항수',
                 '대표 청구항','자국인용횟수', '자국피인용횟수' ,'INPADOC패밀리국가수','발명자수','최종 상태','db']]
    
    data.columns = ['pt_id', 'title', 'abstract', 'applicant', 'IPC', 'CPC', 'application_date', 'ind_claims_cnt', 'total_claims_cnt',
                    'claims_rep', 'forward_cites_cnt', 'backward_cites_cnt', 'family_country_cnt', 'inventor_cnt','state' ,'db']    

    data = data[['pt_id','title','abstract','CPC','application_date','claims_rep','state','db']]
    data = data.dropna(subset = ['application_date']).reset_index(drop = 1)

    data['application_year'] = data['application_date'].apply(lambda x : x.split('.')[0])
    data['TA'] = data['title'] +' '+ data['abstract']
    
    # before eda
    eda_df = pd.DataFrame()
    for db in sorted(set(data['db'])) :
        temp_data = data.loc[data['db'] == db, :]
        c = Counter(temp_data['application_year'])
        c = pd.DataFrame(c.values(), index = c.keys())
        c = c.sort_index()
        eda_df[db] = c


    data = data.sort_values(by=['application_year']).reset_index(drop = 1)
    
    
    #%% 2. Data Filtering (i.e., Data cleansing + Pre-processing 1)
        
    from Levenshtein import distance as lev
    
    data_ = pd.DataFrame()
    
    for idx, row in data.iterrows() : 
        
        code_list = ['US', 'EP', 'WO' , 'KR' , 'JP']
        
        if any(code in row['pt_id'] for code in code_list):
            
            state_list = ['Rejected', 'Withdrawn', 'Abandoned'] 
            
            if any(state == row['state'] for state in state_list):
                pass
            else :  
                if str(row['abstract']) != 'nan' :
                    data_ = data_.append(row)
    
    # 유사문서제거 
    for idx, row in data_.iterrows() :
        
        abstract = row['abstract']
        
        for idx_, row_ in data_.iterrows() :
            if idx != idx_ :
                abstract_ = row_['abstract']
                if lev(abstract, abstract_) <= 5 :
                    data_ = data_.drop(idx_)
                    print(idx_)
                    
    data_ = data_.reset_index(drop = 1)
    
    c = Counter(data_['db'])
    
    # 2. EDA AFTER
    
    data_['country'] = data_['pt_id'].apply(lambda x : x[0:2])
    country_list = data_['country']
    country_list = list(set(country_list))
    total_count = pd.DataFrame()
    
    for country in country_list :
        data_sample = data_.loc[data_['country'] == country,:]
        c = Counter(data_sample['application_year'])
        c = pd.DataFrame(c.values(), index = c.keys())
        c = c.sort_index()
        total_count[country] = c
    
    # 시계열 처리 (Phase 2)
    span_dict = {}
    span_dict['2000'] = 0
    span_dict['2001'] = 0
    span_dict['2002'] = 0
    span_dict['2003'] = 0
    span_dict['2004'] = 0
    span_dict['2005'] = 0
    span_dict['2006'] = 1
    span_dict['2007'] = 1
    span_dict['2008'] = 1
    span_dict['2009'] = 1
    span_dict['2010'] = 1
    span_dict['2011'] = 2
    span_dict['2012'] = 2
    span_dict['2013'] = 2
    span_dict['2014'] = 2
    span_dict['2015'] = 2
    span_dict['2016'] = 3
    span_dict['2017'] = 3
    span_dict['2018'] = 3
    span_dict['2019'] = 3
    span_dict['2020'] = 3
    span_dict['2021'] = 3
    
    data_['application_span'] = data_['application_year'].apply(lambda x : span_dict[x])

    total_count = pd.DataFrame()
    
    for country in country_list :
        data_sample = data_.loc[data_['country'] == country,:]
        c = Counter(data_sample['application_span'])
        c = pd.DataFrame(c.values(), index = c.keys())
        c = c.sort_index()
        total_count[country] = c
        
        #%% preprocessing 된 데이터셋 저장
    data_.to_csv(directory + '/output/preprocessed_data.csv' ,index = 0)
    #%% 저장된 preprocessed dataset 불러오기
    
    data_ = pd.read_csv(directory + '/output/preprocessed_data.csv')
   # data_ = data_.loc[data_['application_year'] != 2000].reset_index(drop = 1)
    data_ = data_.loc[data_['application_year'] != 2021].reset_index(drop = 1)
    
    data_['country'] = data_['pt_id'].apply(lambda x : x[0:2])
    # data_sample = data_.loc[data_['country'] == 'US' , :].reset_index(drop = 1)
    data_sample = data_.loc[((data_['country'] == 'JP') | (data_['country'] == 'KR'))   , :].reset_index(drop = 1)
    #pov 는 시장
    
    #%%
    from collections import Counter
    
    # 1. col
    #EDA Counts by year
    c = Counter(data_['application_year'])
    c = pd.DataFrame(c.values(), index = c.keys())
    c = c.sort_index()
    # eda_df[db] = c
    
    #EDA Counts by country
    c_ = Counter(data_['country'])
    c_ = pd.DataFrame(c_.values(), index = c_.keys())
    c_ = c_.sort_index()
    
    #EDA Counts by Periods
    c__ = Counter(data_['application_span'])
    c__ = pd.DataFrame(c__.values(), index = c__.keys())
    c__ = c__.sort_index()
    
    #EDA Counts by CPC? 
    # 2. by 2 variables
    grouped = data_.groupby(['application_span', 'country'])
  
    # Grouped()
    temp = grouped['country'].count().unstack() 
    temp_ = grouped['application_span'].count().unstack()
    
    #%% 3. Textmining & LDA preparation2
    
    # text preprocess
    from nltk.corpus import stopwords
    import spacy
    import re
    from collections import Counter
    import numpy as np 
    
    nlp = spacy.load("en_core_web_sm")
    nlp.enable_pipe("senter")
    stopwords_nltk = set(stopwords.words('english'))
    stopwords_spacy = nlp.Defaults.stop_words
    stopwords_add = ['method', 'invention', 'comprise', 'use', 'provide', 'present'
                     ,'relate', 'thereof', 'include', 'contain', 'disclose', 'effect', 
                     'method', 'invention', 'comprise', 'use', 'provide', 'present'
                     'relate', 'thereof', 'include', 'contain', 'disclose', 'effect', 
                     'apparatus', 'second', 'module', 'mean', 'accord', 'efficiency', 'input'
                     , 'unit', 'set', 'design', 'site', 'learn', 'line', 'value', 
                     'datum', 'system', 'base', 'time', 'generate', 'determine', 'receive', 
                     'accord', 'step', 'level', 'apply', 'result', 'select', 'correspond', 'state',
                     'occur', 'based', 'medium', 'basis', 'interest', 'cell', 'order', 'ground', 
                     'line', 'allow', 'multi', 'enable', 'attribute', 'source', 'device', 'control', 'area']
    
    #tf-idf add stopwords based on the results
    def preprocess_text(df, col) :
        
        # download('en_core_web_sm')
        # stopwords_.append('-PRON-')
        
        col_ = col + '_wordlist'
        df[col_] = [nlp(i) for i in df[col]]
        
        print('nlp finished')
        
        # get keyword
        df[col_] = [[token.lemma_.lower() for token in doc] for doc in df[col_]] # lemma
        df[col_] = [[token for token in doc if len(token) > 2] for doc in df[col_]] # 길이기반 제거
        df[col_] = [[re.sub(r"[^a-zA-Z0-9-]","",token) for token in doc ] for doc in df[col_]] #특수문자 교체    
        df[col_] = [[token for token in doc if not token.isdigit() ] for doc in df[col_]]  #숫자제거 
        df[col_] = [[token for token in doc if len(token) > 2] for doc in df[col_]] # 길이기반 제거
        df[col_] = [[token for token in doc if token not in stopwords_nltk] for doc in df[col_]] 
        df[col_] = [[token for token in doc if token not in stopwords_spacy] for doc in df[col_]]
        df[col_] = [[token for token in doc if token not in stopwords_add] for doc in df[col_]] 
           
        
        return(df)
    
    # TF-IDF
    
    def tf_idf_counter(word_list) :
        temp = sum(word_list , [])
        c = Counter(temp)
        counter = pd.DataFrame(c.items())
        counter = counter.sort_values(1, ascending=False).reset_index(drop = 1)
        counter.columns = ['term', 'tf']
        counter = counter[counter['tf'] >= 1]
        counter['df'] = 0
        
        for idx,row in counter.iterrows() :
            term = row['term']
            for temp_list in word_list :
                if term in temp_list : counter['df'][idx] +=1
        
        counter['tf-idf'] = counter['tf'] / np.sqrt((1+ counter['df']))
        counter = counter.sort_values('tf-idf', ascending=False).reset_index(drop = 1)
        #counter = counter.loc[counter['tf-idf'] >= 1.5 , :].reset_index(drop = 1)
        return(counter)
    
    data_ = preprocess_text(data_, 'TA')
    
    #%% 4. TF-IDF, WC
    
    from wordcloud import (WordCloud, get_single_color_func)
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # temp_data = data_.loc[data_['db'] == '보습제'].reset_index(drop = 1)
    temp_data = data_.reset_index(drop = 1)
    
    # c = Counter(sum(data_['TA_wordlist'] , []))
    
    tf_idf = tf_idf_counter(temp_data['TA_wordlist'])
    tf_idf_times = pd.DataFrame()
    
    for time in [0,1,2,3] :
        data_sample = temp_data.loc[temp_data['application_span'] == time, :]
        temp = tf_idf_counter(data_sample['TA_wordlist'])
        temp.index = temp['term']
        temp = temp[['tf-idf']]
        tf_idf_times[time] = temp
        tf_idf['term'].tolist()
        
    class GroupedColorFunc(object):
        """
        Uses different colors for different groups of words. 
        """
    
        def __init__(self, color_to_words, default_color):
            self.color_func_to_words = [
                (get_single_color_func(color), set(words))
                for (color, words) in color_to_words.items()]
    
            self.default_color_func = get_single_color_func(default_color)
    
        def get_color_func(self, word):
            """Returns a single_color_func associated with the word"""
            try:
                color_func = next(
                    color_func for (color_func, words) in self.color_func_to_words
                    if word in words)
            except StopIteration:
                color_func = self.default_color_func
    
            return color_func
    
        def __call__(self, word, **kwargs):
            return self.get_color_func(word)(word, **kwargs)
            return self.get_color_func(word)(word, **kwargs)
        
        # Define functions to select a hue of colors arounf: grey, red and green
    def red_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 100%%, %d%%)" % random.randint(30, 50)
    
    def green_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(100, 100%%, %d%%)" % random.randint(20, 40)
    
    
    data_sample = temp_data
    tf_idf_ = tf_idf_counter(data_sample['TA_wordlist'])
    tf_idf_ = pd.merge(left = tf_idf_ , right = tf_idf, how = "left", on = "term", 
                       suffixes=('_sample', '_total'))   
    dct = dict(zip(tf_idf_['term'], tf_idf_['tf-idf_sample']))
    mask_ = np.array(Image.open(directory +'input/meta2.jpg'))
    wordcloud = WordCloud(
        background_color="white",
        mask = mask_)
    wordcloud = wordcloud.generate_from_frequencies(dct)
    # Apply our color function
    # wordcloud.recolor(color_func=grouped_color_func) 
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    #%% 5. LDA tunning
    
    from gensim.corpora import Dictionary
    import LDA_tunning
    import LDA_handling

    # with open('./output/LDA_result.pickle', 'rb') as f:
    #     LDA_obj = pickle.load(f)dl138913
    # temp_data = data_.loc[data_['db'] == '클린'].reset_index(drop = 1)
    
    texts = data_['TA_wordlist']

    keyword_dct = Dictionary(texts)
    keyword_dct.filter_extremes(no_below = 3, no_above = 0.3) #3번 이하 등장 or 40% 이상 등장하는 단어 제거
    keyword_list = list(keyword_dct.token2id.keys())
     
    keyword_dct = Dictionary(texts)
    keyword_dct.filter_extremes(no_below = 3, no_above = 0.3) #3번 이하 등장 or 40% 이상 등장하는 단어 제거
    keyword_list_ = list(keyword_dct.token2id.keys())

    test = [i for i in keyword_list if i not in keyword_list_]
    
    #%%
    corpus = [keyword_dct.doc2bow(text) for text in texts]
    # encoded_keyword = embedding.keyword_embedding(keyword_list)
    texts = [[k for k in doc if k in keyword_list] for doc in texts]
    docs = [" ".join(i) for i in texts]
    
    tunning_result = LDA_tunning.tunning(texts, keyword_dct, corpus, 10, 21, 5)
    #corpus는 bag of words이고 texts 원문은  coherence계산에 필요헤서 포함
    #text: title + abstract (전처리)
    
    tunning_result.to_csv(directory + 'output/lda_tune_result_'+temp_data['db'][0]+'.csv',index = 0)
    #%%
    tunning_result = pd.read_csv(directory + 'output/lda_tune_result_'+temp_data['db'][0]+'.csv')
    # 6. LDA handling
    from sklearn.preprocessing import MinMaxScaler
    
    transformer = MinMaxScaler() # 최대 최소를 0-1로바꾸어서 두개의다른 지표를 같이 비교
    temp = transformer.fit_transform(tunning_result[['Perplexity', 'C_v']]) #MinMaxScaler 모델에 x_train_df 데이터 적용 (최소값, 최대값 계산)
    temp[:,0] = 1 - temp[:,0] #perplexity는 마이너스 값이어서 양의값으로 바꿔준다
    temp_score = temp[:,0] + temp[:,1] 
    # temp = temp[:,0] + temp[:,1] - 0.02 * tunning_result['Topics']
    tunning_result['total'] = list(temp_score)
    
    temp_score =  temp[:,0] + temp[:,1] - 0.02 * tunning_result['Topics']
    tunning_result['total_p'] = list(temp_score)
    #투닝 결과 값이 클수록 좋긴한데, 값과 토픽의 수가 비례하는 경향이 있어서...완벽한 솔루션은 아니다 조금 완화시키기 위해서 토픽 갯수에 대한 가중치/패널티를 부여 (상수랑 곱해줌)
    #%%
   # 자동으로 맥스 찾기
    min_idx = tunning_result['total_p'].idxmax()
#%%    
    LDA_obj = LDA_tunning.LDA_obj(texts, tunning_result['Topics'][min_idx]
                                  , tunning_result['Alpha'][min_idx], 
                                  tunning_result['Beta'][min_idx],
                                  keyword_dct) # best score 입력
#%%
    topic_doc_df = LDA_handling.get_topic_doc(LDA_obj.model, LDA_obj.corpus)
    topic_word_df = LDA_handling.get_topic_word_matrix(LDA_obj.model)
    topic_topword_df = LDA_handling.get_topic_topword_matrix(LDA_obj.model, 20) # 상위 단어개수 수정 
    topic_time_df =  LDA_handling.get_topic_vol_time(LDA_obj.model, topic_doc_df, data_, 'application_span')
    topic_time_topn = {}
    topn_list = []
    
    for idx,row in topic_time_df.iterrows() :
        temp = row.tolist()
        topic_list = [temp.index(x) for x in sorted(temp, reverse = 1)]
        topic_time_topn[idx] = topic_list[0:5]
        topn_list.extend(topic_list[0:5])
        
    topn_list = list(set(topn_list))
    topic_time_topn_df = topic_time_df[topn_list]
    topic_time_topn_df.index = ['2000-2005', '2006-2010' , '2011-2015', '2016-2020']
    
    topic_title_df = LDA_handling.get_most_similar_doc2topic(data_, topic_doc_df,5, 'title', 'application_year' )
    volumn_dict = LDA_handling.get_topic_vol(LDA_obj.model, LDA_obj.corpus)
    
    #%%
    #결과저장
    import xlsxwriter  
    import pandas as pd
    
    # directory = 'C:/Users/tmlab/Desktop/작업공간/'
    writer = pd.ExcelWriter(directory+ 'output/LDA_results_'+temp_data['db'][0]+'2.xlsx', 
                            engine='xlsxwriter')
    
    topic_word_df.to_excel(writer , sheet_name = 'topic_word', index = 1)
    topic_topword_df.to_excel(writer , sheet_name = 'topic_topword', index = 1)
    topic_time_df.to_excel(writer , sheet_name = 'topic_time_vol', index = 1)
    topic_title_df.to_excel(writer , sheet_name = 'topic_doc_title', index = 1)
    pd.DataFrame(topic_doc_df).to_excel(writer , sheet_name = 'topic_doc', index = 1)
    temp = pd.DataFrame(topic_time_topn)
    temp.columns = ['2000-2005', '2006-2010' , '2011-2015', '2016-2020']
    temp.to_excel(writer , sheet_name = 'topic_time_topn', index = 1)
    topic_time_topn_df.to_excel(writer , sheet_name = 'topic_time_topn_vol', index = 1)
    
    topic_topn_topword_df = topic_topword_df[topic_time_topn[0]]
    try :topic_topn_topword_df = topic_topn_topword_df.join(topic_topword_df[topic_time_topn[1]])
    except : topic_topn_topword_df = pd.merge(topic_topn_topword_df, topic_topword_df[topic_time_topn[1]])
    try :topic_topn_topword_df = topic_topn_topword_df.join(topic_topword_df[topic_time_topn[2]])
    except : topic_topn_topword_df = pd.merge(topic_topn_topword_df, topic_topword_df[topic_time_topn[2]])
    
    topic_topn_topword_df.to_excel(writer , sheet_name = 'topic_topn_topword_vol', index = 1)
    

    writer.save()
    writer.close()
    
    #%% 6. Network part
    
    
    data_['CPC_list'] = data_['CPC'].apply(lambda x : x.split(', ') if str(x) != 'nan' else x)
    ##
    data_['CPC_list_mainG'] = data_['CPC_list'].apply(lambda x : list(set([i.split('/')[0] for i in x])) if str(x) != 'nan' else x)
    data_['CPC_list_subC'] = data_['CPC_list'].apply(lambda x : list(set([i[0:4] for i in x])) if str(x) != 'nan' else x)
    
    #%%
    
    from itertools import combinations
    
    edge_df = pd.DataFrame()
    
    for idx,row in data_[0:1000].iterrows() :
        
        CPC_list = row['CPC_list_mainG']
        
        if str(CPC_list) == 'nan' : continue    
        
        comb = list(combinations(CPC_list, 2))
        
        DICT = {}
        
        for i in comb :
            
            # 1
            if i[0][0] == i[1][0] : 
                continue
    
            # 2
            DICT['source'] = i[0]
            DICT['target'] = i[1]
            DICT['weight'] = 1
            DICT['application_span'] = row['application_span']
            
            edge_df = edge_df.append(DICT, ignore_index = 1)
   
    
    edge_df.to_csv(directory + 'output/edge_df.csv', index =0)
    
    #%%
    
    # save by span
    edge_df_ = edge_df.loc[edge_df['application_span'] == 0].reset_index(drop = 1)

    
    
    