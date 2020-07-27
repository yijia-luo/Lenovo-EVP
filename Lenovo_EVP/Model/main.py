# get phrase
def filter_word(company,word):
    import pandas as pd
    import numpy as np
    file = 'Phrase/'+company+'_phrase.csv'
    df=pd.read_csv(file,index_col=0)
    pro=[]; con=[]
    for i in range(len(df['pro'])):
        if type(df['pro'][i])==str:
            pro.append(df['pro'][i].lower())
    for i in range(len(df['con'])):
        if type(df['con'][i])==str:
            con.append(df['con'][i].lower())
    pro_list=list(set(list(filter(lambda x: word in x and len(x.split())>1, pro))))
    con_list=list(set(list(filter(lambda x: word in x and len(x.split())>1, con))))
    col_pro = word+'_pro'
    col_con = word+'_con'
    df1=pd.DataFrame(pro_list,columns=[col_pro])
    df2=pd.DataFrame(con_list,columns=[col_con])
    result = pd.concat([df1,df2],axis=1)
    return result

#get review
def filter_review(company,word,p_number,n_number):
    import pandas as pd
    import numpy as np
    file = 'reviews/'+company+'.csv'
    df=pd.read_csv(file,index_col=0)
    df_p=df[df['Pros'].apply(lambda x: word in x) ==True].iloc[:p_number]
    df_n=df[df['Cons'].apply(lambda x: word in x) ==True].iloc[:n_number]
    
    return df_p,df_n

# get sentiment
def get_affect(company,word,lower=True):
    import nltk
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import pandas as pd
    import numpy as np
    content = pd.read_csv('reviews/'+company+'.csv',index_col=0)
    full_review = ''
    for review in content['Pros']:
        full_review = full_review+review+' '
    for review in content['Cons']:
        full_review = full_review+review+' '
        
    analyzer = SentimentIntensityAnalyzer()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(full_review.strip())
    sentence_count = 0
    running_total = 0
    for sentence in sentences:
        if lower: 
            sentence = sentence.lower()
            word = word.lower()
        if word in sentence:
            vs = analyzer.polarity_scores(sentence) 
            running_total += vs['compound']
            sentence_count += 1
    if sentence_count == 0: return 0
    return running_total/sentence_count

def __main__():
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from IPython.display import display
    
    company = input("Which company? ").lower()
    print()
    company_list=['apple','microsoft','alibaba','baidu','dell','hp','huawei','ibm','lenovo','samsung']
    if company not in company_list:
        print("Please check the company name. The company should be in ['Apple','Microsoft','Alibaba','Baidu','Dell','Hp','Huawei','IBM','Lenovo','Samsung']")
        company = input("Which company? ").lower()
        
    word = input("Input your word: ").lower()
    print()
    p_reviews = input("How many positive reviews related to the word inputted you want (default 10)")
    n_reviews = input("How many negative reviews related to the word inputted you want (default 10)")
    phrases = input("How many phrases related to the word inputted you want (default 10)")
    print()
    try:
        p_reviews = int(p_reviews)
    except:
        p_reviews = 10
    try:
        n_reviews = int(n_reviews)
    except:
        n_reviews = 10
    try:
        phrases = int(phrases)
    except:
        phrases = 10
    
    sentiment = get_affect(company,word)
    phrase_result = filter_word(company,word).iloc[:phrases]
    positive_review, negative_review = filter_review(company,word,p_reviews,n_reviews)
    print("The sentiment score for this word is: " + str(sentiment))
    print()
    print("Phrase results are: ")
    print(tabulate(phrase_result, headers='keys', tablefmt='psql'))
    print("Positive reviews are: ")
    display(positive_review)
    print("Negative reviews are: ")
    display(negative_review)
    return sentiment, phrase_result, positive_review, negative_review

__main__()   
