####################################################################################
# Feature Engineering Method 2: 
# Identifying the average amount of shared words between question1 and question2
####################################################################################
#Loads in nltk stopwords
stop_words = set(stopwords.words("english"))

##########################################
#Function: shared_words
#Purpose: Finds the shared words between question1 and question2 (exclude stop words)
#Parameters: Dataframe row containing question1 and question
##########################################
def shared_words(row):
    question1_words = []
    question2_words = []
    for word in set(str(row.question1).lower().split()):
        if word not in stop_words:
            question1_words.append(word)
            
    for word in set(str(row.question2).lower().split()):
        if word not in stop_words:
            question2_words.append(word)
    
    #Question contains only stop words (or is an empty string)
    if len(question1_words) == 0 or len(question2_words) == 0:
        return 0
    
    question1_shared_words = [w for w in question1_words if w in question2_words]
    question2_shared_words = [w for w in question2_words if w in question1_words]
    
    avg_words_shared = (len(question1_shared_words) + len(question2_shared_words))/(len(question1_words) + len(question2_words))
    return avg_words_shared
##########################################

#data['avg_words_shared'] = data.apply(shared_words, axis=1, raw=True)
####################################################################################