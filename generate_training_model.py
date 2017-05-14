##########################################
# Load Required Python Libraries
##########################################
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split as tts
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from pylev import levenshtein
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin

#Define transformer functions 

#These are the functions for our transformers
class LevDistanceTransformer(BaseEstimator, TransformerMixin):
    """Takes in two lists of strings, extracts the lev distance between each string, returns list"""

    def __init__(self):
        pass

    def transform(self, question_list):
        q1_list = question_list[0]
        q2_list = question_list[1]
        
        lev_distance_strings = [[a,b] 
        for a,b in zip(q1_list, q2_list)]
        
        lev_dist_array = np.array([
    (float(levenshtein(pair[0], pair[1]))/
    (float(sum([x.count('') for x in pair[0]])) + 
    float(sum([x.count('') for x in pair[1]])))) 
    for pair in lev_distance_strings 
        ])
        
        return lev_dist_array.reshape(len(lev_dist_array),1)

    def fit(self, question_list, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class TfIdfDiffTransformer(BaseEstimator, TransformerMixin):
    """Takes in two lists of strings, extracts the lev distance between each string, returns list"""

    def __init__(self, total_words):
        pass

    def transform(self, question_list):
        q1_list = question_list[0]
        q2_list = question_list[1]
        total_questions = q1_list + q2_list
        total_questions = [x for x in total_questions if type(x) != float]
        
        vectorizer = TfidfVectorizer(stop_words = 'english', vocabulary = total_words)
        vectorizer.fit(total_questions)
        tf_diff = vectorizer.transform(q1_list) - vectorizer.transform(q2_list)
        return tf_diff

    def fit(self, question_list, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class CosineDistTransformer(BaseEstimator, TransformerMixin):
    """Takes in two lists of strings, extracts the lev distance between each string, returns list"""

    def __init__(self):
        pass

    def transform(self, question_list):
        q1_list = question_list[0]
        q2_list = question_list[1]
        total_questions = q1_list + q2_list
        total_questions = [x for x in total_questions if type(x) != float]
        
        vectorizer = TfidfVectorizer(stop_words = 'english')
        vectorizer.fit(total_questions)
        
        q1_tf = vectorizer.transform(q1_list) 
        q2_tf = vectorizer.transform(q2_list)
        cos_sim = []
        for i in range(0,len(q1_list)):
            cos_sim.append(cosine_similarity(q1_tf[i], q2_tf[i])[0][0])
            
        return np.array(cos_sim).reshape(len(cos_sim),1)

    def fit(self, question_list, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


##########################################
# Loads in Quora Dataset
##########################################
#Training Dataset
data = pd.read_csv('train.csv')
data['question1'] = data['question1'].astype(str)
data['question2'] = data['question2'].astype(str)
y = data['is_duplicate'][0:1000]

test_data = pd.read_csv('test.csv')
test_data['question1'] = test_data['question1'].astype(str)
test_data['question2'] = test_data['question2'].astype(str)
#Drop irrelevant features
data = data.drop(['id', 'qid1', 'qid2'], axis=1)


#Use word vocabulary from training data
vectorizer = TfidfVectorizer(stop_words = 'english')
vectorizer.fit(data['question1'][0:1000] + data['question2'][0:1000])
total_words = list(set(vectorizer.get_feature_names()))
# # ##########################################

#Combine the two features and predict using the combined set
comb_features = FeatureUnion([('tf', TfIdfDiffTransformer(total_words)), ('cos_diff',CosineDistTransformer()), ('lev', LevDistanceTransformer())])
comb_features.fit([data['question1'][0:1000], data['question2'][0:1000]])
train_features = comb_features.transform([data['question1'][0:1000], data['question2'][0:1000]])

#Create a Random Forest Classifier and train it on our data
clf = RandomForestClassifier()
clf.fit(train_features, y)

#Perform a cross vaidation check on the training set
#we're going to use this percentage as the average accuracy of the model
scores = cross_val_score(clf, train_features, y, cv=10, scoring = 'f1_macro')
print scores
with open('cross_validation_scores.sav', 'wb') as file:
	pickle.dump(scores, file)

#Output the fitted model to a pickeled file
with open('rf_tfidf_cos_lev.sav', 'wb') as file:
	pickle.dump(scores, file)
#Predict on the test set

test_features = comb_features.transform([test_data['question1'][0:1000], test_data['question2'][0:1000]])
test_prediction = clf.predict(test_features)

#Output the predictions 
submission = pd.DataFrame()
submission['test_id'] = test_data['test_id'][0:1000]
submission['is_duplicate'] = test_prediction
submission['question1'] = test_data['question1'][0:1000] 
submission['question2'] = test_data['question2'][0:1000] 
submission.to_csv('submission.csv', index = False)