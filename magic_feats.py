def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0

def magiic_features(data = data, test_data = test_data, test = False):
	
	df1 = data[['question1']].copy()
	df2 = data[['question2']].copy(),
	df1_test = test_data[['question1']].copy()
	df2_test = test_data[['question2']].copy()

	df2.rename(columns = {'question2':'question1'},inplace=True)
	df2_test.rename(columns = {'question2':'question1'},inplace=True)

	train_questions = df1.append(df2)
	train_questions = train_questions.append(df1_test)
	train_questions = train_questions.append(df2_test)
	train_questions.drop_duplicates(subset = ['question1'],inplace=True)

	train_questions.reset_index(inplace=True,drop=True)

	questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
	
	train_cp = data.copy()
	test_cp = test_data.copy()
	train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

	test_cp['is_duplicate'] = -1
	test_cp.rename(columns={'test_id':'id'},inplace=True)

	comb = pd.concat([train_cp,test_cp])

	comb['q1_hash'] = comb['question1'].map(questions_dict)
	comb['q2_hash'] = comb['question2'].map(questions_dict)

	q1_vc = comb.q1_hash.value_counts().to_dict()
	q2_vc = comb.q2_hash.value_counts().to_dict()

	#map to frequency space
	comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
	comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

	train_comb = comb[comb['is_duplicate'] >= 0][['q1_hash','q2_hash','q1_freq','q2_freq']]
	test_comb = comb[comb['is_duplicate'] < 0][['q1_hash','q2_hash','q1_freq','q2_freq']]

	if test == True:
		return np.array(test_comb)
	else:
		return np.array(train_comb)