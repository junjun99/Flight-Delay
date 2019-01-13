import pandas as pd
from sklearn.model_selection import train_test_split

def read_csv(path):
	raw_df = pd.read_csv(path)
	return raw_df

def drop(df):
	drop_df = df.drop(['TAIL_NUM', 'FL_NUM', 'DEP_TIME', 'DEP_DELAY', 
                       'DEP_DEL15', 'ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME','UNIQUE_CARRIER','YEAR', 'QUARTER',
                      'ORIGIN', 'DEST','CANCELLED', 'DIVERTED'],axis=1)
    drop_df = drop_df.dropna()
	return drop_df




def get_train_test(df, y_col="ARR_DEL15", x_cols, ratio):
    """ 
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
     
    df_train , df_test = train_test_split(df, test_size=train_test_ratio, random_state=42)
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test

    
def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models

def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))
 
