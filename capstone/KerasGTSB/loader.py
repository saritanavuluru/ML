#loader


# Load pickled data
from sklearn.model_selection import train_test_split
import pickle




def load_data():
    ''' This function loads the train, test, validatioon data  from the pickle files.
    '''

    training_file = "data/train.p"
    validation_file= "data/valid.p"
    testing_file = "data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test) == len(y_test))
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test





