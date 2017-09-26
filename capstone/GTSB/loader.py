#loader


# Load pickled data
from sklearn.model_selection import train_test_split
import pickle
import os




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

# Load pickled data

def load_split_input_data() :

    # TODO: fill this in based on where you saved the training and testing data
    training_file = './data/train.p'
    testing_file = './data/test.p'
    validation_file = './data/valid.p'


    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    X_val, y_val = valid['features'], valid['labels']

    print('Shapes:')
    print('X_train: '+str(X_train.shape))
    print('X_test: '+str(X_test.shape))
    print('X_valid: '+str(X_val.shape))

    # STOP: Do not change the tests below. Your implementation should pass these tests. 
    assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
    
    return X_train, y_train, X_test, y_test, X_val, y_val


    





#Save to Disk
def save_to_disk(X_train,y_train,X_test,y_test,X_val,y_val,X_aug,y_aug,flag_color,preop):
# Save the data for easy access
    
    
    assert(len(X_train) == len(y_train)), "The number of images in train is not equal to the number of labels."
    assert(len(X_test) == len(y_test)), "The number of images in test is not equal to the number of labels."
    assert(len(X_val) == len(y_val)), "The number of images in val is not equal to the number of labels."
    assert(len(X_aug) == len(y_aug)), "The number of images in aug is not equal to the number of labels."

    if flag_color == True:
        label="_"+preop+"_color"
    else:
        label = "_"+preop+"_gray"
    
    #print(label)
    
    
    pickle_file = 'preprocessed/preprocessed'+label+".p"
    #print("Saving  to "+pickle_file)
    print(pickle_file)
    if not os.path.isfile(pickle_file):
        print('Saving data to '+pickle_file+' file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'y_train': y_train,
                        'X_train': X_train,
                        'X_val': X_val,
                        'y_val': y_val,
                        'X_test': X_test,
                        'y_test': y_test,
                        'X_aug': X_aug,
                        'y_aug':y_aug

                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
                raise
        
        print('Data cached in '+pickle_file)
    else:
        print("File already exists: "+pickle_file)
        
def save_aug_all_to_disk(X_aug_all,y_aug_all):
    assert(len(X_aug_all) == len(y_aug_all)), "The number of images is not equal to the number of labels."

    
    label="preprocessed_aug_all_gray"
    
    
    print(label)
    
    
    pickle_file = label+".p"
    print(pickle_file)
    if not os.path.isfile(pickle_file):
        print('Saving data to'+pickle_file+' file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'y_aug_all': y_aug_all,
                        'X_aug_all': X_aug_all,
                        

                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
                raise
        
        print('Data cached in pickle file.')


def load_preprocessed_data(pickle_file):

    print(pickle_file)
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        X_train = pickle_data['X_train']
   
        y_train = pickle_data['y_train']
        X_val = pickle_data['X_val']
        y_val = pickle_data['y_val']
        X_test = pickle_data['X_test']
        y_test = pickle_data['y_test']
        X_aug = pickle_data['X_aug']
        y_aug = pickle_data['y_aug']

        del pickle_data  # Free up memory

        print("X_train: ", len(X_train))
        print("y_train: ", len(y_train))
        print("X_val: ", len(X_val))
        print("y_val: ", len(y_val))
        print("X_test: ", len(X_test))
        print("y_test: ", len(y_test))
        print("X_aug: ", len(X_aug))
        print("y_aug: ", len(y_aug))
        # STOP: Do not change the tests below. Your implementation should pass these tests. 
        assert(len(X_train) == len(y_train)), "The number of images is not equal to the number of labels."
        assert(len(X_test) == len(y_test)), "The number of images is not equal to the number of labels."
        assert(len(X_val) == len(y_val)), "The number of images is not equal to the number of labels."

        print('Data and modules loaded.')
        
    return X_train,y_train,X_test,y_test,X_val,y_val,X_aug,y_aug

#save_to_disk(X_train[:4],y_train[:4],X_test[:4],y_test[:4],X_val[:4],y_val[:4],False,"original")
def load_preprocessed_aug(pickle_file):

    print(pickle_file)
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        X_aug_all = pickle_data['X_aug_all']
   
        y_aug_all = pickle_data['y_aug_all']
        

        del pickle_data  # Free up memory

        print("X_aug_all: ", len(X_aug_all))
        print("y_aug_all: ", len(y_aug_all))
       
        # STOP: Do not change the tests below. Your implementation should pass these tests. 
        assert(len(X_aug_all) == len(y_aug_all)), "The number of images is not equal to the number of labels."
        print('Data and modules loaded.')
        
    return X_aug_all,y_aug_all

#save_to_disk(X_train[:4],y_train[:4],X_test[:4],y_test[:4],X_val[:4],y_val[:4],False,"original")





