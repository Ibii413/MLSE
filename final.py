import sys
import os
from xmlrpc.client import boolean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from keras.models import load_model
import pickle



def prepare_data(text):
     #PREPARING DATA

    
    #First we clean the text from simbols, punctuation, blank spaces...
    pattern = r"[,.!@#$%^&*()_+=\[\]{};:\"<>?/|\\'\d]"
    clean_review = text.apply(lambda x: re.sub(pattern, "", str(x)))

    #Then we split the text in words
    review_tokenized = clean_review.apply(lambda x: word_tokenize(x))

    #Finally we discard useless word and change the others to lowercase
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    final_review = review_tokenized.apply(lambda tokens: [token.lower() for token in tokens if token.lower() not in stop_words])


    nltk.download('wordnet')
    # Lemmatizing the data
    lemmatizer = WordNetLemmatizer()
    lemmatized_review = final_review.apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
    
    return lemmatized_review
    

def train_model(text, outcome):
    from sklearn import model_selection
    from sklearn import preprocessing
    from sklearn.decomposition import TruncatedSVD
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN
    from keras.optimizers import Adam
    
    #Code the outcome into 0 an 1.
    target_coder = preprocessing.LabelEncoder()
    coded_target = target_coder.fit_transform(outcome)
    
    #to make smaller the database i quit infrequent words
    from nltk.probability import FreqDist

    all_words = [word for tokens in text for word in tokens]
    freq_dist = FreqDist(all_words)

    threshold = 5
    infrequent_words = set([word for word, freq in freq_dist.items() if freq < threshold])
    filtered_review = text.apply(lambda tokens: [word for word in tokens if word not in infrequent_words])
    
    #Split the database into training and testing
    attribute_train, attribute_test, outcome_train, outcome_test = model_selection.train_test_split(filtered_review, coded_target, test_size=0.2, random_state=42)
  
    #Vectorizing the data to train and test the model
    vectorizer = TfidfVectorizer()
    
    attribute_train = attribute_train.apply(' '.join)
    attribute_test = attribute_test.apply(' '.join)

    train_matrix = vectorizer.fit_transform(attribute_train)
    test_matrix = vectorizer.transform(attribute_test)

    n_components = 100  
    svd = TruncatedSVD(n_components=n_components)
    
    train_matrix_svd = svd.fit_transform(train_matrix)
    test_matrix_svd = svd.transform(test_matrix)

    #Saving the vectorizer and svd to vectorize the review thta we want to test later
    with open('vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

    with open('svd.pkl', 'wb') as file:
        pickle.dump(svd, file)

    #Creating the model with different layers to make it more complex
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=(train_matrix_svd.shape[1],1),return_sequences=True))
    model.add(SimpleRNN(128, return_sequences=True)) 
    model.add(SimpleRNN(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    optimizer = Adam(learning_rate=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_matrix_svd, outcome_train, epochs=50, batch_size=100, validation_data=(test_matrix_svd, outcome_test))

    loss, accuracy = model.evaluate(test_matrix_svd, outcome_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy) 

    model.save('my_model.h5')

        
    
def analyze_text(text,model):
    
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
    
    #The function prepare_data is made for databases so we create a new one for our review
    data = {
    "review": [text]
    }
    
    dataset_text = pd.DataFrame(data)
    review = prepare_data(dataset_text['review'])
    
    
    #Join all the words into a chain
    review = review.apply(' '.join)
   
    
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('svd.pkl', 'rb') as file:
        svd = pickle.load(file)
    
    #With the vectorizer and svd used when training the model now we vectorized the review
    review_matrix = vectorizer.transform(review)
    review_matrix_svd = svd.transform(review_matrix)
    
    review_matrix_svd = np.reshape(review_matrix_svd, (review_matrix_svd.shape[0], 100, 1))
    
    
    #Finally we can predict with tghe model
    predictions = model.predict(review_matrix_svd)
    
    print(predictions)
    predicted_class = "positive" if predictions[0] >= 0.5 else "negative"
    
    print('The review is : ',predicted_class, )
    

def main():
    
    
    
    model_filename = "my_model.h5"
    
    if os.path.exists(model_filename):
        
        model = load_model('my_model.h5')
        print("Trained model found and loaded.")
    else:
        
        print("The trained model was not found. Training a new model...")

       #open the dataset
        dataset = pd.read_csv('IMDB dataset.csv', header=0)
        
        
        #processing the data 
        text = dataset['review']
        outcome = dataset['sentiment']
        processed_text = prepare_data(text)
        processed_dataset = pd.DataFrame({'texto_procesado': processed_text})
        
        # Saving the new dataset
        processed_dataset.to_csv('final_dataset.csv', index=False)
        
        model = train_model(processed_dataset['texto_procesado'],outcome)
    
    
    while True:
        text = input("Write a review about a movie (or type 'exit' to close the program): ")
        
        if text.lower() == "exit":
            print("Clossing...")
            sys.exit
            break
        else:
            
            analyze_text(text.lower(), model)
            
        
        decision = input("Do you wanna write another review? (y/n): ")
        if decision.lower() == "n":
            print("Clossing...")
            sys.exit
            break
        elif decision.lower() != "y":
            print("Invalid input. Please enter 'y' or 'n'.")
            

if __name__ == "__main__":
    main()