import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle


def get_data():
    data = []
    with open('datasets/text.txt') as f:
        for line in f.readlines():
            text, emotion = line.strip().split(';')
            data.append({'text': text, 'emotion': emotion})
    data = pd.DataFrame(data)
    return data


def text_reformat(text):
    """lowercase without punctuation"""
    import re
    return re.sub(r'[^\w\s]', '', text.lower())


def preprocess(data):
    print('Removing punctuation and converting to lowercase..')
    data['text'] = data['text'].apply(text_reformat)


def split_data(data):
    X = data['text']
    le = LabelEncoder()
    y = le.fit_transform(data['emotion'])
    pickle.dump(le, open('pretrained/le_transformer.pkl', 'wb'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, X_test, y_train, y_test


def train():
    data = get_data()
    n_classes = len(set(data['emotion']))
    preprocess(data)
    X_train, X_test, y_train, y_test = split_data(data)

    vec = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))
    vec.fit(X_train)
    pickle.dump(vec, open('pretrained/vectorizer.pkl', 'wb'))

    X_train_vecs = vec.transform(X_train).todense().tolist()
    X_test_vecs = vec.transform(X_test).todense().tolist()

    clf = SGDClassifier(max_iter=100, penalty='l1', random_state=0)

    clf.fit(X_train_vecs, y_train)
    print("Score on train:", accuracy_score(y_train, clf.predict(X_train_vecs)))
    print("Score on test:", accuracy_score(y_test, clf.predict(X_test_vecs)))

    le = pickle.load(open('pretrained/le_transformer.pkl', 'rb'))
    class_labels = [le.inverse_transform([i])[0] for i in range(n_classes)]

    plot_confusion_matrix(clf, X_test_vecs, y_test, display_labels=class_labels)
    plt.show()

    for i in range(100):
        print(y_test[i], clf.predict([X_test_vecs[i]]))
    pickle.dump(clf, open('pretrained/model.pkl', 'wb'))


if __name__ == '__main__':
    train()
