import pickle


def text_to_emotion(text):
    clf = pickle.load(open('pretrained/model.pkl', 'rb'))
    le = pickle.load(open('pretrained/le_transformer.pkl', 'rb'))
    vec = pickle.load(open('pretrained/vectorizer.pkl', 'rb'))

    text_vec = vec.transform([text])
    label = clf.predict(text_vec)
    return le.inverse_transform(label)[0]


if __name__ == '__main__':
    le = pickle.load(open('pretrained/le_transformer.pkl', 'rb'))
    print('available emotions:', le.classes_)
    text = input('Enter text: ')
    print(text_to_emotion(text))
