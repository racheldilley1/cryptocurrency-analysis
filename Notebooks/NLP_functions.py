def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def prob_df(X, topics):
    cols = []
    for x in range(1,topics+1):
        cols.append('topic_'+str(x))

    df = pd.DataFrame(X.round(5),
             index = ['2017', '2018', '2019', '2020', '2021'],
             columns = cols)
    return df

def topic_matrix(model, feature_names, topics):
    idx = []
    for x in range(1,topics+1):
        idx.append('topic_'+str(x))

    topic_word = pd.DataFrame(model.components_.round(3),
             index = idx,
             columns = feature_names)
    
    return topic_word