import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from surprise import NMF
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanSquaredError

from sklearn.ensemble import RandomForestClassifier

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def load_profiles():
    return pd.read_csv("user_profile.csv")


def load_genres():
    return pd.read_csv("course_genre.csv")

def load_user_embeddings():
    return pd.read_csv("user_embeddings.csv")

def load_course_embeddings():
    return pd.read_csv("course_embeddings.csv")

def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                
                if unselect_course not in res:
                        res[unselect_course] = sim
                else:
                        if sim >= res[unselect_course]:
                            res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def generate_recommendation_scores(idx_id_dict, enrolled_course_ids, sim_threshold):
    courses = []
    scores = []
    genres_df = load_genres()
    
    # get user vector for the current user id
    course_df = genres_df['COURSE_ID'].isin(enrolled_course_ids)
    course_df = genres_df[course_df]
    user_genres = course_df.drop(['COURSE_ID', 'TITLE'], axis=1)
    user_vector = user_genres.sum()
    user_vector = pd.DataFrame(user_vector).T
    user_vector = user_vector.iloc[0, 0:].values
        
    # get the unknown course ids for the current user id
    all_courses = set(idx_id_dict.values())
    unselected_courses = all_courses.difference(enrolled_course_ids)
    unknown_genres = genres_df[genres_df['COURSE_ID'].isin(unselected_courses)]
    unselected_course_ids = unknown_genres['COURSE_ID'].values
    course_df = unknown_genres.iloc[:, 2:].values

    # user np.dot() to get the recommendation scores for each course
    recommendation_scores = np.dot(course_df, user_vector)

    # Append the results into the users, courses, and scores list
    
    for i in range(0, len(unselected_course_ids)):
        score = recommendation_scores[i]
        # Only keep the courses with high recommendation score
        if score >= sim_threshold:
            courses.append(unselected_course_ids[i])
            scores.append(recommendation_scores[i])


    return courses, scores

def k_clusters(k, enrolled_course_ids, enrollments):
    profile_df = load_profiles()
    genres_df = load_genres()
    rec = {}

    course_df = genres_df['COURSE_ID'].isin(enrolled_course_ids)
    course_df = genres_df[course_df]
    user_genres = course_df.drop(['COURSE_ID', 'TITLE'], axis=1)
    user_vector = user_genres.sum()
    user_vector = pd.DataFrame(user_vector).T
    user_vector = user_vector.iloc[0, 0:].values
    user_vector = list(user_vector)

    user_vector.insert(0, 9999999)
    profile_df.loc[profile_df.shape[0]] = user_vector

    user_ids = profile_df['user']
    profile_df = profile_df.drop('user', axis=1)
    ss = StandardScaler()
    profile_df = ss.fit_transform(profile_df)
    km = KMeans(n_clusters=k, init='k-means++')
    km.fit(profile_df)
    km_labels = km.labels_
    labels_df = pd.DataFrame(km_labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']

    user_label = cluster_df[cluster_df['user'] == 9999999]
    user_label = list(user_label['cluster'])
    user_label =user_label[0]

    ratings_df = load_ratings()
    ratings_df = ratings_df[['user','item']]
    merged_df = pd.merge(ratings_df, cluster_df, left_on='user', right_on='user')
    merged_df = merged_df.query('cluster == @user_label')

    merged_df = merged_df[['item']]
    merged_df['count'] = [1]*len(merged_df)
    merged_df = merged_df.groupby(['item']).agg(enrollments = ('count','sum')).reset_index()

    en = enrollments
    queried = merged_df.query('enrollments > @en')
    recommends = set(queried['item'].tolist())
    enrolled = set(enrolled_course_ids)
    recommends = recommends.difference(enrolled)

    for course in recommends:
        rec[course] = int(queried[queried['item'] == course]['enrollments'].values)

    return rec

def k_clusters_with_pca(k, p, enrolled_course_ids, enrollments):
    profile_df = load_profiles()
    genres_df = load_genres()
    rec = {}

    course_df = genres_df['COURSE_ID'].isin(enrolled_course_ids)
    course_df = genres_df[course_df]
    user_genres = course_df.drop(['COURSE_ID', 'TITLE'], axis=1)
    user_vector = user_genres.sum()
    user_vector = pd.DataFrame(user_vector).T
    user_vector = user_vector.iloc[0, 0:].values
    user_vector = list(user_vector)

    user_vector.insert(0, 9999999)
    profile_df.loc[profile_df.shape[0]] = user_vector

    user_ids = profile_df['user']
    profile_df = profile_df.drop('user', axis=1)
    ss = StandardScaler()
    profile_df = ss.fit_transform(profile_df)

    pca = PCA(n_components=p)
    pca.fit(profile_df)
    profile_df = pd.DataFrame(pca.transform(profile_df), columns=[f'PC{i}' for i in range(0,p)])    

    km = KMeans(n_clusters=k, init='k-means++')
    km.fit(profile_df)
    km_labels = km.labels_
    labels_df = pd.DataFrame(km_labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']

    user_label = cluster_df[cluster_df['user'] == 9999999]
    user_label = list(user_label['cluster'])
    user_label =user_label[0]

    ratings_df = load_ratings()
    ratings_df = ratings_df[['user','item']]
    merged_df = pd.merge(ratings_df, cluster_df, left_on='user', right_on='user')
    merged_df = merged_df.query('cluster == @user_label')

    merged_df = merged_df[['item']]
    merged_df['count'] = [1]*len(merged_df)
    merged_df = merged_df.groupby(['item']).agg(enrollments = ('count','sum')).reset_index()

    en = enrollments
    queried = merged_df.query('enrollments > @en')
    recommends = set(queried['item'].tolist())
    enrolled = set(enrolled_course_ids)
    recommends = recommends.difference(enrolled)

    for course in recommends:
        rec[course] = int(queried[queried['item'] == course]['enrollments'].values)

    return rec

def KNNeighbours(enrolled_course_ids, n):
    res = {}
    ratings_df = load_ratings()
    all_courses = set(ratings_df['item'])
    new_courses = all_courses.difference(enrolled_course_ids)

    for course in enrolled_course_ids:
        ratings_df.loc[ratings_df.shape[0]] = 9999999, course, 3.0
    
    reader = Reader(rating_scale=(2, 3))
    course_dataset = Dataset.load_from_df(ratings_df[['user','item','rating']], reader=reader)
    trainset, testset = train_test_split(course_dataset, test_size=.3)
    algo = KNNBasic(min_k=n)
    algo.fit(trainset)
    #predictions = algo.test(testset)
    #acc = accuracy.rmse(predictions)
    for course in new_courses:
        rating = algo.predict(9999999, course)
        res[course] = rating.est
    return res

def NonMatrixFact(enrolled_course_ids):
    res = {}
    ratings_df = load_ratings()
    all_courses = set(ratings_df['item'])
    new_courses = all_courses.difference(enrolled_course_ids)

    for course in enrolled_course_ids:
        ratings_df.loc[ratings_df.shape[0]] = 9999999, course, 3.0
    
    reader = Reader(rating_scale=(2, 3))
    course_dataset = Dataset.load_from_df(ratings_df[['user','item','rating']], reader=reader)
    trainset, testset = train_test_split(course_dataset, test_size=.3)
    algo = NMF(random_state=123, n_epochs=70, n_factors=14)
    algo.fit(trainset)
    for course in new_courses:
        rating = algo.predict(9999999, course)
        res[course] = rating.est
    return res

class RecommenderNet(keras.Model):
    
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """
           Constructor
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")
        
        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")
        
    def call(self, inputs):
        """
           method to be called during model fitting
           
           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Sigmoid output layer to output the probability
        return tf.nn.relu(x)

def process_dataset(raw_data):
    
    encoded_data = raw_data.copy()
    
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    
    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_id2idx_dict, course_id2idx_dict

def generate_train_test_datasets(dataset, scale=True):

    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Assuming training on 80% of the data and validating on 10%, and testing 10%
    indices = int(0.8 * dataset.shape[0])
    #test_indices = int(0.9 * dataset.shape[0])

    x_train, x_val, y_train, y_val = (
        x[:indices],
        x[indices:],
        y[:indices],
        y[indices:],
    )
    return x_train, x_val, y_train, y_val

def NeuralNetwork(enrolled_course_ids):
    ratings_df = load_ratings()
    all_courses = ratings_df['item'].unique()
    new_courses = list(set(all_courses).difference(set(enrolled_course_ids)))

    for course in enrolled_course_ids:
        ratings_df.loc[ratings_df.shape[0]] = 9999999, course, 3.0
    
    encoded_data, user_id2idx_dict, course_id2idx_dict = process_dataset(ratings_df)

    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())

    model = RecommenderNet(num_users, num_items, 16)
    model.compile(
    loss='MeanSquaredError',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics='RootMeanSquaredError'
)

    x_train, x_val, y_train, y_val = generate_train_test_datasets(encoded_data, scale=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)

    user_id = user_id2idx_dict.get(9999999)
    new_courses_enc = [[course_id2idx_dict.get(x)] for x in new_courses]
    user_course_df = pd.DataFrame(new_courses_enc, ([user_id]*len(new_courses_enc))).reset_index()
    user_course_array = (np.asarray(user_course_df)).astype('float32')
    scores = model.predict(user_course_array).flatten()
    courses = new_courses

    return courses, scores

def RegressionEmbedding(enrolled_course_ids):
    #user_emb = load_user_embeddings()
    #course_emb = load_course_embeddings()
    ratings_df = load_ratings()
    all_courses = ratings_df['item'].unique()
    new_courses = list(set(all_courses).difference(set(enrolled_course_ids)))

    for course in enrolled_course_ids:
        ratings_df.loc[ratings_df.shape[0]] = 9999999, course, 3.0
    
    encoded_data, user_id2idx_dict, course_id2idx_dict = process_dataset(ratings_df)

    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())

    model = RecommenderNet(num_users, num_items, 16)
    model.compile(
    loss='MeanSquaredError',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics='RootMeanSquaredError'
)

    x_train, x_val, y_train, y_val = generate_train_test_datasets(encoded_data, scale=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)

    user_latent = model.get_layer('user_embedding_layer').get_weights()[0]
    course_latent = model.get_layer('item_embedding_layer').get_weights()[0]

    user_names = ratings_df['user'].unique()
    course_names = ratings_df['item'].unique()

    u_features = [f"UFeature{i}" for i in range(16)]
    c_features = [f"CFeature{i}" for i in range(16)]

    user_emb = pd.DataFrame(user_latent)
    user_emb.columns = u_features
    user_emb.insert(0, 'user', user_names)

    course_emb = pd.DataFrame(course_latent)
    course_emb.columns = c_features
    course_emb.insert(0, 'item', course_names)

    user_emb_merged = pd.merge(ratings_df, user_emb, how='left', left_on='user', right_on='user').fillna(0)
    merged_df = pd.merge(user_emb_merged, course_emb, how='left', left_on='item', right_on='item').fillna(0)

    #u_features = [f"UFeature{i}" for i in range(16)]
    #c_features = [f"CFeature{i}" for i in range(16)]

    user_embeddings = merged_df[u_features]
    course_embeddings = merged_df[c_features]
    ratings = merged_df['rating']

# Aggregate the two feature columns using element-wise add
    reg_data = user_embeddings + course_embeddings.values
    reg_data.columns = [f"Feature{i}" for i in range(16)]
    reg_data['rating'] = ratings

    X = reg_data.iloc[:, :-1]
    y = reg_data.iloc[:, -1]

    lr = LinearRegression()
    lr.fit(X, y)

    course_df = course_emb.copy()
    course_df = course_df.loc[~course_df['item'].isin(enrolled_course_ids)]

    user_id = user_emb[user_emb['user'] == 9999999].reset_index().drop('index', axis=1)
    u_features2 = u_features.copy()
    u_features2.insert(0,'user')
    user_df = pd.DataFrame(columns=u_features2)
    for i in range(0, course_df.shape[0]):
        user_df.loc[i] = user_id.loc[0].values

    user_emb2 = user_df[u_features]
    course_emb2 = course_df[c_features]
    reg_data = user_emb2 + course_emb2.values
    reg_data.columns = [f"Feature{i}" for i in range(16)]

    scores = lr.predict(reg_data)
    courses = course_df['item']

    return courses, scores

def ClassificationEmbedding(enrolled_course_ids):
    #user_emb = load_user_embeddings()
    #course_emb = load_course_embeddings()
    ratings_df = load_ratings()
    all_courses = ratings_df['item'].unique()
    new_courses = list(set(all_courses).difference(set(enrolled_course_ids)))

    for course in enrolled_course_ids:
        ratings_df.loc[ratings_df.shape[0]] = 9999999, course, 3.0
    
    encoded_data, user_id2idx_dict, course_id2idx_dict = process_dataset(ratings_df)

    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())

    model = RecommenderNet(num_users, num_items, 16)
    model.compile(
    loss='MeanSquaredError',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics='RootMeanSquaredError'
)

    x_train, x_val, y_train, y_val = generate_train_test_datasets(encoded_data, scale=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)

    user_latent = model.get_layer('user_embedding_layer').get_weights()[0]
    course_latent = model.get_layer('item_embedding_layer').get_weights()[0]

    user_names = ratings_df['user'].unique()
    course_names = ratings_df['item'].unique()

    u_features = [f"UFeature{i}" for i in range(16)]
    c_features = [f"CFeature{i}" for i in range(16)]

    user_emb = pd.DataFrame(user_latent)
    user_emb.columns = u_features
    user_emb.insert(0, 'user', user_names)

    course_emb = pd.DataFrame(course_latent)
    course_emb.columns = c_features
    course_emb.insert(0, 'item', course_names)

    user_emb_merged = pd.merge(ratings_df, user_emb, how='left', left_on='user', right_on='user').fillna(0)
    merged_df = pd.merge(user_emb_merged, course_emb, how='left', left_on='item', right_on='item').fillna(0)

    #u_features = [f"UFeature{i}" for i in range(16)]
    #c_features = [f"CFeature{i}" for i in range(16)]

    user_embeddings = merged_df[u_features]
    course_embeddings = merged_df[c_features]
    ratings = merged_df['rating']

# Aggregate the two feature columns using element-wise add
    reg_data = user_embeddings + course_embeddings.values
    reg_data.columns = [f"Feature{i}" for i in range(16)]
    reg_data['rating'] = ratings

    X = reg_data.iloc[:, :-1]
    y_raw = reg_data.iloc[:, -1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.values.ravel())


    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X, y)

    course_df = course_emb.copy()
    course_df = course_df.loc[~course_df['item'].isin(enrolled_course_ids)]

    user_id = user_emb[user_emb['user'] == 9999999].reset_index().drop('index', axis=1)
    u_features2 = u_features.copy()
    u_features2.insert(0,'user')
    user_df = pd.DataFrame(columns=u_features2)
    for i in range(0, course_df.shape[0]):
        user_df.loc[i] = user_id.loc[0].values

    user_emb2 = user_df[u_features]
    course_emb2 = course_df[c_features]
    class_data = user_emb2 + course_emb2.values
    class_data.columns = [f"Feature{i}" for i in range(16)]

    scores = rf.predict(class_data)
    courses = course_df['item']

    return courses, scores

# Model training
def train(model_name, params, selected_courses_df):
    #if model_name == models[0]:
        
    # TODO: Add model training code here
    pass


# Prediction
def predict(model_name, user_ids, params, selected_courses_df):
    sim_threshold = 0.6
    top_courses = 10
    n = 4
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    if "top_courses" in params:
        top_courses = params["top_courses"]
    if "cluster_no" in params:
        k = params["cluster_no"]
    if "enrollments" in params:
        enrollments = params["enrollments"]
    if "components" in params:
        p = params["components"]
    if "neighbours" in params:
        n = params["neighbours"]

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    courses = []
    scores = []
    res_dict = {}
            # Course Similarity model
    if model_name == models[0]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)

        for key, score in res.items():
            if score >= sim_threshold:
                courses.append(key)
                scores.append(score)
        # TODO: Add prediction model code here

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)
        
        return res_df
    
    if model_name == models[1]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        courses, scores = generate_recommendation_scores(idx_id_dict, enrolled_course_ids, sim_threshold)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df

    if model_name == models[2]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        res = k_clusters(k, enrolled_course_ids, enrollments)

        for key, score in res.items():
            courses.append(key)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df

    if model_name == models[3]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        res = k_clusters_with_pca(k, p, enrolled_course_ids, enrollments)

        for key, score in res.items():
            courses.append(key)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df
    
    if model_name == models[4]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        res = KNNeighbours(enrolled_course_ids, n)

        for key, score in res.items():
            courses.append(key)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df

    if model_name == models[5]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        res = NonMatrixFact(enrolled_course_ids)

        for key, score in res.items():
            courses.append(key)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df

    if model_name == models[6]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        courses, scores = NeuralNetwork(enrolled_course_ids)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df

    if model_name == models[7]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        courses, scores = RegressionEmbedding(enrolled_course_ids)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df

    if model_name == models[8]:
        enrolled_course_ids = selected_courses_df['COURSE_ID']
        courses, scores = ClassificationEmbedding(enrolled_course_ids)

        res_dict['COURSE_ID'] = courses
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])
        res_df = res_df.sort_values(by='SCORE', ascending=False)
        if res_df.shape[0] > top_courses:
            res_df = res_df.head(top_courses)

        return res_df