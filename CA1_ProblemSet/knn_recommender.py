from similarity import Efficient_Similarity
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
import time


class KNNRecommender:
    """Implement the required functions for Q2"""
    def __init__(self, similarity_matrix, ratings_matrix, weight_type="item-based", k=20):
        #TODO: implement any necessary initalization function here
        # sim_func is one of the similarity functions implemented in similarity.py 
        # weight_type is one of ["item-based", "user-based"]
        # k is the size of nearest neighbor set 
        #You can add more input parameters as needed.
        self.similarity_matrix = similarity_matrix  # [num_items x num_items] 或 [num_users x num_users]
        self.ratings_matrix = ratings_matrix  # [num_users x num_items]
        self.weight_type = weight_type
        self.k = k

    def rating_predict(self, userID, itemID):
        #TODO: implement the rating prediction function for a given user-item pair 
        """Predicitng the rating for a given user-item pair"""
        if userID >= self.similarity_matrix.shape[0]:  # 防止索引越界
            print(f"Warning: userID {userID} is out of bounds for similarity_matrix.")
            return 0
        if self.weight_type == "item-based":
            item_similarities = self.similarity_matrix[itemID]      # get itemID's similarity with all items
            user_ratings = self.ratings_matrix[userID]              # get user's rating for all items

            rated_items = np.where(~np.isnan(user_ratings))[0]  # indices that are already rated by the user
            if len(rated_items) == 0:
                return np.nanmean(user_ratings)
            neighbors = sorted(rated_items, key=lambda j: item_similarities[j], reverse=True)[:self.k]

            numerator = np.nansum(user_ratings[neighbors] * item_similarities[neighbors])
            denominator = np.nansum(np.abs(item_similarities[neighbors]))

        elif self.weight_type == "user-based":
            user_similarities = self.similarity_matrix[userID]  # get userID's similarity with all users
            item_ratings = self.ratings_matrix[:, itemID]  # all users' rating on itemID

            rated_users = np.where(~np.isnan(item_ratings))[0]  # list: users‘s indices that have rated ItemID 
            if len(rated_users) == 0:
                return np.nanmean(item_ratings) 
         
            neighbors = sorted(rated_users, key=lambda v: user_similarities[v], reverse=True)[:self.k]  # find topk similar users

            numerator = np.nansum(item_ratings[neighbors] * user_similarities[neighbors])
            denominator = np.nansum(np.abs(user_similarities[neighbors]))


        return numerator / denominator if denominator != 0 else 0
    
    def topk(self, userID, topk=10):
        #TODO: implement top-k recommendations for a given user
        """generate topk recommendations for a user"""
        if userID >= self.similarity_matrix.shape[0]:  # 防止索引越界
            print(f"Warning: userID {userID} is out of bounds for similarity_matrix.")
            return []
        rated_items = np.where(~np.isnan(self.ratings_matrix[userID]))[0]   # indices for this user's rated films
        all_items = np.arange(self.ratings_matrix.shape[1])
        unrated_items = np.setdiff1d(all_items, rated_items)  # get indices of films haven't been rated

        scores = [(item, self.rating_predict(userID, item)) for item in unrated_items]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]

        print(f"Top-{topk} recommendations for User {userID}: {scores}")
        return scores
    


# Some functions for Q3(c)
def get_k_neighbors(similarity_matrix, k, algorithm="brute"):
    """
    find every user's knn
    """
    num_users = similarity_matrix.shape[0]

    metric = "cosine" if algorithm == "brute" else "euclidean"

    knn = NearestNeighbors(n_neighbors=min(k, num_users), metric=metric, algorithm=algorithm)
    knn.fit(similarity_matrix)

    distances, indices = knn.kneighbors(similarity_matrix)
    return indices  

def rating_predict_knn(userID, itemID, similarity_matrix, ratings_matrix, neighbors):
    """
    Q3(c)
    """
    user_similarities = similarity_matrix[userID]
    item_ratings = ratings_matrix[:, itemID]  # get all users' rating on itemID
    rated_users = np.where(~np.isnan(item_ratings))[0]  
    if len(rated_users) == 0:
        return np.nanmean(item_ratings)  
    # get neighbors
    nearest_neighbors = [u for u in neighbors[userID] if u in rated_users]
    if len(nearest_neighbors) == 0:
        return np.nanmean(item_ratings)
    numerator = np.nansum(item_ratings[nearest_neighbors] * user_similarities[nearest_neighbors])
    denominator = np.nansum(np.abs(user_similarities[nearest_neighbors]))
    return numerator / denominator if denominator != 0 else 0

def topk_recommendations_knn(userID, topk=10, k=10, algorithm="brute"):
    """
    Q3(c)
    """
    neighbors = get_k_neighbors(cosine_similarity_matrix_user, k, algorithm=algorithm)
    rated_items = np.where(~np.isnan(ratings_matrix[userID]))[0]  
    all_items = np.arange(ratings_matrix.shape[1])
    unrated_items = np.setdiff1d(all_items, rated_items)    # only recommend unrated films
    scores = [(item, rating_predict_knn(userID, item, cosine_similarity_matrix_user, ratings_matrix, neighbors))
          for item in unrated_items]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]
    print(f"Top-{topk} recommendations for User {userID} (k={k}, algorithm={algorithm}): {scores}")
    return scores
    
    

if __name__ == '__main__':
    ratings_path="/Users/luninghao/Desktop/CSCI-SHU-381-Recsys/CA1_ProblemSet/ml-1m/ratings.dat"
    dataset = pd.read_csv(ratings_path, sep="::",engine="python",
                                   names=["UserID", "MovieID", "Rating", "Timestamp"])
    ratings_df = dataset.pivot(index="UserID", columns="MovieID", values="Rating")
    # print(ratings_df)
    ratings_matrix = ratings_df.to_numpy()
    # print(ratings_matrix)
    # get cosine and pearson sim matrices
    sim_model = Efficient_Similarity(ratings_matrix)
    cosine_similarity_matrix = sim_model.compute_similarity(weight_type="item-based", metric="cosine")
    pearson_similarity_matrix = sim_model.compute_similarity(weight_type="item-based", metric="pearson")



    # print the solution to Q3a here
    userID=381
    print("\nUsing Cosine Similarity")
    recommender_cosine = KNNRecommender(cosine_similarity_matrix, ratings_matrix, weight_type="item-based", k=10)
    top10_cosine = recommender_cosine.topk(userID=userID, topk=10)

    
    print("\nUsing Pearson Similarity")
    recommender_pearson = KNNRecommender(pearson_similarity_matrix, ratings_matrix, weight_type="item-based", k=10)
    top10_pearson = recommender_pearson.topk(userID=userID, topk=10)

    print("\nComparison of Top-10 Recommendations")
    print("\nCosine Similarity:")
    for item, score in top10_cosine:
        print(f"Movie {item}: Predicted Rating {score:.2f}")

    print("\nPearson Similarity:")
    for item, score in top10_pearson:
        print(f"Movie {item}: Predicted Rating {score:.2f}")

    print("-----------------------------------------------------------------------------------")
    
    # print the solution to Q3b here and calculate the time spent
    cosine_similarity_matrix_user = sim_model.compute_similarity(weight_type="user-based", metric="cosine")
    cosine_similarity_matrix_item = sim_model.compute_similarity(weight_type="item-based", metric="cosine")
    recommender_user_based = KNNRecommender(cosine_similarity_matrix_user, ratings_matrix, weight_type="user-based", k=10)
    recommender_item_based = KNNRecommender(cosine_similarity_matrix_item, ratings_matrix, weight_type="item-based", k=10)
    # print(f"Ratings Matrix Shape: {ratings_matrix.shape}")  # [num_users, num_items]
    # print(f"Similarity Matrix Shape: {cosine_similarity_matrix.shape}")  # [num_users, num_users]
    top10_cos_2025_user = recommender_user_based.topk(userID=2025, topk=10)
    top10_cos_2025_item = recommender_item_based.topk(userID=2025, topk=10)
    print("\nComparison of Top-10 Recommendations")
    print("\nuser-based:")
    for item, score in top10_cos_2025_user:
        print(f"Movie {item}: Predicted Rating {score:.2f}")

    print("\nitem-based:")
    for item, score in top10_cos_2025_item:
        print(f"Movie {item}: Predicted Rating {score:.2f}")
    # recommender = KNNRecommender(sim.cosine_similarity, weight_type="user-based", ratings_matrix = ratings_df)
    # recommender.fit()
    # recommender.topk(userID=381)

    print("-----------------------------------------------------------------------------------")
    # print the solution to Q3c here
    k_values = [10, 100, 1000]
    algorithms = ["brute", "kd_tree"]
    timing_results = {}
    recommendation_results = {}

    for algo in algorithms:
        for k in k_values:
            start_time = time.time()
            neighbors = get_k_neighbors(cosine_similarity_matrix_user, k, algorithm=algo)
            end_time = time.time()
            timing_results[(algo, k)] = end_time - start_time

    timing_df = pd.DataFrame(timing_results, index=["Time (seconds)"])
    print(timing_df)


    for k in k_values:
        for algo in algorithms:
            scores = topk_recommendations_knn(userID=2025, topk=10, k=k, algorithm=algo)
            recommendation_results[(k, algo)] = scores
    recommendation_df = pd.DataFrame(recommendation_results)
    print(recommendation_df)
    

    # for k in [5,10,20]:
    #     # calculate the time for each k and compare with Q3b
    #     recommender = MemoryRecommender(sim.cosine_similarity, weight_type="user-based", k=k)
    #     recommender.topk(userID=381)