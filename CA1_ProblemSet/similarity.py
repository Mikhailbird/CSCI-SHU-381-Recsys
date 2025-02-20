import pandas as pd
import numpy as np

class Similarity:
    """Implement the required functions for Q2"""
    def __init__(self, rating_path="/Users/luninghao/Desktop/CSCI-SHU-381-Recsys/CA1_ProblemSet/ml-1m/ratings.dat"):
        #TODO: implement any necessary initalization function here
        #You can add more input parameters as needed.
        self.ratings = pd.read_csv(rating_path, sep="::", engine="python",
                                   names=["UserID", "MovieID", "Rating", "Timestamp"])
    
    def jaccard_similarity(self, item1, item2):
        #TODO: implement the required functions and print the solution to Question 2a here
        user_item1 = set(self.ratings[self.ratings["MovieID"] == item1]["UserID"])
        user_item2 = set(self.ratings[self.ratings["MovieID"] == item2]["UserID"])
        intersection = len(user_item1 & user_item2)
        union = len(user_item1 | user_item2)
        
        jaccard_score = intersection / union if union != 0 else 0.0
        print(f"Jaccard Similarity between Movie {item1} and Movie {item2}: {jaccard_score:.4f}")
        return jaccard_score


        
    def cosine_similarity(self, item1, item2):
        #TODO: implement the required functions and print the solution to Question 2b here
        # find common users of the two items
        ## UserID::Rating
        user_item1 = self.ratings[self.ratings["MovieID"] == item1][["UserID", "Rating"]].set_index("UserID")
        user_item2 = self.ratings[self.ratings["MovieID"] == item2][["UserID", "Rating"]].set_index("UserID")
        ## UserID :: R_i :: R_j and drop users who don't rate both of the items
        common_users = user_item1.join(user_item2, lsuffix='_i', rsuffix='_j').dropna()
        
        if len(common_users) == 0:
            print(f"Cosine Similarity between Movie {item1} and Movie {item2}: 0.0000")
            return 0.0
        
        # cosine sim
        dot_product = np.dot(common_users["Rating_i"], common_users["Rating_j"])
        norm_i = np.linalg.norm(common_users["Rating_i"])
        norm_j = np.linalg.norm(common_users["Rating_j"])
        cosine_sim = dot_product / (norm_i * norm_j) if norm_i != 0 and norm_j != 0 else 0

        print(f"Cosine similarity between Movie {item1} and Movie {item2}: {cosine_sim:.4f}")
        return cosine_sim

    
    def pearson_similarity(self, item1, item2):
        #TODO: implement the required functions and print the solution to Question 2c here
        user_item1 = self.ratings[self.ratings["MovieID"] == item1][["UserID", "Rating"]].set_index("UserID")
        user_item2 = self.ratings[self.ratings["MovieID"] == item2][["UserID", "Rating"]].set_index("UserID")
        common_users = user_item1.join(user_item2, lsuffix='_i', rsuffix='_j').dropna()

        if len(common_users) == 0:
            print(f"Pearson smilarity between Movie {item1} and Movie {item2}: 0.0000")
            return 0.0

        mean_i = common_users["Rating_i"].mean()
        mean_j = common_users["Rating_j"].mean()

        numerator = np.sum((common_users["Rating_i"] - mean_i) * (common_users["Rating_j"] - mean_j))
        denominator = np.sqrt(np.sum((common_users["Rating_i"] - mean_i)**2)) * np.sqrt(np.sum((common_users["Rating_j"] - mean_j)**2)) 

        Pearson_sim = numerator / denominator if denominator != 0 else 0.0
        print(f"Pearson Similarity between Movie {item1} and Movie {item2}: {Pearson_sim:.4f}")
        return Pearson_sim
    

if __name__ == '__main__':
    
    sim = Similarity()
    
    # print the solution to Q2a here
    sim.jaccard_similarity(item1=1, item2=2)
    sim.jaccard_similarity(item1=1, item2=3114)
    
    # print the solution to Q2b here
    sim.cosine_similarity(item1=1, item2=2)
    sim.cosine_similarity(item1=1, item2=3114)
    
    # print the solution to Q2c here
    sim.pearson_similarity(item1=1, item2=2)
    sim.pearson_similarity(item1=1, item2=3114)