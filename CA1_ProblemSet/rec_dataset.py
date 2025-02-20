import pandas as pd
import matplotlib.pyplot as plt
class RecDataset:
    """Implement the required functions for Q1"""
    def __init__(self, ratings_path="/Users/luninghao/Desktop/CSCI-SHU-381-Recsys/CA1_ProblemSet/ml-1m/ratings.dat",
                 users_path="/Users/luninghao/Desktop/CSCI-SHU-381-Recsys/CA1_ProblemSet/ml-1m/users.dat"):
        #TODO: implement any necessary initalization function such as data loading here
        #You can add more input parameters as needed.
        self.ratings = pd.read_csv(ratings_path, sep="::",engine="python",
                                   names=["UserID", "MovieID", "Rating", "Timestamp"])
        self.users = pd.read_csv(users_path, sep="::", engine="python",
                                 names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        self.age_groups = {
            1: "Under 18",
            18: "18-24",
            25: "25-34",
            35: "35-44",
            45: "45-49",
            50: "50-55",
            56: "56+"
        }
        self.users["AgeGroup"] = self.users["Age"].map(self.age_groups)
    
    def describe(self):
        #TODO: implement the required functions and print the solution to Question 1a here
        n_users = self.ratings["UserID"].nunique()
        n_items = self.ratings["MovieID"].nunique()
        n_ratings = len(self.ratings)

        # compute nums of ratings for each movie
        movie_counts = self.ratings["MovieID"].value_counts()
        min_ratings = movie_counts.min()
        max_ratings = movie_counts.max()

        print(f"Number of unique users: {n_users}")
        print(f"Number of unique items: {n_items}")
        print(f"Total ratings: {n_ratings}")
        print(f"Minimum number of ratings per item: {min_ratings}")
        print(f"Maximum number of ratings per item: {max_ratings}")
        
    def query_user(self, userID):
        #TODO: implement the required functions and print the solution to Question 1b here
        user_ratings = self.ratings[self.ratings["UserID"] == userID]
        num_ratings = len(user_ratings)
        avg_rating = user_ratings["Rating"].mean() if num_ratings > 0 else None

        print(f"User {userID} has given {num_ratings} ratings.")
        if avg_rating is not None:
            print(f"The average rating by user {userID} is {avg_rating:.2f}")
        else:
            print(f"No ratings found for user {userID}.")
    
    def dist_by_age_groups(self):
        #TODO: implement the required functions and print the solution to Question 1c here
        #You could import `users.dat` here or in __init__(). 
        #This function is expected to return two lists - you shall use these lists to 
        #draw the bar plots and attach them in your answer sheet.
        merged_df = pd.merge(self.ratings, self.users, on="UserID")

        # num of ratings for each age group
        rating_counts = merged_df.groupby("AgeGroup")["Rating"].count()
        avg_ratings = merged_df.groupby("AgeGroup")["Rating"].mean()

        # plot
        # num of rating distribution
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        rating_counts.sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
        plt.xlabel("Age Group")
        plt.ylabel("Number of Ratings")
        plt.title("Distribution of Number of Ratings by Age Group")

        # avg rating distribution
        plt.subplot(1, 2, 2)
        avg_ratings.sort_index().plot(kind="bar", color="salmon", edgecolor="black")
        plt.xlabel("Age Group")
        plt.ylabel("Average Rating")
        plt.title("Distribution of Average Ratings by Age Group")

        plt.tight_layout()
        plt.show()

        return rating_counts, avg_ratings  
        

if __name__ == '__main__':
    
    dataset = RecDataset()
    
    # print the solution to Q1a here
    dataset.describe() 
    
    # print the solution to Q1b here
    dataset.query_user(userID=100)
    dataset.query_user(userID=381) 
    dataset.query_user(userID=2025)
    
    # print the solution to Q1c here
    dataset.dist_by_age_groups()