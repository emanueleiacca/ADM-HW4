# **Homework 4: Movie Recommendation System**

## **Project Overview**
This project involves implementing a movie recommendation system and grouping movies using clustering techniques. Additionally, it explores an algorithmic problem using game theory. The assignment is divided into the following sections:

1. **Recommendation System with Locality-Sensitive Hashing (LSH):**
   - Implementation of a recommendation system using MinHash and LSH to identify similar users.
   - Suggestions for movies based on user similarities.

2. **Grouping Movies Together:**
   - Feature engineering and clustering to group movies with similar attributes.
   - Implementation of K-means and other clustering algorithms.

3. **Bonus Question:**
   - Visualization of the clustering process over iterations.

4. **Algorithmic Question:**
   - A game theory-based problem-solving strategy using recursion and dynamic programming.

---


<div style="text-align: center;">
  <img src="https://movielens.org/images/site/main-screen.png"/>
</div>

____

## **Project Structure**
___
- In this repository you can find:

    <br>


  > __main.ipynb__:
  
    <br>

    
    - A Jupyter Notebook where we gather all the answers and the explanations to the Research and Algorithmic Questions.
 
    <br>
     
  > __functions.py__:
    
    <br>

    - A python script where we have define the functions we have used in the `main.ipynb`
    
    <br>

  > __ Full Project Structure is the following__:

[Click here to have access to the whole Project Directory Structure](https://uithub.com/emanueleiacca/ADM-HW4)

## **Implementation Details**

### **1. Recommendation System with Locality-Sensitive Hashing (LSH)**
- **Data Preparation:**
  - Downloaded and preprocessed the MovieLens dataset.
  - Analyzed dataset structure and applied necessary cleaning steps.

- **MinHash Signatures:**
  - Implemented a custom MinHash function from scratch.
  - Experimented with various hash functions and threshold values to optimize similarity detection.

- **Locality-Sensitive Hashing (LSH):**
  - Divided MinHash signatures into bands and grouped users into buckets.
  - Recommended movies by identifying similar users in the same buckets.
  - Logic for recommendations:
    - Movies rated by both similar users were recommended based on average ratings.
    - Top-rated movies from similar users completed the recommendations.

### **2. Grouping Movies Together**
- **Feature Engineering:**
  - Created features such as genres, average ratings, and user tags for clustering.
  - Engineered additional features (e.g., number of ratings, genre counts, etc.) to represent movies effectively.

- **Data Normalization and Dimensionality Reduction:**
  - Normalized data for clustering and used PCA for dimensionality reduction when needed.

- **Clustering Algorithms:**
  - Implemented K-means and K-means++ algorithms from scratch.
  - Applied a third clustering algorithm suggested by an LLM and compared results.
  - Evaluated clustering quality using metrics like Silhouette Score, Davies-Bouldin Index, and Inertia.

### **3. Bonus Question**
- Visualized the progression of clusters over iterations using 2D plots.
- Selected features/components for effective visualization and discussed the method for selection.

### **4. Algorithmic Question**
- **Problem:** Predict Arya's chances of winning in a number-picking game against Mario.
- **Approach:**
  - Implemented a recursive solution to find the optimal strategy for Arya.
  - Optimized the algorithm using dynamic programming to achieve polynomial time complexity.
  - Compared results and runtimes between the recursive and dynamic programming solutions.
  - Consulted an LLM for a third optimized implementation and analyzed its correctness and efficiency.

---

## **Results**
- **Recommendation System:**
  - Successfully recommended movies to users based on LSH clustering.
  - Adjusted parameters to ensure high-quality recommendations.

- **Clustering:**
  - Identified optimal features and number of clusters.
  - Observed differences between K-means, K-means++, and the DBscan algorithm (LLM-recommended).
  - Evaluated clustering effectiveness using multiple metrics.

- **Algorithmic Question:**
  - Demonstrated Arya’s optimal strategy for winning or determining her chances.
  - Compared efficiency of recursive, dynamic programming, and LLM implementations.

---
