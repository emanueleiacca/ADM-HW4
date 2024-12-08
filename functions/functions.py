import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import pandas as pd

# Section 1.2

def jaccard_similarity_exact(movies1, movies2):
    intersection = len(movies1 & movies2)
    union = len(movies1 | movies2)
    return intersection / union if union > 0 else 0

def jaccard_similarity_hashed(signature1, signature2):
    matches = sum(1 for x, y in zip(signature1, signature2) if x == y)
    return matches / len(signature1)

def generate_hash_function(a,b,c):
    def hash_function(x):
        return (a * (x%c) + b)
    return hash_function

def generate_hash_functions(num_hashes, max_movie_id, seed=None):

    if seed is not None:
        np.random.seed(seed)
    a = np.random.randint(1, max_movie_id * 2, size=num_hashes)
    b = np.random.randint(0, max_movie_id * 2, size=num_hashes)
    c = max_movie_id * 2 + 1
    return [generate_hash_function(a_, b_, c) for a_, b_ in zip(a, b)]

def create_custom_hash_function(a_coeff, b_coeff, prime_mod):
    """"Defines a cubic hash function based on 2x."""
    def hash_function(input_value):
        return (a_coeff * 2 * input_value + b_coeff) % prime_mod
    return hash_function

def create_hash_function_family(num_functions, max_movie_val, random_seed=None):
    """Generates a series of hash functions using the coefficient 2x in the formula."""
    if random_seed is not None:
        np.random.seed(random_seed)

    a_coefficients = np.random.randint(1, max_movie_val * 2, size=num_functions)
    b_coefficients = np.random.randint(0, max_movie_val * 2, size=num_functions)
    prime_modulo = max_movie_val * 2 + 1

    return [create_custom_hash_function(a, b, prime_modulo) for a, b in zip(a_coefficients, b_coefficients)]

def create_quadratic_hash_function(coeff_a, coeff_b, prime_modulo):
    """"Defines a cubic hash function based on x^2."""
    def hash_function(input_value):
        return (coeff_a * (input_value / 2) + coeff_b * input_value) % prime_modulo
    return hash_function

def create_quadratic_hash_family(num_functions, max_val, random_seed=None):
    """"Generates a series of hash functions using the coefficient x^2 in the formula."."""
    if random_seed is not None:
        np.random.seed(random_seed)

    coeff_a_list = np.random.randint(1, max_val * 2, size=num_functions)
    coeff_b_list = np.random.randint(0, max_val * 2, size=num_functions)
    prime_modulo = max_val * 2 + 1

    return [create_quadratic_hash_function(a, b, prime_modulo) for a, b in zip(coeff_a_list, coeff_b_list)]

def create_cubic_hash_function(coeff_a, coeff_b, prime_mod):
    """"Defines a cubic hash function based on x^3."""
    def hash_function(input_val):
        return (coeff_a * input_val**3 + coeff_b) % prime_mod
    return hash_function

def create_cubic_hash_family(num_functions, max_val, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    a_coeffs = np.random.randint(1, max_val * 2, size=num_functions)
    b_coeffs = np.random.randint(0, max_val * 2, size=num_functions)
    prime_modulo = max_val * 2 + 1

    return [create_cubic_hash_function(a, b, prime_modulo) for a, b in zip(a_coeffs, b_coeffs)]

def generate_signatures(subset_users, user_movie_data, hash_functions):
    """"Create MinHash signatures for a specific group of users."""
    signatures = {}

    for user_id in subset_users:
        watched_movies = user_movie_data.get(user_id, set())

        if not watched_movies:

            signatures[user_id] = [float('inf')] * len(hash_functions)
            continue

        user_signature = []
        for hash_func in hash_functions:

            min_hash_value = min(hash_func(movie) for movie in watched_movies)
            user_signature.append(min_hash_value)

        signatures[user_id] = user_signature

    return signatures

def compute_mse(user_movie_data, signatures, selected_users):
    """Calculate the mean squared error (MSE) between real and estimated similarities for a set of users"""
    total_error = 0
    pair_count = 0

    for idx, user_a in enumerate(selected_users):
        for user_b in selected_users[idx + 1:]:
            # Real similarity based on the Jaccard definition.
            actual_jaccard = jaccard_similarity_hashed(user_movie_data[user_a], user_movie_data[user_b])

            # "Estimated similarity using MinHash signatures."
            estimated_jaccard = jaccard_similarity_hashed(signatures[user_a], signatures[user_b])


            total_error += (actual_jaccard - estimated_jaccard) ** 2
            pair_count += 1

    # Calculate MSE
    return (total_error / pair_count) if pair_count > 0 else 0

# Section 1.3

def LSH_Scratch(minhash_signatures, num_bands, rows_per_band):
    """
    Locality-Sensitive Hashing (LSH) implementation for clustering similar users based on MinHash signatures.

    Parameters:
    - minhash_signatures (dict): A dictionary where keys are user IDs and values are MinHash signatures.
    - num_bands (int): Number of bands to divide each MinHash signature into.
    - rows_per_band (int): Number of rows in each band.

    Returns:
    - global_buckets (defaultdict): A mapping of hash values to user IDs grouped across all bands.
    - candidate_pairs (set): A set of pairs of users that are candidate matches for similarity checks.
    """

    # signature length matches the number of bands and rows per band?
    signature_length = len(next(iter(minhash_signatures.values())))
    assert num_bands * rows_per_band == signature_length, \
        "Number of bands and rows per band must match the MinHash signature length."

    # Initialize 
    global_buckets = defaultdict(list)
    candidate_pairs = set()

    # Process each band
    for band in range(num_bands):
        # slice of rows for the current band
        start_idx = band * rows_per_band
        end_idx = start_idx + rows_per_band

        # Band-specific buckets to group users with the same hash value
        band_buckets = defaultdict(list)
        for user_id, signature in minhash_signatures.items():
            band_slice = tuple(signature[start_idx:end_idx]) # Extract the portion of the signature for this band
            band_hash = hash(band_slice) # Compute a hash value for the band slice
            band_buckets[band_hash].append(user_id)


        # Add users from this band's buckets to the global buckets and identify candidate pairs
        for band_hash, users in band_buckets.items():
            if len(users) > 1:  # Only consider buckets with multiple users
                global_buckets[band_hash].extend(users)
                for pair in combinations(users, 2):
                    candidate_pairs.add(pair)

    return global_buckets, candidate_pairs

def plot_buckets(buckets, section_size=10):
    """
    Visualizes the distribution of users in buckets divided into sections for clarity.

    Parameters:
    - global_buckets (defaultdict): A mapping of hash values to user IDs grouped across all bands.
    - section_size (int): Number of buckets to display per section.
    """
    # Prepare data
    bucket_sizes = [(bucket_hash, len(users)) for bucket_hash, users in buckets.items() if len(users) > 1]
    total_buckets = len(bucket_sizes)

    # Divide into sections
    for i in range(0, total_buckets, section_size):
        section = bucket_sizes[i:i + section_size]
        bucket_labels = [f"Hash {hash_val}" for hash_val, _ in section]
        bucket_values = [size for _, size in section]

        # Plot the section
        plt.figure(figsize=(12, 6))
        plt.bar(bucket_labels, bucket_values, alpha=0.7, edgecolor='black')
        plt.title(f"Bucket Sizes (Section {i // section_size + 1})", fontsize=16)
        plt.xlabel("Bucket Hash", fontsize=12)
        plt.ylabel("Number of Users", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def find_similar_users(user, buckets, user_movie_dict=None, minhash_signatures=None, num_bands=None, rows_per_band=None):
    """
    Identifies the two most similar users to a given user based on bucket placement.

    Parameters:
        user (int): The user for whom to find similar users.
        buckets (dict): Dictionary of buckets mapping hashes to lists of users.
        user_movie_dict (dict, optional): Dictionary mapping users to their movie sets for additional similarity checks.
        minhash_signatures (dict, optional): Dictionary of minhash signatures for LSH parameter adjustment.
        num_bands (int, optional): Number of bands used in LSH.
        rows_per_band (int, optional): Rows per band used in LSH.

    Returns:
        list: A list of the two most similar users to the given user.
    """
    from collections import Counter

    # Step 1: Check current buckets for similar users
    similar_users = Counter()
    for bucket_users in buckets.values():
        if user in bucket_users:
            for other_user in bucket_users:
                if other_user != user:
                    similar_users[other_user] += 1

    # Step 2: If no similar users are found, adjust LSH parameters
    if not similar_users and minhash_signatures and num_bands and rows_per_band:
        print("No similar users found. Adjusting LSH parameters...")
        num_bands = max(1, num_bands - 1)  # Decrease bands to increase granularity
        rows_per_band = len(next(iter(minhash_signatures.values()))) // num_bands
        _, updated_buckets = LSH_Scratch(minhash_signatures, num_bands, rows_per_band)
        return find_two_most_similar_users(
            user, updated_buckets, user_movie_dict, minhash_signatures, num_bands, rows_per_band
        )

    # Step 3: Rank users by number of shared buckets
    ranked_users = similar_users.most_common()

    # Step 4: Add additional criteria (e.g., Jaccard similarity) if user_movie_dict is provided
    if user_movie_dict:
        def jaccard_similarity(user1, user2):
            set1 = user_movie_dict.get(user1, set())
            set2 = user_movie_dict.get(user2, set())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0

        ranked_users = sorted(
            ranked_users,
            key=lambda x: (x[1], jaccard_similarity(user, x[0])),  # Sort by buckets and Jaccard
            reverse=True
        )

    # Step 5: Return the top 2 users
    return [user for user, _ in ranked_users[:2]]

# Function to create a rating dictionary from a DataFrame
def create_rating_dict(rating_df):
    """
    Creates a dictionary mapping each userId to their rated movieId and corresponding rating.

    Parameters:
        rating_df (pd.DataFrame): DataFrame containing columns ['userId', 'movieId', 'rating'].

    Returns:
        dict: A dictionary in the format:
              {
                  userId1: {movieId1: rating1, movieId2: rating2, ...},
                  userId2: {movieId3: rating3, movieId4: rating4, ...},
                  ...
              }
    """
    grouped = rating_df_pd.groupby('userId')
    user_ratings = {
        user: dict(zip(group['movieId'], group['rating']))
        for user, group in grouped
    }
    return user_ratings

# Function to recommend movies based on similar users
def recommend_movies(user, similar_users, user_movie_dict, rating_dict, max_recommendations=5):
    """
    Recommends movies to a user based on the ratings of the two most similar users.

    Parameters:
        user (int): Target user for whom recommendations are generated.
        similar_users (list): List of two most similar users.
        user_movie_dict (dict): Dictionary mapping users to the movies they have rated.
        rating_dict (dict): Dictionary mapping users to their movie ratings as {movieId: rating}.
        max_recommendations (int): Maximum number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame with columns ['movieId', 'score'] containing the recommendations.
    """
    watched_movies = user_movie_dict.get(user, set())
    movies_user1 = set(rating_dict.get(similar_users[0], {}).keys())
    movies_user2 = set(rating_dict.get(similar_users[1], {}).keys())

    # Commonly rated movies
    common_movies = movies_user1 & movies_user2
    recommendations = {}

    if common_movies:
        for movie in common_movies:
            avg_rating = (rating_dict[similar_users[0]][movie] + rating_dict[similar_users[1]][movie]) / 2
            recommendations[movie] = avg_rating

    if not recommendations:
        most_similar_user = similar_users[0]
        for movie, rating in rating_dict.get(most_similar_user, {}).items():
            if movie not in watched_movies:
                recommendations[movie] = rating

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:max_recommendations]

    return pd.DataFrame(sorted_recommendations, columns=['movieId', 'score'])


# Preprocessing for se
def check_memory():
    memory_pool = cp.get_default_memory_pool()
    total_memory = cp.cuda.Device(0).mem_info[0]  # Total memory available
    used_memory = memory_pool.used_bytes()       # Memory allocated by the pool
    free_memory = total_memory - used_memory
    print(f"GPU Memory - Total: {total_memory / (1024**3):.2f} GB, Used: {used_memory / (1024**3):.2f} GB, Free: {free_memory / (1024**3):.2f} GB")

# KMeans and Kmeans++ for Section 2.3
def initialize_centroids(data, k, method="random",seed=42):
    """
    Initialize centroids using the chosen method.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic initialization or "kmeans++" for K-me ()ans++ initialization.
    """
    if method == "random":
        np.random.seed(seed)  # Set the random seed for reproducibility
        # Randomly select k unique indices
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

    elif method == "kmeans++":
        np.random.seed(seed)
        # K-means++ initialization
        centroids = [data[np.random.choice(data.shape[0])]]  # First centroid randomly chosen
        for _ in range(1, k):
            # Compute distances from nearest centroid for all points
            distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in centroids], axis=0)
            # Compute probabilities proportional to squared distances
            probabilities = distances ** 2 / np.sum(distances ** 2)
            # Choose next centroid based on probabilities
            next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
            centroids.append(data[next_centroid_index])
        return np.array(centroids)

    else:
        raise ValueError("Invalid method. Choose 'random' or 'kmeans++'.")

def compute_distance(point, centroids):
    """Compute the distance of a point to all centroids and return the nearest one."""
    distances = np.linalg.norm(centroids - point, axis=1)
    return np.argmin(distances)  # Return the index of the closest centroid

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid."""
    clusters = []
    for point in data:
        cluster_id = compute_distance(point, centroids)
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Update centroids as the mean of points in each cluster."""
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:  # Handle empty cluster
            new_centroids.append(np.zeros(data.shape[1]))
    return np.array(new_centroids)

def kmeans(data, k, method="random", max_iterations=100, tolerance=1e-4, seed = 42):
    """
    K-means clustering algorithm with option for basic or K-means++ initialization.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic K-means or "kmeans++" for K-means++.
        - max_iterations: Maximum number of iterations.
        - tolerance: Convergence tolerance.
    """
    # Initialize centroids
    centroids = initialize_centroids(data, k, method=method)

    for iteration in range(max_iterations):
        # Assign clusters
        clusters = assign_clusters(data, centroids)

        # Update centroids
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids, clusters

# KMeans Tracking Section 3

def kmeans_iterations(data, k, method="random", max_iterations=100, tolerance=1e-4):
    """
    Perform K-means clustering and track iterations.
    
    Parameters:
        data: numpy.ndarray
            The dataset to cluster.
        k: int
            Number of clusters.
        method: str
            Initialization method ("random" or "kmeans++").
        max_iterations: int
            Maximum number of iterations.
        tolerance: float
            Convergence tolerance.
    
    Returns:
        centroids_history: list of numpy.ndarray
            History of centroid positions at each iteration.
        cluster_history: list of numpy.ndarray
            History of cluster assignments at each iteration.
    """
    """
    It's the same function we already did before in our KMeans clustering Function, 
    this time btw we need to store each iteration to retrieve it later, 
    using the already implemented function was saving only the initial and final iterations
    """
    centroids = initialize_centroids(data, k, method)  # Initialize centroids
    centroids_history = [centroids]  # Store initial centroids
    cluster_history = []  # Store cluster assignments for each iteration

    for iteration in range(max_iterations):
        # Assign clusters based on current centroids
        clusters = assign_clusters(data, centroids)
        cluster_history.append(clusters)

        # Update centroids
        new_centroids = update_centroids(data, clusters, k)
        centroids_history.append(new_centroids)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids_history, cluster_history

def visualize_kmeans_iterations(data, k, selected_features, selected_features_labels, method="random", max_iterations=100):
    """
    Visualize K-means clustering progress over iterations using the predefined kmeans_iterations function.

    Parameters:
    - data: The dataset (NumPy array or DataFrame) for clustering.
    - k: Number of clusters.
    - selected_features: Indices of features to use for visualization.
    - selected_features_labels: Names of the selected features for plotting.
    - method: Initialization method for centroids ("random" or "kmeans++").
    - max_iterations: Maximum number of iterations to run K-means.
    """
    selected_data = data[:, selected_features]

    centroids_history, cluster_history = kmeans_iterations(data, k=k, method=method, max_iterations=max_iterations)

    # Visualize 
    for iteration, (centroids, clusters) in enumerate(zip(centroids_history, cluster_history)):
        plt.figure(figsize=(8, 6))
        for cluster_id in np.unique(clusters):
            cluster_points = selected_data[clusters == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
        plt.scatter(centroids[:, selected_features[0]], centroids[:, selected_features[1]],
                    color='red', marker='x', s=100, label='Centroids')
        plt.title(f"Iteration {iteration + 1}")
        plt.xlabel(selected_features_labels[0])
        plt.ylabel(selected_features_labels[1])
        plt.legend()
        plt.show()

    print("Visualization complete.")

# AQ Question

def minimax(nums, start, end, who):
  #Base case for the recursion, which occurs after we've picked all the elements (start & end were the same index in the previous call).
  #If so, Arya has no more gain left.
  if start > end:
    return 0

  if who == 1:
    #Arya's turn, so we look for the choice that maximizes her score. Her score is calculated as the value of the current number (start or end),
    #added to the recursive minimax value of the remaining subarray.
    pick_start = nums[start] + minimax(nums, start + 1, end, 2)
    pick_end = nums[end] + minimax(nums, start, end - 1, 2)
    return max(pick_start, pick_end)

  else:
    #Mario's turn, so he minimizes Arya's score.
    pick_start = minimax(nums, start + 1, end, 1)
    pick_end = minimax(nums, start, end - 1, 1)
    return min(pick_start, pick_end)

def canAryaWin(nums):
  total_sum = sum(nums)

  arya_score = minimax(nums, 0, len(nums) - 1, 1)
  print("(Brute-force Minimax) Arya's score: " + str(arya_score) + " Total sum: " + str(total_sum))

  try:
    ratio = arya_score / total_sum
  except:
    #edge case if the total sum is zero, to avoid zero division  errors. If total_sum is 0, Arya and Mario are tied at 0, so Arya wins.
    return "True"

  if ratio >= 0.5:
    return "True"
  else:
    return "False"

def minimax_memo(nums, start, end, who, memo):

  if start > end:
    return 0

  #If the result for the current call has already been stored, don't recompute.
  if (start, end, who) in memo:
    return memo[(start, end, who)]

  if who == 1:
    pick_start = nums[start] + minimax_memo(nums, start + 1, end, 2, memo)
    pick_end = nums[end] + minimax_memo(nums, start, end - 1, 2, memo)
    result = max(pick_start, pick_end)

  else:
    pick_start = minimax_memo(nums, start + 1, end, 1, memo)
    pick_end = minimax_memo(nums, start, end - 1, 1, memo)
    result = min(pick_start, pick_end)

  #Store result in memoization dictionary.
  memo[(start, end, who)] = result
  return result

def canAryaWinMemo(nums):
  total_sum = sum(nums)
  #Initialize an empty dictionary.
  memo = {}

  arya_score = minimax_memo(nums, 0, len(nums) - 1, 1, memo)
  print("(Minimax with Memoization) Arya's score: " + str(arya_score) + " Total sum: " + str(total_sum))

  try:
    ratio = arya_score / total_sum
  except ZeroDivisionError:
    return "True"

  if ratio >= 0.5:
    return "True"
  else:
    return "False"

def PredictTheWinner(nums):
    n = len(nums)

    # dp[i][j] represents the maximum score difference (Arya's score - Mario's score) for the subarray nums[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # Base case: when i == j, the subarray contains only one element, so Arya takes it
    for i in range(n):
        dp[i][i] = nums[i]

    # Fill the DP table for subarrays of increasing length
    for length in range(2, n + 1):  # length of subarray
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])

    # If dp[0][n-1] >= 0, Arya can guarantee a win, else she cannot
    return dp[0][n - 1] >= 0

