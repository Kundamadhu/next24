project-1>> House Price Prediction System Using Data Analytics Algorithms:


In this task, you'll delve into the realm of real estate data analysis. Your goal will be to develop a predictive model that can accurately forecast house prices based on various factors.
This project will involve data collection, preprocessing, feature engineering, model selection, and evaluation.


project-2>> Ecommerce-product-recommendation-system


Product Recommendation System is a machine learning-based project that provides personalized product recommendations to users based on their browsing and purchase history. The system utilizes collaborative filtering and content-based filtering algorithms to analyze user behavior and generate relevant recommendations. This project aims to improve the overall shopping experience for users, increase sales for e-commerce businesses

Dataset
I have used an amazon dataset on user ratings for electronic products, this dataset doesn't have any headers. To avoid biases, each product and user is assigned a unique identifier instead of using their name or any other potentially biased information.

You can find the dataset here - https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation/download?datasetVersionNumber=1

You can find many other similar datasets here - https://jmcauley.ucsd.edu/data/amazon/

Approach
1) Rank Based Product Recommendation
Objective -

Recommend products with highest number of ratings.
Target new customers with most popular products.
Solve the Cold Start Problem
Outputs -

Recommend top 5 products with 50/100 minimum ratings/interactions.
Approach -

Calculate average rating for each product.
Calculate total number of ratings for each product.
Create a DataFrame using these values and sort it by average.
Write a function to get 'n' top products with specified minimum number of interactions.
2) Similarity based Collaborative filtering
Objective -

Provide personalized and relevant recommendations to users.
Outputs -

Recommend top 5 products based on interactions of similar users.
Approach -

Here, user_id is of object, for our convenience we convert it to value of 0 to 1539(integer type).
We write a function to find similar users -
Find the similarity score of the desired user with each user in the interaction matrix using cosine_similarity and append to an empty list and sort it.
extract the similar user and similarity scores from the sorted list
remove original user and its similarity score and return the rest.
We write a function to recommend users -
Call the previous similar users function to get the similar users for the desired user_id.
Find prod_ids with which the original user has interacted -> observed_interactions
For each similar user Find 'n' products with which the similar user has interacted with but not the actual user.
return the specified number of products.
3) Model based Collaborative filtering
Objective -

Provide personalized recommendations to users based on their past behavior and preferences, while also addressing the challenges of sparsity and scalability that can arise in other collaborative filtering techniques.
Outputs -

Recommend top 5 products for a particular user.
Approach -

Taking the matrix of product ratings and converting it to a CSR(compressed sparse row) matrix. This is done to save memory and computational time, since only the non-zero values need to be stored.
Performing singular value decomposition (SVD) on the sparse or csr matrix. SVD is a matrix decomposition technique that can be used to reduce the dimensionality of a matrix. In this case, the SVD is used to reduce the dimensionality of the matrix of product ratings to 50 latent features.
Calculating the predicted ratings for all users using SVD. The predicted ratings are calculated by multiplying the U matrix, the sigma matrix, and the Vt matrix.
Storing the predicted ratings in a DataFrame. The DataFrame has the same columns as the original matrix of product ratings. The rows of the DataFrame correspond to the users. The values in the DataFrame are the predicted ratings for each user.
A funtion is written to recommend products based on the rating predictions made :
It gets the user's ratings from the interactions_matrix.
It gets the user's predicted ratings from the preds_matrix.
It creates a DataFrame with the user's actual and predicted ratings.
It adds a column to the DataFrame with the product names.
It filters the DataFrame to only include products that the user has not rated.
It sorts the DataFrame by the predicted ratings in descending order.
It prints the top num_recommendations products.
Evaluating the model :
Calculate the average rating for all the movies by dividing the sum of all the ratings by the number of ratings. 2, Calculate the average rating for all the predicted ratings by dividing the sum of all the predicted ratings by the number of ratings.
Create a DataFrame called rmse_df that contains the average actual ratings and the average predicted ratings.
Calculate the RMSE of the SVD model by taking the square root of the mean of the squared errors between the average actual ratings and the average predicted ratings.
The squared parameter in the mean_squared_error function determines whether to return the mean squared error (MSE) or the root mean squared error (RMSE). When squared is set to False, the function returns the RMSE, which is the square root of the MSE. In this case, you are calculating the RMSE, so you have set squared to False. This means that the errors are first squared, then averaged, and finally square-rooted to obtain the RMSE.


project>>3 -- Twitter Sentiment Analysis

You can find the dataset here -(https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners/input?select=training.1600000.processed.noemoticon.csv)

Natural Language Processing (NLP): The discipline of computer science, artificial intelligence and linguistics that is concerned with the creation of computational models that process and understand natural language. These include: making the computer understand the semantic grouping of words (e.g. cat and dog are semantically more similar than cat and spoon), text to speech, language translation and many more

Sentiment Analysis: It is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis allows organizations to identify public sentiment towards certain words or topics.
Importing dataset¶
The dataset being used is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the Twitter API. The tweets have been annotated (0 = Negative, 4 = Positive) and they can be used to detect sentiment.

[The training data isn't perfectly categorised as it has been created by tagging the text according to the emoji present. So, any model built using this dataset may have lower than expected accuracy, since the dataset isn't perfectly categorised.]

It contains the following 6 fields:

sentiment: the polarity of the tweet (0 = negative, 4 = positive)
ids: The id of the tweet (2087)
date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (robotickilldozr)
text: the text of the tweet (Lyx is cool)
We require only the sentiment and text fields, so we discard the rest.

Furthermore, we're changing the sentiment field so that it has new values to reflect the sentiment. (0 = Negative, 1 = Positive)

Preprocess Text
Text Preprocessing is traditionally an important step for Natural Language Processing (NLP) tasks. It transforms text into a more digestible form so that machine learning algorithms can perform better.

The Preprocessing steps taken are:

Lower Casing: Each text is converted to lowercase.
Replacing URLs: Links starting with "http" or "https" or "www" are replaced by "URL".
Replacing Emojis: Replace emojis by using a pre-defined dictionary containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")
Replacing Usernames: Replace @Usernames with word "USER". (eg: "@Kaggle" to "USER")
Removing Non-Alphabets: Replacing characters except Digits and Alphabets with a space.
Removing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")
Removing Short Words: Words with length less than 2 are removed.
Removing Stopwords: Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
Lemmatizing: Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)

Analysing the data¶
Now we're going to analyse the preprocessed data to get an understanding of it. We'll plot Word Clouds for Positive and Negative tweets from our dataset and see which words occur the most.

Splitting the Data¶
The Preprocessed Data is divided into 2 sets of data:

Training Data: The dataset upon which the model would be trained on. Contains 95% data.
Test Data: The dataset upon which the model would be tested against. Contains 5% data.

TF-IDF Vectoriser¶
TF-IDF indicates what the importance of the word is in order to understand the document or dataset. Let us understand with an example. Suppose you have a dataset where students write an essay on the topic, My House. In this dataset, the word a appears many times; it’s a high frequency word compared to other words in the dataset. The dataset contains other words like home, house, rooms and so on that appear less often, so their frequency are lower and they carry more information compared to the word. This is the intuition behind TF-IDF.

TF-IDF Vectoriser converts a collection of raw documents to a matrix of TF-IDF features. The Vectoriser is usually trained on only the X_train dataset.

ngram_range is the range of number of words in a sequence. [e.g "very expensive" is a 2-gram that is considered as an extra feature separately from "very" and "expensive" when you have a n-gram range of (1,2)]

max_features specifies the number of features to consider. [Ordered by feature frequency across the corpus].

Tranforming the dataset¶
Transforming the X_train and X_test dataset into matrix of TF-IDF Features by using the TF-IDF Vectoriser. This datasets will be used to train the model and test against it.

Creating and Evaluating Models¶
We're creating 3 different types of model for our sentiment analysis problem:

Bernoulli Naive Bayes (BernoulliNB)
Linear Support Vector Classification (LinearSVC)
Logistic Regression (LR)
Since our dataset is not skewed, i.e. it has equal number of Positive and Negative Predictions. We're choosing Accuracy as our evaluation metric. Furthermore, we're plotting the Confusion Matrix to get an understanding of how our model is performing on both classification types.

Using the Model.¶
To use the model for Sentiment Prediction we need to import the Vectoriser and LR Model using Pickle.

The vectoriser can be used to transform data to matrix of TF-IDF Features. While the model can be used to predict the sentiment of the transformed Data. The text whose sentiment has to be predicted however must be preprocessed.
