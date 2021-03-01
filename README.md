# cryptocurrency-analysis

### Objective:

Identify and analyze trends in the cryptocurrrency community using natural language processing (NLP) techniques.

-----------------

### Approach:

Text data was preprocessed to remove links, punctualtion, non-aplhabetical words, change to lowercase, and to remove stop words for each document string. Text was also stemmed and lemmatized. EDA (bar graphs and word maps) and intial topic modeling were performed to find most frequently used words and to create an extensive stop word list, to better seperate topics. 

Topic modeling was done to identify the most talked about topics in cryptocurrency. Latent Semantic Allocation, Latent Direct Allocation, and Non-Negative Matrix Factorization was done with both Count Vectorization and TF-IDF Vectorization. Vader Sentiment Analysis was used on uncleaned data. Cryptocurrency prices of the most talked about cryptocurrencies was brought in to add further analysis into topics and trends. Final analysis and visualization was performed in Tableau. 

-----------------

### Featured Techniques:

* PRAW API
* Cryptocompare API
* Natural Language Processing
* Count Vectorization
* TF-IDF Vectorization
* Latent Semantic Allocation
* Latent Direct Allocation
* Non-Negative Matrix Factorization
* Vader Sentiment Analysis
* RegEx
* NLTK (Parts of speech tagging, Tokenization, Stop word removal, Lemmatization, Stemming)
* Tableau

-----------------

### Data:

Over 17,00 Reddit comments from the [r/Cryptocurrency](https://www.reddit.com/r/CryptoCurrency/) subreddit were used for NLP. Data was gathered using the [PRAW API](https://praw.readthedocs.io/en/latest/) and only the top posts from 2017 and 2021 were used. Cryptocurrency price data was gathered using the [Cryptocompare API](https://min-api.cryptocompare.com/) for Bitcoin, Ethereum, Ripple, Nano, and Doge. 

-----------------

### Results Summary:

The final topic model was a Non-Negative Matrix Factorization model was used with a Count Vectorizer that yeilded the most clear and concise topic seperation. Nine topics were chosen: Bitcoin, Ethereum, Nano, Ripple, Doge, Central Bank of Digital Currency (CBDC), Privacy, Cryptocurrency Exchanges, and Crytocurrency Trusts. Analysis and findings can be found in the Cryptocurrency Presentation pdf. 
