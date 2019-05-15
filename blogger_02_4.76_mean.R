#install.packages('tm')
#
library(tm)
library(caret)
library(dplyr)
library(caTools)  
#
setwd("~/Desktop/SYS6018/kaggle_competition/03_blogger")
train_all <- read.csv('train.csv', stringsAsFactors=FALSE)
set.seed(123)
# data split to training and validation
# returned value is True/False for each observations
split = sample.split(train_all$age, SplitRatio = 0.8) 
# if it is True, the value goes to training set
training_csv = subset(train_all, split == TRUE)              
validate_csv = subset(train_all, split == FALSE)
# ---------------------------------------------------------
# training
# create a corpus and clean this corpus
group_text <- training_csv[,'text']
group_text <- Corpus(VectorSource(group_text))
group_text <- tm_map(group_text, content_transformer(tolower))      # convert to lower case
group_text <- tm_map(group_text, removeNumbers)                     # remove numbers
group_text <- tm_map(group_text, removeWords, stopwords('english')) # remove stopwords
group_text <- tm_map(group_text, removePunctuation)                 # remove punctuation
group_text <- tm_map(group_text, stripWhitespace)                   # remove extra white space
group_text <- tm_map(group_text, stemDocument)                      # change words to original form
training_group_text <- group_text
#
# The following way is from Professor, but it does not have good result, so I did not use it finally.
# recompute TF-IDF matrix using the cleaned corpus
#group_text.tfidf = DocumentTermMatrix(training_group_text, control = list(weighting = weightTfIdf))
## reinspect the first 5 documents and first 5 terms
#group_text.tfidf[1:5,1:5]
#as.matrix(group_text.tfidf[1:5,1:5])
## we've still got a very sparse document-term matrix. remove sparse terms at various thresholds.
#group_text_top = removeSparseTerms(group_text.tfidf, 0.90)  # remove terms that are absent from at least 99% of documents (keep most terms)
#group_text_top[1:5,1:5]
#as.matrix(group_text_top[1:5,1:5])
## get the most frequent words/terms
#top_words <- colnames(as.matrix(group_text_top))
#top_words
#length(top_words)
#x_train_df <- as.data.frame(as.matrix(group_text_top))
#x_train_df[1:5,]

# get words frequency of the cleaned corpus (changed to sparse matrix)
word_freq_train <- slam::col_sums(DocumentTermMatrix(training_group_text))
# use top freqent words
nword <- 500
top_word_list <- names(sort(word_freq_train, decreasing = TRUE)[1:nword])
# subset of dataframe with most frequent words
x_train_matrix <- as.matrix(DocumentTermMatrix(training_group_text)[, top_word_list])
x_train_df <- as.data.frame(x_train_matrix)
# scaling those most frequent words columns
preprocess_para <- preProcess(x_train_df, method=c('center', 'scale')) # creat an scaling object 
x_train_scaled <- predict(preprocess_para, x_train_df)   # use the object to scale x_train_df
# add age, gender, topic, sign features
x_train_scaled$age <- training_csv$age
x_train_scaled$gender <- factor(training_csv$gender)
x_train_scaled$topic <- factor(training_csv$topic)
x_train_scaled$sign <- factor(training_csv$sign)
# linear regression
linear_mod <- lm(age ~ ., data=x_train_scaled)

#-------------------------------
# validation
# create a corpus for validate set
group_text <- validate_csv[,'text']
group_text <- Corpus(VectorSource(group_text))
# clean the corpus
group_text <- tm_map(group_text, content_transformer(tolower))      # convert to lower case
group_text <- tm_map(group_text, removeNumbers)                     # remove numbers
group_text <- tm_map(group_text, removeWords, stopwords('english')) # remove stopwords
group_text <- tm_map(group_text, removePunctuation)                 # remove punctuation
group_text <- tm_map(group_text, stripWhitespace)                   # remove extra white space
group_text <- tm_map(group_text, stemDocument)                      # change words to original form
# save the cleaned corpus as validation_group_text
validation_group_text <- group_text
# get subset of validation with top_word_list as features
doc_matrix <- DocumentTermMatrix(validation_group_text) 
x_validate <- doc_matrix[, top_word_list]
x_validate_df <- as.data.frame(as.matrix(x_validate))
# scaling those top_word-list features
x_validate_scaled <- predict(preprocess_para, x_validate_df)
# add gender, topic, sign features
x_validate_scaled$gender <- factor(validate_csv$gender)
x_validate_scaled$topic <- factor(validate_csv$topic)
x_validate_scaled$sign <- factor(validate_csv$sign)
# predicted age of the validation set
validate_age <- predict(linear_mod, x_validate_scaled)
# calculate mse
mse <- mean((validate_csv$age-validate_age)^2)
rms <- sqrt(mse)
rms
# [1] 7.344044 original
# [1] 87.93848 sparsity 70%
# [1] 141.0126 sparsity 90%
# [1] 7.371173 top 50
# [1] 7.318076 top 100 
# [1] 7.24863 top 200 
# [1] 7.121909 top 500 
# [1] 7.559354 top 5 
# [1] 6.724401 top 50 after adding gender, topic, sign
# [1] 6.456624 top 500 after adding gender, topic, sign using top_word_list

# ------------------------------
# test
# create a corpus from test set
test_csv <- read.csv('test.csv', stringsAsFactors=FALSE)
group_text <- test_csv[,'text']
group_text <- Corpus(VectorSource(group_text))
# clean the corpus 
group_text <- tm_map(group_text, content_transformer(tolower))      # convert to lower case
group_text <- tm_map(group_text, removeNumbers)                     # remove numbers
group_text <- tm_map(group_text, removeWords, stopwords('english')) # remove stopwords
group_text <- tm_map(group_text, removePunctuation)                 # remove punctuation
group_text <- tm_map(group_text, stripWhitespace)                   # remove extra white space
group_text <- tm_map(group_text, stemDocument)                      # change words to original form
# save the cleaned corpus as text_group_text
test_group_text <- group_text
# get the subset of text_group_text with the top_word_list as features
doc_matrix <- DocumentTermMatrix(test_group_text) 
x_test <- doc_matrix[, top_word_list]
x_test_df <- as.data.frame(as.matrix(x_test))
# scaling the features from the top_word_list
x_test_scaled <- predict(preprocess_para, x_test_df)
# add gender, topic, sign
x_test_scaled$gender <- factor(test_csv$gender)
x_test_scaled$topic <- factor(test_csv$topic)
x_test_scaled$sign <- factor(test_csv$sign)
# predicted age for test set
test_age <- predict(linear_mod, x_test_scaled)
# write to csv file
test_df <- data.frame(age=test_age, user.id=test_csv$user.id)
user_df <- test_df %>% group_by(user.id) %>% summarise_at(vars(age), funs(mean(age))) # use median to get age for the same user id
write.csv(user_df[,c('user.id','age')], row.names=FALSE, file = 'predicted.csv')