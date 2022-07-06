# LIBRARIES  ####
library(tidyverse)
library(lubridate)
library(caret)

#### RMSE CALCULATION ####
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#### PRELIMINARY  ####
train_set <- edx
test_set <- validation

#### timestamp to year, then extraction of movie date, and finally time difference between rating date and movie date  ####

neg_ratinPeriog <- train_set %>%   # this is just for looking at the negative ratingPeriod to include in the report
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%   # transforming timestamp to year
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%  # extracting the year of the movie from the title
  mutate(ratingPeriod = ratingDate - movieDate) %>% # calculating the time difference between the rating date and the movie date
  select(timestamp, title, ratingDate, movieDate, ratingPeriod) %>%
  group_by(ratingPeriod) %>%
  arrange(ratingPeriod) %>%
  summarize(n = n()) %>%
  filter(ratingPeriod <= 0)

train_set <- train_set %>% # modifications for training set
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%   # transforming timestamp to year
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%  # extracting the year of the movie from the title
  mutate(ratingPeriod = ratingDate - movieDate) %>% # calculating the time difference between the rating date and the movie date
  mutate(ratingPeriod = ifelse(ratingPeriod < 0, 0, ratingPeriod)) # if time difference is negative, which does not make sense, change to zero

test_set <- test_set %>% # modifications for test set
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%  # transforming timestamp to year
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%   # extracting the year of the movie from the title
  mutate(ratingPeriod = ratingDate - movieDate) %>% # calculating the time difference between the rating date and the movie date
  mutate(ratingPeriod = ifelse(ratingPeriod < 0, 0, ratingPeriod))  # if time difference is negative, which does not make sense, change to zero


#### naive model: calculating the Mean mu  ####

mu <- mean(train_set$rating) # building the naive model
train_set <- train_set %>%
  mutate(mu = mu)

predicted_ratings <- mu  # calculating the prediction on the test set
naive_rmse <- RMSE(test_set$rating, predicted_ratings) # calculating the loss function with the test set
rmse_results <- data.frame(method = "Just the Average Model", RMSE = naive_rmse, target_RMSE = " < 0.86490")  # making a summary table

#### 1st model: Movie effect b_i  ####

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu), n_i = n()) # Movie effect calculation

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 50, data = ., color = I("black"))  +
  scale_x_continuous(limits = c(-2, 2)) # movie effect observation

predicted_ratings <- test_set %>% # calculation of the prediction of the test set
  left_join(movie_avgs, by = 'movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

model1_rmse <- RMSE(test_set$rating, predicted_ratings) # calculation of the RMSE

rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "+ Movie Effect Model", RMSE = model1_rmse, target_RMSE = " < 0.86490")) # adding result to final table

#### 2nd model: User effect b_u  ####

user_avgs <- train_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i), n_u = n()) # adding the user effect calculation

user_avgs %>% qplot(b_u, geom ="histogram", bins = 50, data = ., color = I("black"))  +
  scale_x_continuous(limits = c(-2, 2)) # user effect observation

predicted_ratings <- test_set %>%    # calculation of the prediction of the test set
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model2_rmse <- RMSE(test_set$rating, predicted_ratings)  # calculation of the RMSE

rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "+ User Effect Model", RMSE = model2_rmse, target_RMSE = " < 0.86490")) # adding result to final table

#### 3rd model: Time effect b_t  ####

time_avgs <- train_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  group_by(ratingPeriod) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u), n_t = n()) # adding the time effect calculation

time_avgs %>% qplot(b_t, geom ="histogram", bins = 100, data = ., color = I("black"))  +
  scale_x_continuous(limits = c(-2, 2)) # time effect observation

predicted_ratings <- test_set %>%      # calculation of the prediction of the test set
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  left_join(time_avgs, by = 'ratingPeriod') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  pull(pred)

model3_rmse <- RMSE(test_set$rating, predicted_ratings)  # calculation of the RMSE

rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "+ Time Effect Model", RMSE = model3_rmse, target_RMSE = " < 0.86490")) # adding result to final table

#### 4th model: Genres effect b_g  ####

genre_avgs <- train_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  left_join(time_avgs, by = 'ratingPeriod') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_t), n_g = n()) # adding the genres effect calculation

genre_avgs %>% qplot(b_g, geom ="histogram", bins = 100, data = ., color = I("black"))  +
  scale_x_continuous(limits = c(-2, 2)) # genres effect observation

predicted_ratings <- test_set %>%  # calculation of the prediction of the test set
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  left_join(time_avgs, by = 'ratingPeriod') %>%
  left_join(genre_avgs, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
  pull(pred)

model4_rmse <- RMSE(test_set$rating, predicted_ratings)  # calculation of the RMSE

rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "+ Genre Effect Model", RMSE = model4_rmse, target_RMSE = " < 0.86490")) # adding result to final table

#### Tuning regularization #####

set.seed(1, sample.kind = "Rounding")  # Partitioning the train_set in half to create a new train and test set for tuning lambda with regularization
test_index <- createDataPartition(
  train_set$rating,
  times = 1,
  p = 0.5,
  list = FALSE)

train_sample <- train_set %>% slice(-test_index[,1])
temp <- train_set %>% slice(test_index[,1])
# Make sure userId and movieId in test_sample are also in train_sample
test_sample <- temp %>% 
  semi_join(train_sample, by = "movieId") %>%
  semi_join(train_sample, by = "userId") %>%
  semi_join(train_sample, by = "ratingPeriod") %>%
  semi_join(train_sample, by = "genres")
# Add rows removed from validation set back into train set
removed <- anti_join(temp, test_sample)
train_sample <- rbind(train_sample, removed)
rm(removed, temp)
lambdas <- seq(0, 10, 0.25)   # tuning parameter lambda

#calculation of the rmses of the new test set with the tuning regularization parameter lambda
## TAKES A LITTLE LONG (FEW MINUTES)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_sample$rating)
  
  movie_avgs <- train_sample %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  user_avgs <- train_sample %>%
    left_join(movie_avgs, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  time_avgs <- train_sample %>%
    left_join(movie_avgs, by = 'movieId') %>%
    left_join(user_avgs, by = 'userId') %>%
    group_by(ratingPeriod) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+l))
  
  genre_avgs <- train_sample %>%
    left_join(movie_avgs, by = 'movieId') %>%
    left_join(user_avgs, by = 'userId') %>%
    left_join(time_avgs, by = 'ratingPeriod') %>% 
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_t)/(n()+l))
  
  predicted_ratings <- test_sample %>%
    left_join(movie_avgs, by = 'movieId') %>%
    left_join(user_avgs, by = 'userId') %>%
    left_join(time_avgs, by = 'ratingPeriod') %>%
    left_join(genre_avgs, by = 'genres') %>%
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_sample$rating))
})

qplot(lambdas, rmses)  # drawing of all 'rmses' results from lambda
l <- lambdas[which.min(rmses)] # choosing the best tune for lambda

#### results after regularization  ####

# modifying the data sets with all models and regularization tuning parameter
movie_avgs_reg <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

user_avgs_reg <- train_set %>%
  left_join(movie_avgs_reg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+l))

time_avgs_reg <- train_set %>%
  left_join(movie_avgs_reg, by = 'movieId') %>%
  left_join(user_avgs_reg, by = 'userId') %>%
  group_by(ratingPeriod) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+l))

genre_avgs_reg <- train_set %>%
  left_join(movie_avgs_reg, by = 'movieId') %>%
  left_join(user_avgs_reg, by = 'userId') %>%
  left_join(time_avgs_reg, by = 'ratingPeriod') %>% 
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u - b_t)/(n()+l))

predicted_ratings <- test_set %>%
  left_join(movie_avgs_reg, by = 'movieId') %>%
  left_join(user_avgs_reg, by = 'userId') %>%
  left_join(time_avgs_reg, by = 'ratingPeriod') %>%
  left_join(genre_avgs_reg, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
  pull(pred)

model5_rmse <- RMSE(test_set$rating, predicted_ratings) # calculating RMSE
rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "+ Regularization Effect Model", RMSE = model5_rmse, target_RMSE = " < 0.86490")) # adding result to final table

rmse_results %>%
  knitr::kable() # visualizing the results

#### REMOVING UNNEEDED FILES TO SAVE SPACE WHEN SAVING ENVIRONMENT  ####

rm(predicted_ratings, train_sample, test_sample, validation, edx)

#### SAVING IMAGE FOR RMARKDOWN  ####

save.image("~/Capstone_Movielens.RData")
