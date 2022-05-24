MovieLens Project
================
Courtellemont Remi
2022-05-24

## EXECUTIVE SUMMARY

The goal is to create a movie recommendation system using the MovieLens
Dataset <https://grouplens.org/datasets/movielens/10m/>.  
the train and validation sets are provided within the course in
www.edx.org.

We develop our algorithm using the edx set.  
For a final test of our final algorithm, we predict movie ratings in the
validation set (the final hold-out test set) as if they were unknown.  
RMSE will be used to evaluate how close our predictions are to the true
values in the validation set (the final hold-out test set).

``` r
library(tidyverse)
library(lubridate)
library(caret)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
```

The first step will be to clean and transform if necessary the data.  
The second step will be to build the model with the training set.  
The third step will be to apply regularization.  
The fourth step will be to apply the model to the validation set and
check the RMSE result.

## METHOD AND ANALYSIS

### INITIATION

The initial script comes from EDX course, and it creates edx and
validation sets.

In order to avoid modifying the original data, the files are renamed:

``` r
train_set <- edx
test_set <- validation
```

### DATA EXPLORATION

#### General

The first 5 lines of train_set :

``` r
head(train_set, 5) %>% knitr::kable(align = 'c')
```

| userId | movieId | rating | timestamp |             title             |              genres              |
|:------:|:-------:|:------:|:---------:|:-----------------------------:|:--------------------------------:|
|   1    |   122   |   5    | 838985046 |       Boomerang (1992)        |         Comedy\|Romance          |
|   1    |   185   |   5    | 838983525 |        Net, The (1995)        |     Action\|Crime\|Thriller      |
|   1    |   292   |   5    | 838983421 |        Outbreak (1995)        | Action\|Drama\|Sci-Fi\|Thriller  |
|   1    |   316   |   5    | 838983392 |        Stargate (1994)        |    Action\|Adventure\|Sci-Fi     |
|   1    |   329   |   5    | 838983392 | Star Trek: Generations (1994) | Action\|Adventure\|Drama\|Sci-Fi |

The *output* we want to predict/estimate is the **rating** variable.

The train_set has the following dimensions:

``` r
dim(train_set) %>% knitr::kable(align = 'c')
```

|    x    |
|:-------:|
| 9000055 |
|    6    |

The *variable* names :

``` r
colnames(train_set) %>% knitr::kable(align = 'c')
```

|     x     |
|:---------:|
|  userId   |
|  movieId  |
|  rating   |
| timestamp |
|   title   |
|  genres   |

We identify 3 main *predictors*: **MovieId**, **userId** and
**genres**.  
It is easy to understand that they are impacting the *rating*.

The number of unique *predictors* in the data set :

``` r
table_data <- data.frame(variable = c("movies", "users", "genres"),
                         unique_numbers = c(n_distinct(train_set$movieId),
                                    n_distinct(train_set$userId),
                                    n_distinct(train_set$genres)))
table_data %>% knitr::kable(align = 'c')
```

| variable | unique_numbers |
|:--------:|:--------------:|
|  movies  |     10677      |
|  users   |     69878      |
|  genres  |      797       |

The 5 most rated movies in the data set :

``` r
train_set %>% group_by(movieId, title) %>%
    summarize(count = n()) %>%
    arrange(desc(count)) %>%
  head(5) %>%
  knitr::kable(align = 'c')
```

| movieId |              title               | count |
|:-------:|:--------------------------------:|:-----:|
|   296   |       Pulp Fiction (1994)        | 31362 |
|   356   |       Forrest Gump (1994)        | 31079 |
|   593   | Silence of the Lambs, The (1991) | 30382 |
|   480   |       Jurassic Park (1993)       | 29360 |
|   318   | Shawshank Redemption, The (1994) | 28015 |

…and the 5 least rated movies in the data set :

``` r
train_set %>% group_by(movieId, title) %>%
    summarize(count = n()) %>%
    arrange(desc(count)) %>%
  tail(5) %>%
  knitr::kable(align = 'c')
```

| movieId |         title          | count |
|:-------:|:----------------------:|:-----:|
|  64976  |      Hexed (1993)      |   1   |
|  65006  |     Impulse (2008)     |   1   |
|  65011  | Zona Zamfirova (2002)  |   1   |
|  65025  | Double Dynamite (1951) |   1   |
|  65027  | Death Kiss, The (1933) |   1   |

#### Time predictor

The data base include a **timestamp**, which corresponds to the date of
rating.  
It can be converted into rating date’s year, with a new column called
**ratingDate** :

``` r
train_set %>%
  select(timestamp, title) %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  head(5) %>%
  knitr::kable(align = 'c')
```

| timestamp |             title             | ratingDate |
|:---------:|:-----------------------------:|:----------:|
| 838985046 |       Boomerang (1992)        |    1996    |
| 838983525 |        Net, The (1995)        |    1996    |
| 838983421 |        Outbreak (1995)        |    1996    |
| 838983392 |        Stargate (1994)        |    1996    |
| 838983392 | Star Trek: Generations (1994) |    1996    |

In the title of the movie, we can see the date of release of the
movie.  
We can extract this year, and create a new column called **movieDate**.

``` r
train_set %>%
  select(timestamp, title) %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%
  head(5) %>%
  knitr::kable(align = 'c')
```

| timestamp |             title             | ratingDate | movieDate |
|:---------:|:-----------------------------:|:----------:|:---------:|
| 838985046 |       Boomerang (1992)        |    1996    |   1992    |
| 838983525 |        Net, The (1995)        |    1996    |   1995    |
| 838983421 |        Outbreak (1995)        |    1996    |   1995    |
| 838983392 |        Stargate (1994)        |    1996    |   1994    |
| 838983392 | Star Trek: Generations (1994) |    1996    |   1994    |

It can be interesting to have a *predictor* based on time difference
between the year of the rating and the year of the movie.  
It is easy to understand that if a movie has been released long time
ago, it might has less impact on the user, compare to a movie just
released.  
We call this new column **ratingPeriod**.

``` r
train_set %>%
  select(timestamp, title) %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%
  mutate(ratingPeriod = ratingDate - movieDate) %>%
  head(5) %>%
  knitr::kable(align = 'c')
```

| timestamp |             title             | ratingDate | movieDate | ratingPeriod |
|:---------:|:-----------------------------:|:----------:|:---------:|:------------:|
| 838985046 |       Boomerang (1992)        |    1996    |   1992    |      4       |
| 838983525 |        Net, The (1995)        |    1996    |   1995    |      1       |
| 838983421 |        Outbreak (1995)        |    1996    |   1995    |      1       |
| 838983392 |        Stargate (1994)        |    1996    |   1994    |      2       |
| 838983392 | Star Trek: Generations (1994) |    1996    |   1994    |      2       |

An issue is that the **ratingPeriod** is sometimes negative, which in
practice is not possible.

``` r
train_set %>%
  select(timestamp, title) %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%
  mutate(ratingPeriod = ratingDate - movieDate) %>%
  group_by(ratingPeriod) %>%
  arrange(ratingPeriod) %>%
  summarize(n = n()) %>%
  filter(ratingPeriod <= 0) %>%
  knitr::kable(align = 'c')
```

| ratingPeriod |   n    |
|:------------:|:------:|
|      -2      |   3    |
|      -1      |  172   |
|      0       | 380915 |

In order to avoid this situation, we modify the **ratingPeriod** to 0
when negative.

``` r
train_set %>%
  select(timestamp, title) %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%
  mutate(ratingPeriod = ratingDate - movieDate) %>%
  mutate(ratingPeriod = ifelse(ratingPeriod < 0, 0, ratingPeriod)) %>%
  group_by(ratingPeriod) %>%
  arrange(ratingPeriod) %>%
  summarize(n = n()) %>%
  knitr::kable(align = 'c')
```

| ratingPeriod |    n    |
|:------------:|:-------:|
|      0       | 381090  |
|      1       | 1068070 |
|      2       | 853680  |
|      3       | 647650  |
|      4       | 473660  |
|      5       | 436378  |
|      6       | 410159  |
|      7       | 354368  |
|      8       | 320713  |
|      9       | 292203  |
|      10      | 281351  |
|      11      | 264241  |
|      12      | 244377  |
|      13      | 238584  |
|      14      | 222441  |
|      15      | 194625  |
|      16      | 170496  |
|      17      | 154513  |
|      18      | 144901  |
|      19      | 133861  |
|      20      | 122445  |
|      21      | 109990  |
|      22      |  94676  |
|      23      |  82010  |
|      24      |  74585  |
|      25      |  73021  |
|      26      |  69501  |
|      27      |  59227  |
|      28      |  53212  |
|      29      |  49049  |
|      30      |  44398  |
|      31      |  44287  |
|      32      |  46000  |
|      33      |  39517  |
|      34      |  33714  |
|      35      |  33530  |
|      36      |  33839  |
|      37      |  34296  |
|      38      |  30624  |
|      39      |  29587  |
|      40      |  30588  |
|      41      |  30372  |
|      42      |  30251  |
|      43      |  27747  |
|      44      |  25520  |
|      45      |  26971  |
|      46      |  25890  |
|      47      |  22031  |
|      48      |  20230  |
|      49      |  19090  |
|      50      |  17164  |
|      51      |  15676  |
|      52      |  14286  |
|      53      |  13986  |
|      54      |  13302  |
|      55      |  12266  |
|      56      |  13501  |
|      57      |  13496  |
|      58      |  15795  |
|      59      |  18694  |
|      60      |  16729  |
|      61      |  14360  |
|      62      |  11911  |
|      63      |  11795  |
|      64      |  11698  |
|      65      |  12204  |
|      66      |  12288  |
|      67      |  10100  |
|      68      |  8185   |
|      69      |  7132   |
|      70      |  4232   |
|      71      |  3491   |
|      72      |  3617   |
|      73      |  3224   |
|      74      |  3098   |
|      75      |  2901   |
|      76      |  1948   |
|      77      |  1685   |
|      78      |  1408   |
|      79      |  1242   |
|      80      |  1132   |
|      81      |   925   |
|      82      |   709   |
|      83      |   632   |
|      84      |   421   |
|      85      |   412   |
|      86      |   290   |
|      87      |   143   |
|      88      |   167   |
|      89      |   108   |
|      90      |   60    |
|      91      |   41    |
|      92      |   28    |
|      93      |   14    |

The train_set and test_set are entirely modified to add the
**ratingPeriod** column.

``` r
train_set <- train_set %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>%
  mutate(ratingPeriod = ratingDate - movieDate) %>%
  mutate(ratingPeriod = ifelse(ratingPeriod < 0, 0, ratingPeriod))

test_set <- test_set %>%
  mutate(ratingDate = as.integer(year(as_datetime(timestamp)))) %>%
  mutate(movieDate = as.integer(str_sub(title, -5, -2))) %>% 
  mutate(ratingPeriod = ratingDate - movieDate) %>%
  mutate(ratingPeriod = ifelse(ratingPeriod < 0, 0, ratingPeriod))
```

### MODELING

Due to the number of movies and users, training the data set is not
feasible.  
Then we will build the model, according to the following formula:  

![Y\_{u,i,t,g} = \\mu + b_i + b_u + b_t + b_g + \\varepsilon\_{u,i,t,g}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_%7Bu%2Ci%2Ct%2Cg%7D%20%3D%20%5Cmu%20%2B%20b_i%20%2B%20b_u%20%2B%20b_t%20%2B%20b_g%20%2B%20%5Cvarepsilon_%7Bu%2Ci%2Ct%2Cg%7D "Y_{u,i,t,g} = \mu + b_i + b_u + b_t + b_g + \varepsilon_{u,i,t,g}")

Where:  
![Y\_{u,i,t,g}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y_%7Bu%2Ci%2Ct%2Cg%7D "Y_{u,i,t,g}")
represents the estimated value.
![\\mu](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu "\mu")
represents the mean  
![b_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b_i "b_i")
represents the movie effect  
![b_u](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b_u "b_u")
represents the user effect  
![b_t](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b_t "b_t")
represents the time effect  
![b_g](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b_g "b_g")
represents the gender effect

And then we will apply regularization by creating a partition on the
train_set.

## RESULTS

### BUILDING THE MODEL WITH THE TRAINING SET

#### THE NAIVE MODEL

We build a naive model where all movies are rated the average
![\\mu](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cmu "\mu").  
We calculate the average of the train_set and then the RMSE of the
test_set (validation set).

``` r
mu <- mean(train_set$rating) # building the naive model
train_set <- train_set %>%
  mutate(mu = mu)
predicted_ratings <- mu  # calculating the prediction on the test set
naive_rmse <- RMSE(test_set$rating, predicted_ratings) # calculating the loss function with the test set
rmse_results <- data.frame(method = "Just the Average Model", RMSE = naive_rmse, target_RMSE = " < 0.86490")  # making a summary table
rmse_results %>%
  knitr::kable(align = 'c')
```

|         method         |   RMSE   | target_RMSE |
|:----------------------:|:--------:|:-----------:|
| Just the Average Model | 1.061202 | \< 0.86490  |

We see that the RMSE is far from the target.
