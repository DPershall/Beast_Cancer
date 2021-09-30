# SH work quietly please

options(warning = F, message = F)

# Libraries

if(!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(gam))
  install.packages("gam", repos = "http://cran.us.r-project.org")

if(!require(randomForest))
  install.packages("randomForest", repos = "http://cran.us.r-project.org")

if(!require(matrixStats))
  install.packages("matrixStats", repos = "http://cran.us.r-project.org")

if(!require(rattle))
  install.packages("rattle", repos = "http://cran.us.r-project.org")

if(!require(ggrepel))
  install.packages("ggrepel", repos = "http://cran.us.r-project.org")

if(!require(doParallel))
  install.packages("doParallel", repos = "http://cran.us.r-project.org")


library(matrixStats)
library(tidyverse)
library(caret)
library(gam)
library(rattle)
library(ggrepel)

# Data pull

temp_dat <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"),
                     header = FALSE,
                     col.names = c("ID", "diagnosis", "radius_mean", 
                                   "texture_mean", "perimeter_mean", 
                                   "area_mean", "smoothness_mean", 
                                   "compactness_mean", "concavity_mean", 
                                   "concave_pts_mean", "symmetry_mean", 
                                   "fractal_dim_mean", "radius_se", 
                                   "texture_se", "perimeter_se", "area_se", 
                                   "smoothness_se", "compactness_se", 
                                   "concavity_se", "concave_pts_se", 
                                   "symmetry_se", "fractal_dim_se", 
                                   "radius_worst", "texture_worst", 
                                   "perimeter_worst", "area_worst", 
                                   "smoothness_worst", "compactness_worst", 
                                   "concavity_worst", "concave_pts_worst", 
                                   "symmetry_worst", "fractal_dim_worst"))

# make the diagnosis a factor B or M

temp_dat <- temp_dat %>% mutate(diagnosis = as.factor(diagnosis))

# take a quick look at the data 

glimpse(temp_dat)

# Quick exploratory plot of radius, (mean and worst values & diagnosis colored)

temp_dat %>% 
  group_by(diagnosis) %>% 
  ggplot(aes(radius_mean, radius_worst, color = diagnosis)) +
  geom_point(alpha = .5)

# quick exploratory plot of texture (mean and worst values & diagnosis colored)

temp_dat %>% group_by(diagnosis) %>% 
  ggplot(aes(texture_mean, texture_worst, fill = diagnosis)) +
  geom_boxplot()

# make a list object with a matrix called x that holds all the feature 
# measurements and a variable y which holds the diagnosis as a character factor

dat <- list(x = as.matrix(temp_dat[3:32]), y=as.factor(temp_dat$diagnosis))

# Now I will check the class of the two variables and the dimensions of the
# x feature matrix to ensure all the features I wanted to keep have been
# maintained

class(dat$x)

class(dat$y)

# dimensions proof that our samples and observations have been maintained

dim(dat$x)[1]

dim(dat$x)[2]

# what proportion of the samples have been classified as malignant?

mean(dat$y == "M")

# what proportion of samples have been classified as benign?

mean(dat$y == "B")


# I wonder what feature has the highest mean?

which.max(colMeans(dat$x))


# What feature has the lowest standard deviation? 

which.min(colSds(dat$x))

# column 20 corresponds with the fractal_dim_se variable


# Center and scale the features 

x_centered <- sweep(dat$x, 2, colMeans(dat$x))
x_scaled <- sweep(x_centered, 2, colSds(dat$x), FUN = "/")

sd(x_scaled[,1])
median(x_scaled[,1])

# average distance between all samples

d_samples <- dist(x_scaled)

# average distance of first sample (which is malignant) to other malignant
# samples 

dist_MtoM <- as.matrix(d_samples)[1, dat$y == "M"]
mean(dist_MtoM[2:length(dist_MtoM)])

# average distance of first sample (which is malignant) to benign samples 

dist_MtoB <- as.matrix(d_samples)[1, dat$y == "B"]
mean(dist_MtoB)

# Makes a heatmap of the relationship between features using the scaled matrix

d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features))

# performs hierarchical clustering on the features and cuts them into 5 groups

h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)


# makes a scatter plot with one feature from group one and one feature from 
# group 2 with the tumor type displayed by color.

temp_dat %>% ggplot(aes(radius_mean, texture_worst, color = diagnosis)) +
  geom_point(alpha = 0.5)


# Principal component analysis

pca <- prcomp(x_scaled)
pca_sum <- summary(pca)

# Show the summary of the principal component analysis

pca_sum$importance 

# Plot the proportion of variance explained and the principal component index

var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained, 
     ylab = "Proportion of Variation Explained", 
     xlab = "PC Index")

# We see that 90% of the variance is explained by the first 7 principal 
# principal components alone.

# Makes a quick scatter plot of the first two principal components with color 
# representing the tumor type.

data.frame(pca$x[,1:2], type = dat$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()

# Here we see that benign tumors tend to have higher values of PC1; we also
# observe that benign and malignant tumors have a similar spread of values for 
# PC2. We also achieve much better delineation between classes than the previous 
# analysis achieved. However, there are still some outliers present.o

# Let's make box-plots of the first 7 principal components grouped by 
# tumor type, because as we already discovered, the first 7 PC's explain 90% of 
# the variance.

data.frame(type = dat$y, pca$x[,1:7]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()

# Now that we have better idea of the predictive challenges ahead, let's make a 
# new data frame that will hold the x_scaled features and the diagnosis (which 
# will be stored as y).

x_scaled <- as.data.frame(x_scaled)
db <- x_scaled %>% cbind(y = dat$y)

# double check 

glimpse(db)


# Time to split the data. The goal is to be able to predict as accurately as 
# possible whether a new sample taken from a fine needle aspirate of a tumor is 
# benign or malignant. Therefore, I want to split the data twice so that we can 
# train and test algorithms on a subset before I run the final validation on a 
# hold out set. Each split will be 20/80 (20% for testing and 80% for training). 

# Make the Validation set.

set.seed(1654, sample.kind = "Rounding")
test_index <- createDataPartition(y = db$y, times = 1, p = 0.2, list = FALSE)
Val <- db[test_index,]
training <- db[-test_index,]

# take the training and split again into smaller test and train sets

set.seed(4123, sample.kind = "Rounding")
test_index2 <- createDataPartition(y = training$y, times = 1, p = 0.2, list = FALSE)
test <- training[test_index2, ]
train <- training[-test_index2,]

# remove the indexes and other unnecessary objects

rm(test_index, test_index2, db, h, temp_dat, dat, x_scaled, x_centered, pca_sum,
   pca, d_features, d_samples, dist_MtoM, dist_MtoB, groups, var_explained)

# Let's check the train and test sets for the prevalence of benign tumors.

mean(train$y=="B")
mean(test$y=="B")


# Perfect, the prevalence is relatively the same among our train and test sets.

# Models
# I am now ready to move into generative models. I will compose several
# classification models, and then use an ensemble approach for the predictions on 
# the test set. This method should improve the accuracy and minimize the 
# confidence interval.

## Logistic Regression

train_glm <- train(y~.,data = train, method = "glm")
glm_preds <- predict(train_glm, test)
glm_acc <- mean(glm_preds == test$y)
Accuracy_Results <- tibble(Method = "GLM", Accuracy = glm_acc)
Accuracy_Results %>% knitr::kable()


## Linear Discriminatory Analysis

train_lda <- train(y~., data = train, method = "lda", preProcess = "center")
lda_preds <- predict(train_lda, test)
lda_acc <- mean(lda_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "LDA", Accuracy = lda_acc))
Accuracy_Results %>% knitr::kable()


# Let's examine the LDA model by making a plot of the predictors and their 
# importance in the model for classifying malignant tumors.

t(train_lda$finalModel$means) %>% 
  data.frame() %>% 
  mutate(predictor_name = rownames(.)) %>% 
  ggplot(aes(predictor_name, M, label=predictor_name)) +
  geom_point()+
  coord_flip()


# Interesting, concave_pts_worst was the most important variable for 
# classifying malignant tumors in the LDA model.

## Quadratic Discriminant Analysis

train_qda <- train(y~., train, method = "qda", preProcess = "center")
qda_preds <- predict(train_qda, test)
qda_acc <- mean(qda_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "QDA", Accuracy = qda_acc))
Accuracy_Results %>% knitr::kable()


# Let's again make a plot of the predictors and their importance to the QDA model
# in classifying malignant tumors.

t(train_qda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(predictor_name, M, label=predictor_name)) +
  geom_point()+
  coord_flip()

# So, we see the level of feature importance for the QDA model is the same as 
# the LDA model.

## gamLOESS

set.seed(5, sample.kind = "Rounding")
train_loess <- train(y~., data = train, method = "gamLoess")
loess_preds <- predict(train_loess, test)
loess_acc <- mean(loess_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "gamLoess", Accuracy = loess_acc))
Accuracy_Results %>% knitr::kable()


## K nearest neighbors

set.seed(7, sample.kind="Rounding")
tuning <- data.frame(k = seq(1, 10, 1))
train_knn <- train(y~.,
                   data = train,
                   method = "knn", 
                   tuneGrid = tuning)

# show the best tune
plot(train_knn)

knn_preds <- predict(train_knn, test)
knn_acc <- mean(knn_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "KNN", Accuracy = knn_acc))
Accuracy_Results %>% knitr::kable()


## Random Forest

set.seed(9, sample.kind="Rounding")
tuning <- data.frame(mtry = c(1,2,3))
train_rf <- train(y~., 
                  data = train,
                  method = "rf",
                  tuneGrid = tuning,
                  trControl = trainControl(method = "cv",
                                           number = 10),
                  importance = TRUE)


# We can examine the tuning with a plot.

plot(train_rf)


# Now that we know we have a reasonably tuned random forest model, let's examine
# the importance of the features to the classification process.

plot(varImp(train_rf))


# OK, we find that many of the most important features are the "worst" features. 
# We also find that the importance of features for the Random Forest model is
# different than what we found with the LDA and QDA models. 

# Time to run the predictions and report the accuracy

rf_preds <- predict(train_rf, test)
rf_acc <- mean(rf_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "RF", Accuracy = rf_acc))
Accuracy_Results %>% knitr::kable()

## Neural Network with repeated Cross Validation

set.seed(2976, sample.kind = "Rounding")

# I will use cross validation in the train set in order to find the optimal 
# hidden layers and decay.

tc <- trainControl(method = "repeatedcv", number = 10, repeats=3)
train_nn <- train(y~., 
                  data=train, 
                  method='nnet', 
                  linout=FALSE, 
                  trace = FALSE, 
                  trControl = tc,
                  tuneGrid=expand.grid(.size= seq(5,10,1),.decay=seq(0.1,0.15,0.01))) 


# Plot the different models to see the effect on accuracy

plot(train_nn)


# Show the best Tune

train_nn$bestTune


# calculate the accuracy

nn_preds <- predict(train_nn, test)
nn_acc <- mean(nn_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "NNet", Accuracy = nn_acc))
Accuracy_Results %>% knitr::kable()


## Recursive Partitioning - Rpart

set.seed(10, sample.kind = "Rounding")
train_rpart <- train(y ~ ., 
                     data = train,
                     method = "rpart",
                     trControl = trainControl(method = "cv", number = 10),
                     tuneGrid = data.frame(cp = seq(.022, .023, .0001)))

# Show the best tune
train_rpart$bestTune

# plot the resulting decision tree from the rpart model.
fancyRpartPlot(train_rpart$finalModel, yesno = 2)

# Calc the accuracy
rpart_preds <- predict(train_rpart, test)
rpart_acc <- mean(rpart_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "RPart", Accuracy = rpart_acc))
Accuracy_Results %>% knitr::kable()


# All of the models we have run so far have an accuracy over 90%. We know from 
# examining the feature importance in the models that they are weighting the 
# features differently for their use in classifying the tumor samples. Therefore, 
# we should expect an increase in sensitivity by creating an ensemble.

# First, let's examine the most accurate model so far with a confusion matrix.

confusionMatrix(as.factor(qda_preds), as.factor(test$y))


# We will compare these results to a confusion matrix of an ensemble approach.

## Ensemble Method
# I will create a new data frame that holds all of the benign predictions from 
# the previous generative models.

ensemble <- cbind(glm = glm_preds == "B", lda = lda_preds == "B", 
                  qda = qda_preds == "B", loess = loess_preds == "B", 
                  rf = rf_preds == "B", nn = nn_preds == "B", 
                  rp = rpart_preds == "B", knn = knn_preds == "B")

# I will say that if more than half of the algorithms predicted benign then
# the ensemble will predict benign. This also means that if less than half 
# predict benign then the ensemble will predict malignant.

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "B", "M")


# Let's check the accuracy of the ensemble method against the test set and 
# present the findings by attaching it to our running table. 

ensemble_acc <- mean(ensemble_preds == test$y)
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "Ensemble", 
                                     Accuracy = ensemble_acc))
Accuracy_Results %>% knitr::kable()


# Now, let's examine a confusion matrix to see if we have better results than
# the QDA model had on it's own.

confusionMatrix(as.factor(ensemble_preds), as.factor(test$y))


# OK! The ensemble method has improved the sensitivity. We also see that the 
# negative prediction value is 1. This means that every time (100% of the time) 
# the ensemble method predicted malignancy, the tumor was indeed malignant. 

# Let's repeat the above steps with the data from the first split, and run the
# final validation on the hold out set.

###################### VALIDATION SET ##########################################
# Validation

set.seed(1, sample.kind = "Rounding")
train_glm <- train(y~.,data = training, method = "glm")

set.seed(2, sample.kind = "Rounding")
train_lda <- train(y~., data = training, method = "lda")

set.seed(3, sample.kind = "Rounding")
train_qda <- train(y~., training, method = "qda")

set.seed(4, sample.kind = "Rounding")
train_loess <- train(y~., data = training, method = "gamLoess")

set.seed(5, sample.kind="Rounding")
tuning <- data.frame(k = seq(3, 21, 2))
train_knn <- train(y~.,
                   data = training,
                   method = "knn", 
                   tuneGrid = tuning)

set.seed(6, sample.kind="Rounding")
tuning <- data.frame(mtry = c(1,2,3))
train_rf <- train(y~., 
                  data = training,
                  method = "rf",
                  tuneGrid = tuning,
                  trControl = trainControl(method = "cv",
                                           number = 10),
                  importance = TRUE)


set.seed(7, sample.kind = "Rounding")
# Use cross validation in the train set in order to find the optimal 
# hidden layers and decay.

tc <- trainControl(method = "repeatedcv", number = 10, repeats=3)
train_nn <- train(y~., 
                  data=training, 
                  method='nnet', 
                  linout=FALSE, 
                  trace = FALSE, 
                  trControl = tc,
                  tuneGrid=expand.grid(.size= seq(5,10,1),
                                       .decay=seq(0.15,0.2,0.01))) 

set.seed(8, sample.kind = "Rounding")
train_rpart <- train(y ~ ., 
                     data = training,
                     method = "rpart",
                     trControl = trainControl(method = "cv", number = 10),
                     tuneGrid = data.frame(cp = seq(.022, .023, .0001)))

# Run Predictions for the models
val_preds_glm <- predict(train_glm, Val)
val_preds_lda <- predict(train_lda, Val)
val_preds_qda <- predict(train_qda, Val)
val_preds_loess <- predict(train_loess, Val)
val_preds_rf <- predict(train_rf, Val)
val_preds_nn <- predict(train_nn, Val)
val_preds_rp <- predict(train_rpart, Val)
val_preds_knn <- predict(train_knn, Val)

# Ensemble the models
Val_Ensemble <- cbind(glm = val_preds_glm == "B", lda = val_preds_lda == "B", 
                      qda = val_preds_qda == "B", loess = val_preds_loess == "B",
                      rf = val_preds_rf == "B", nn = val_preds_nn == "B",
                      rp = val_preds_rp == "B", knn = val_preds_knn == "B")

Val_Ensemble_preds <- ifelse(rowMeans(Val_Ensemble)>0.5, "B", "M")

# Validation Accuracy

Val_Acc <- mean(Val_Ensemble_preds == Val$y)

Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "Ensemble Validation", 
                                     Accuracy = Val_Acc))
Accuracy_Results %>% knitr::kable()

# Confusion Matrix
confusionMatrix(as.factor(Val_Ensemble_preds), as.factor(Val$y))