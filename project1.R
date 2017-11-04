setwd("C:/Users/Patrick/Documents/R/MachineLearning/Project")
getwd()
library(caret)
library(dplyr)
train <- read.csv("./data/pml-training.csv")
test <- read.csv("./data/pml-testing.csv")
train
dim(train)
dim(test)
View(train[1:200,])
View(test)
View(train)
train$classe
names(train)
names(test)
nrow(test)
View(train[train$num_window==2,])
View(train[train$user_name=='carlitos' & train$classe=='A',])

by_window <- train %>% group_by(num_window)
by_window %>% summarise(
  count(user_name),
)

test[test$num_window==48,c("user_name", "cvtd_timestamp")]
dim(train[train$num_window==43,c("user_name", "cvtd_timestamp")])
# study NAs on test sample
sapply(test, function(y) sum(length(which(is.na(y)))))

sum(sapply(test, function(y) all(is.na(y))))
sum(sapply(test, function(y) any(is.na(y))))

sapply(test, function(y) sum(length(which(is.na(y)))==nrow(test)))
sapply(test, anyNA)
names(test)[sapply(test, anyNA)]
names(test)[sapply(test, allNA)]
dim(test)

names(train) != names(test)
names(train[,1:159]) != names(test[,1:159])
t <- rbind(train[,1:159],test[,1:159])
dim(t)
# keep onky columns without NAs and blanks
# study NAs
sapply(t, function(y) sum(length(which(is.na(y)))))
summary(train$num_window)
View(train[train$num_window==1,])
View(train[train$num_window==2,'classe'])
na_count <-sapply(t, function(y) sum(length(which(is.na(y) | y ==''))))
na_count
train2 <- train[,na_count == 0]
test2 <- test[,na_count == 0]

str(train2)
str(test2)
names(train2) == names(test2)
# check all values are numeric now
nnum_count <-sapply(train2, function(y) sum(length(which(!is.numeric(y)))))
nnum_count
# what are we solving for?
summary(train2$classe)
train2$kurtosis_roll_arm
test$kurtosis_roll_arm
fit1 <- glm(classe ~ X, data=train2[1:20,], na.action=na.omit)
summary(train2$classe)  
str(train2$classe)  
train2$X[1:20]

set.seed(32343)
modelFit <- train(classe ~.,data=train2, method="gbm")
modelFit

#############################
## lets start from scratch
#############################

names(train)
## there are raw measures and statistics  on those measures
View(train[train$num_window==2,])
mean(train[train$num_window==2,'roll_belt'])
train[train$num_window==2 & train$new_window =='yes',]$avg_roll_belt

mean(train[train$num_window==2 ,'roll_arm'])
train[train$num_window==2 & train$new_window =='yes',]$avg_roll_arm
# the statistics are not available one single observations
nas <- sapply(test, function(y) sum(length(which(is.na(y)))))
nanames <- names(test)[nas == nrow(test)]
# so we are leaving them out
#tain <- train[%in% nanames]

train2 <- train %>% 
  select(-starts_with("kurtosis"),
         -starts_with("skewness"), 
         -starts_with("max"), 
         -starts_with("min"), 
         -starts_with("amplitude"), 
         -starts_with("avg"), 
         -starts_with("var"), 
         -starts_with("stddev"),
         -contains("timestamp"),
         -X,-new_window,-num_window
         )

View(train2)

test2 <- test %>% 
  select(-starts_with("kurtosis"),
         -starts_with("skewness"), 
         -starts_with("max"), 
         -starts_with("min"), 
         -starts_with("amplitude"), 
         -starts_with("avg"), 
         -starts_with("var"), 
         -starts_with("stddev"),
         -contains("timestamp"),
         -X,-new_window,-num_window,-problem_id
  )

View(test2)

data.frame(names(train2[,1:53]), names(test2))

# check for nas
anyNA(train2)
anyNA(test2)

summary(train2)
summary(test2)

# create a training set and a testing set

inTrain <- createDataPartition(y=train2$classe,
                               p=0.75, list=FALSE)
training <- train2[inTrain,]
testing <- train2[-inTrain,]
dim(training)
names(training)
# let's warm up on binary classification

training$isclasseA <- as.factor(training$classe == 'A')
testing$isclasseA <- as.factor(testing$classe == 'A')
summary(training$isclasseA)
View(training)

set.seed(32343)
modelFit <- train(isclasseA ~.-classe ,data=training, method="glm")
modelFit
modelFit$finalModel

predictions <- predict(modelFit,newdata=testing)
predictions
confusionMatrix(predictions,testing$isclasseA)
# 91% accuracy does not sound great with all the data we have

# multiclass
# recursive partitioning
inTrain <- createDataPartition(y=train2$classe,
                               p=0.75, list=FALSE)
training <- train2[inTrain,]
testing <- train2[-inTrain,]
set.seed(32343)
modelFit <- train(classe ~. ,data=training, method="rpart")
modelFit
modelFit$finalModel

predictions <- predict(modelFit,newdata=testing)
#predictions
confusionMatrix(predictions,testing$classe)

# random forest
inTrain <- createDataPartition(y=train2$classe,
                               p=0.75, list=FALSE)
training <- train2[inTrain,]
testing <- train2[-inTrain,]
set.seed(32343)
modelFit <- train(classe ~. ,data=training, method="rf")
# takes too long...
modelFit
modelFit$finalModel

predictions <- predict(modelFit,newdata=testing)
#predictions
confusionMatrix(predictions,testing$classe)

# gbm
inTrain <- createDataPartition(y=train2$classe,
                               p=0.75, list=FALSE)
training <- train2[inTrain,]
testing <- train2[-inTrain,]
set.seed(32343)
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)
modelFit <- train(classe ~. ,data=training, method="gbm",
                  trControl=fitControl)
modelFit$finalModel

predictions <- predict(modelFit,newdata=testing)
#predictions
confusionMatrix(predictions,testing$classe)

# let's try using gbm directly
library("gbm")
modelFit <- gbm(classe ~., data=training, distribution="multinomial",
                train.fraction=0.8)
summary(modelFit)
predictions <- predict(modelFit,newdata=testing[,1:53], n.trees=100, type="response")
predictions
dim(training)
library("randomForest")
# call with formula
set.seed(32343)
modelFitRF <- randomForest(classe ~., data=training)
# call withe default method
modelFitRFd <- randomForest(training[,1:53], y = training$classe)
modelFitRF
modelFitRFd
# out of bag error estimate is 0.35%
# we don't need cross-validation for random forest
# and classification error is  below 0.80%
# (below 0.05% for class A)
# check accuracy on testing subset
predictionsRF <- predict(modelFitRF,newdata=testing[,1:53])
confusionMatrix(predictionsRF,testing$classe)
# accuracy is 99.29%, a bit lower than what we expected
# but consistent with the previous result
predictionsRF
dim(training)

# looks good enough, let's compute predicted on actual testing set
# (or should we rerun on full set first?)
predictions <- predict(modelFitRF, newdata=test2)
predictions
View(predictions)
class(predictions)

# save my perfect prediction
save(predictions, file="./output/predictions.RData", ascii=TRUE)
load(file="./output/predictions.RData", verbose = TRUE)

# refit on whole sample, check if any different
modelFit <- randomForest(classe ~., data=train2)
predictions2 <- predict(modelFit, newdata=test2)
# same predictions
predictions2==predictions
# plot the error vs number of trees
# we probably did not need to go further than 100 trees
plot(modelFit)
round(importance(modelFit),2)
test2
modelFit