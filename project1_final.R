
#############################
## load data
#############################

setwd("C:/Users/Patrick/Documents/R/MachineLearning/Project")
getwd()
library(caret)
library(dplyr)
library(magrittr)
library(Hmisc)

# read files to dataframes
train <- read.csv("./data/pml-training.csv")
test <- read.csv("./data/pml-testing.csv")

# the training test contains 19622 observations of 160 variables
dim(train)
dim(test)

# from http://groupware.les.inf.puc-rio.br/har 
# we know the dataset contains measures for 4 sensors:
# arm, forearm, belt and dumbbell

# first fields are identifier, user name (6 users), time stamps, window identifier and 
# indicator for new window
str(train[,1:7])

# the remaining fields are measures for belt, arm, forearm and dumbbell
names(train)[grep("belt", names(train))]
names(train)[grep("_arm", names(train))]
names(train)[grep("forearm", names(train))]
names(train)[grep("dumbbell", names(train))]
# 13 primary measures for the belt
# roll, pitch, yaw, total acceleration,
# 'gyros', 'accel' and 'magnet'  measures on 3 axes
beltm <- names(train)[grep("belt", names(train))]
i <- grep("(^[a-z]+_belt)|(^total_[a-z]+_belt)", beltm) 
beltm[i]
anyNA(train[,beltm[i]])
apply(train[,beltm[i]],2,function(x) sum(x == ''))
# there are 13 primary measures (roll, pitch, yaw, total_accel 
# and gyros, accel, magnet on 3 axes), with no NAs or blanks

# there are 25 statistics on those measures, with mostly NAs or errors when 
# the new window indicator is False
# the reasonable explanation is that for every new window 
# statistics are given for all measures in that window.
beltm[-i]
t <- train[train$new_window=='no',beltm[-i]]
str(t)

# example for window 2, avg_roll_belt contains average of all measures in that window
mean(train[train$num_window==2,'roll_belt'])
train[train$num_window==2 & train$new_window =='yes',]$avg_roll_belt

# we will leave the statistics out as they are mostly redundant with the raw measure
# we will revisit this choice if we find we don't have enough data after all

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

# the variable names are now consistent between train and test set
data.frame(names(train2[,1:53]), names(test2))

# no nas in our filtered dataset
anyNA(train2)
anyNA(test2)

# quick summary of our data does not show anything untoward
summary(train2)
summary(test2)

###############################
# model fit
###############################

# create a training set and a testing set
set.seed(1342)
inTrain <- createDataPartition(y=train2$classe,
                               p=0.75, list=FALSE)
training <- train2[inTrain,]
testing <- train2[-inTrain,]

# set training parameters: we will run 5-fold cross validation
fitControl <- trainControl(method = "cv", number=5)

# multiclass
#############
# knn
#############

# let's run a k-nearest neighbours models with 1, 2 and 5 neighbours
# with 5-fold cross validation
tuneGrid <-  expand.grid(k = c(1,2,5))
set.seed(32343)
modelFit <- train(classe ~. ,data=training, method="knn",
                  trControl = fitControl, tuneGrid=tuneGrid, preProcess=c("center", "scale"))

# best result is achieved for k=1, with an accuracy of 98.66%
modelFit

#############
# naive Bayes
#############
library("naivebayes")
library("fastICA")
fitControl <- trainControl(method = "cv", number=5)
tuneGrid <-  expand.grid(fL = c(0,0.5,1),
                         usekernel = c(TRUE,FALSE),
                         adjust=c(TRUE,FALSE))

set.seed(32343)
modelFit <- train(classe ~. ,data=training, method="naive_bayes",
                  trControl = fitControl,  tuneGrid=tuneGrid)
modelFit

# the best result is an accuracy of 73.6%


##################
# random forest
##################

library("randomForest")
rfGrid <-  expand.grid(mtry = 7)
set.seed(32343)
model <- train(classe~., data=training, trControl=fitControl, method="rf",
               tuneGrid=rfGrid, ntree=100)
model
model$finalModel

# this is our best model, let's check the accuracy on our test set
predictions <- predict.train(model,newdata=testing[,1:53])
confusionMatrix(predictions,testing$classe)

# test results are consistent with cross validatino results,
# let's compute predicted on actual testing set
submission <- predict.train(model,newdata=test2)
submission

# save submission
#save(submission, file="./output/predictions.RData", ascii=TRUE)
