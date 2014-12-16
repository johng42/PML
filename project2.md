---
title: "Project"
author: "John"
date: "Monday, December 15, 2014"
output: html_document
---

This is the overview of the project for Practical Machine Learning from John Hopkins on Coursera.
This work was created by JohnG

The purpose of this project is to analyze some fitness data and predict what type of users would result from given fitness patterns.  The overall methodology is to take a set of data that classifies existing users, remove extraneous data from it, build 2 models based on the data and then use the better of the 2 models to predict user types.  The test data used to predict users is given as a separate file so the entire original data set can be used for training.  Validation will be done on the coursera website for the 20 test cases as a final submission.

Project Walkthrough:

First, load the needed packages and read the data from the CSV files given.

```{r}
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(e1071))
suppressPackageStartupMessages(library(randomForest))
#you need the CSV files from The training data for this project are available here: 
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#The test data are available here: 
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
#The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 
trainData <- read.csv(file.choose(), na.strings = c("NA", ""))
testData <- read.csv(file.choose(), na.strings = c("NA", ""))
```

Before any processing starts, just examine the data:

```{r}
head(trainData)
summary(trainData)
str(trainData)
```

There is a huge amount of bogus data.  Trim it down in three stages.  The first trim is to remove usernames, timestamps and X (which is just an index).  

```{r}
indicesToRemove <- grep("user_name|*_timestamp|X",names(trainData))
cleanedTrainData <- trainData[,-indicesToRemove]
cleanedTestData <- testData[,-indicesToRemove]
```

Then use the nearZeroVariance method from caret to remove the bulk of the columns with no useful data.

```{r}
#this may take a few seconds to run
indicesToRemove2 <- nearZeroVar(cleanedTrainData, freqCut=1, uniqueCut = 20)
cleanedTrainData <- cleanedTrainData[-indicesToRemove2] 
cleanedTestData <- cleanedTestData[-indicesToRemove2]
#sigh.  this remove the classe column so add it back
cleanedTrainData$classe <- trainData$classe
cleanedTestData["classe"] <- "A" #just random value for this column
```

This still doesn't get rid of enough.  I tried altering the freqCut and uniqueCut parameters for newZeroVar but still could not get these columns, so remove them manually.

```{r}
indicesToRemove3 <- grep("avg_yaw_dumbbell|avg_pitch_dumbbell|avg_roll_dumbbell|min_roll_dumbbell|max_roll_dumbbell|min_yaw_arm|min_roll_belt|max_roll_belt|skewness_pitch_dumbbell|kurtosis_picth_dumbbell",names(cleanedTrainData))
cleanedTrainData <- cleanedTrainData[-indicesToRemove3] 
cleanedTestData <- cleanedTestData[-indicesToRemove3]
```

Now that the data is clean and whittled down to only 6 features ( + classe which is what we are trying to predict), take a quick look at all data.  Nothing fancy, just trying to get a feel for the data.  If this does not show a graph, see "6features.png" in the repro.

```{r, echo=FALSE}
plot(cleanedTrainData)
```

num_window and pitch_dumbbell look a little interesting.  If the graphs don't display in the browser they are in the repo:

```{r, echo=FALSE}
plot(cleanedTrainData$pitch_dumbbell, cleanedTrainData$classe)
plot(cleanedTrainData$num_window, cleanedTrainData$classe)
```
So first try an rpart model to see what kind of fit it can get.

```{r}
modFit<- train(cleanedTrainData$classe ~ ., data=cleanedTrainData, method = "rpart")

acc<-modFit$results
print(acc$Accuracy[1])
print(sum(acc$Accuracy^2)) #should be 3 for a great fit
```

The accuracy here is at best 45% which is better than a random 20% expectation but not otherwise very good.  Try a random forest next.  This takes about hald an hour on a low end Celeron to run even with allowParallel = TRUE. 

```{r}
trainRF <- trainControl(method='cv', number=4, allowParallel=TRUE)
modRF <- train(cleanedTrainData$classe~. , data=cleanedTrainData, method='rf', prof=TRUE, trControl=trainRF)

print(modRF$results$Accuracy[3])
print(sum(modRF$results$Accuracy^2)) #very near 3
```

99% accuracy.  This is very good, but I am a little worried that the model is overfitted due to using the entire data source as the training set.

```{r}
pred<-predict(modRF, cleanedTestData)
cleanedTestData$classe <- pred==cleanedTestData$classe
#cleanedTestData$classe has 100% accuracy for purposes of this class as submitted to the grader.
#but compare to random guessing
guess=rep(c("E","D","C","B","A"),4)
table(guess,pred)
```

The random guesses (E,D,C,B,A repeated in that order 4 times) had a 30% accuracy compared to the 100% accurary of the random forest.

That's it.  Enjoy!
