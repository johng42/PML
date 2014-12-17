#Project for Practical Machine Learning, John Hopkins Data Science track on Coursera
#packages needed  caret, rpart, e1071
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(e1071))
suppressPackageStartupMessages(library(randomForest))
set.seed(75961)#Nacogdoches, TX!!!
#you need the CSV files from the training data for this project which are available here: 
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#The test data are available here: 
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
#The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 
trainData <- read.csv(file.choose(), na.strings = c("NA", ""))
testData <- read.csv(file.choose(), na.strings = c("NA", ""))

#get some basic info about the data
head(trainData)
summary(trainData)
str(trainData)

#make a data set for training and for cross validation
#since we are given a test set there is no need to create it here
#just use a 70/30 split, wiht 30% for cross validation
inTrain <- createDataPartition(y=trainData$classe, p=0.7, list=FALSE)
cleanedTrainData <- trainData[inTrain,]
cleanedCVData <- trainData[-inTrain,]

#remove timestamps, username and non-numeric columns (data) - that first column X is just an index
indicesToRemove <- grep("user_name|*_timestamp|X",names(trainData))
cleanedTrainData <- trainData[,-indicesToRemove]
cleanedTestData <- testData[,-indicesToRemove]
cleanedCVData <- cleanedCVData[-indicesToRemove]

#per lecture, get rid of columns that have near zero variance

#first, make some temp copies of classe for training and CV
tempTrainClasse <- cleanedTrainData$classe
tempCVClasse <- cleanedCVData$classe
#this may take a few seconds to run
indicesToRemove2 <- nearZeroVar(cleanedTrainData, freqCut=1, uniqueCut = 20)
cleanedTrainData <- cleanedTrainData[-indicesToRemove2] 
cleanedTestData <- cleanedTestData[-indicesToRemove2]
cleanedCVData <- cleanedCVData[-indicesToRemove2]
#sigh.  this remove the classe column so add it back to training and validation sets
cleanedTrainData$classe <- tempTrainClasse
cleanedCVData$classe <- tempCVClasse
cleanedTestData["classe"] <- "A" #just random value for this column


#and I could not get nearZeroVar to remove these other 10 bogus columns
#this was the real challenge, getting to a small, clean data set
indicesToRemove3 <- grep("avg_yaw_dumbbell|avg_pitch_dumbbell|avg_roll_dumbbell|min_roll_dumbbell|max_roll_dumbbell|min_yaw_arm|min_roll_belt|max_roll_belt|skewness_pitch_dumbbell|kurtosis_picth_dumbbell",names(cleanedTrainData))
cleanedTrainData <- cleanedTrainData[-indicesToRemove3] 
cleanedTestData <- cleanedTestData[-indicesToRemove3]
cleanedCVData <- cleanedCVData[-indicesToRemove3]
#cleanedTrainData now has ~7 variables (features) and classe
plot(cleanedTrainData)

plot(cleanedTrainData$pitch_dumbbell, cleanedTrainData$classe)
plot(cleanedTrainData$num_window, cleanedTrainData$classe)

#we are given a test set separately from training so the train set is now ready to fit to a model
#and use classe as data we are trying to fit
#first try rpart
modFit<- train(cleanedTrainData$classe ~ ., data=cleanedTrainData, method = "rpart")

#print the accuracy of this model, about 43%
acc<-modFit$results
print(acc$Accuracy)
#with accuracy this low, find a better model


#this takes about 30 minutes or so to run on my low end celeron, even with allowParallel set to TRUE
trainRF <- trainControl(method='cv', number=4, allowParallel=TRUE)
modRF <- train(cleanedTrainData$classe~. , data=cleanedTrainData, method='rf', prof=TRUE, trControl=trainRF)

print(modRF$results$Accuracy)
#Boom.  99% accurate but possibly overfitted since the entire data set is used to train

#try a glm model? - no, glm can only do 2 cases
#modglm = train(classe ~ .,data=trainData,method="glm")

#since the random forest has 99% accuracy, use it to assess the cross validation set
pred<-predict(modRF, cleanedCVData)
cleanedCVData$classe <- pred==cleanedCVData$classe
print(sum(cleanedCVData$classe==TRUE)/length(cleanedCVData$classe))
#the cross validation has 100% accuracy 

#since the random forest has 99% accuracy, use it to predict the test set answers
pred<-predict(modRF, cleanedTestData)
cleanedTestData$classe <- pred==cleanedTestData$classe
#cleanedTestData$classe has 100% accuracy for purposes of this class
#but compare to random guessing
guess=rep(c("E","D","C","B","A"),4)
table(guess,pred)
#that was only 30% accurate (expect 20% with random guessing)



#to create output files for grading, not needed for any other purpose
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)
