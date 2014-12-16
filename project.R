#Project for Practical Machine Learning, John Hopkins Data Science track on Coursera
#packages needed  caret, rpart, e1071
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(e1071))
suppressPackageStartupMessages(library(randomForest))
set.seed(75961)#Nacogdoches, TX!!!
#you need the CSV files from The training data for this project are available here: 
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
#actual data omitted since it is huge

#remove timestamps, username and non-numeric columns (data) - that first column X is just an index
indicesToRemove <- grep("user_name|*_timestamp|X",names(trainData))
cleanedTrainData <- trainData[,-indicesToRemove]
cleanedTestData <- testData[,-indicesToRemove]

#per lecture, get rid of columns that have near zero variance
#this may take a few seconds to run
indicesToRemove2 <- nearZeroVar(cleanedTrainData, freqCut=1, uniqueCut = 20)
cleanedTrainData <- cleanedTrainData[-indicesToRemove2] 
cleanedTestData <- cleanedTestData[-indicesToRemove2]
#sigh.  this remove the classe column so add it back
cleanedTrainData$classe <- trainData$classe
cleanedTestData["classe"] <- "A" #just random value for this column

#and I could not get nearZeroVar to remove these other 10 bogus columns
#this was the real challenge, getting to a small, clean data set
indicesToRemove3 <- grep("avg_yaw_dumbbell|avg_pitch_dumbbell|avg_roll_dumbbell|min_roll_dumbbell|max_roll_dumbbell|min_yaw_arm|min_roll_belt|max_roll_belt|skewness_pitch_dumbbell|kurtosis_picth_dumbbell",names(cleanedTrainData))
cleanedTrainData <- cleanedTrainData[-indicesToRemove3] 
cleanedTestData <- cleanedTestData[-indicesToRemove3]

#cleanedTrainData now has ~7 variables (features) and classe
#take a quick look at plots
plot(cleanedTrainData)
#these 2 features look a little interesting (hard to tell from the small plots though)
plot(cleanedTrainData$pitch_dumbbell, cleanedTrainData$classe)
plot(cleanedTrainData$num_window, cleanedTrainData$classe)
#maybe interesting.  Still hard to tell so let's see if the classifiers can sort this out.

#we are given a test set separately from training so the train set is now ready to fit to a model
#and use classe as data we are trying to fit
#first try rpart
modFit<- train(cleanedTrainData$classe ~ ., data=cleanedTrainData, method = "rpart")

#print the accuracy of this model, about 45%
acc<-modFit$results
print(acc$Accuracy[1])
print(sum(acc$Accuracy^2)) #.39, not good

#this takes about 30 minutes or so to run on my crappy celeron
trainRF <- trainControl(method='cv', number=4, allowParallel=TRUE)
modRF <- train(cleanedTrainData$classe~. , data=cleanedTrainData, method='rf', prof=TRUE, trControl=trainRF)

print(modRF$results$Accuracy[3])
print(sum(modRF$results$Accuracy^2))
#Boom.  99% accurate but possibly overfitted since the entire data set is used to train

#try a glm model? - no, glm can only do 2 cases
#modglm = train(classe ~ .,data=trainData,method="glm")

#since the random forest has 99% accuracy, use it to predict the test set answers
pred<-predict(modRF, cleanedTestData)
cleanedTestData$classe <- pred==cleanedTestData$classe
#cleanedTestData$classe has 100% accuracy for purposes of this class
#but compare to random guessing
guess=rep(c("E","D","C","B","A"),4)
table(guess,pred)
#that was only 30% accurate (expect 20% with random guessing)



#to create output files for grading
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)

