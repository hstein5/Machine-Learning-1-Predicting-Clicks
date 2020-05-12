# Ricky Chen, Yifei Ren, Halle Steinberg, Robert Wei
# Team 4

######################## Data Cleaning ########################
library(data.table)
Train <- fread("ProjectTrainingData.csv")
colnames(Train) <- c("id","click","hour","C1","banner_pos","site_id","side_domain","site_category","app_id","app_domain","app_category",
                     "device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21")
Train[, id := NULL] # remove id
Train[, device_id := NULL] # remove device_ip
Train[, device_ip := NULL] # remove device_id
# break up hour into time category for hour of day and date for day
Train[, time := hour %% 100]
Train[, time_cate := as.integer(time/6)]
Train[, time := NULL]
Train[, date := (hour %% 10000 - hour %% 100)/100]
Train[, hour := NULL]

# since around 80-85% of the data for all variables is within at least the top 20 categories, we will keep the top 19 and group everything else into a 20th called "others"
# if there is less than 19 categories already, we will group into one less than the category and make the last "others"
for(i in 2:20){
  name <- colnames(Train)[2]
  newcategory <- paste(name,"category",sep = "_")
  Train[, count := .N, by=name]
  Train$values <- Train[,2]
  data <- as.data.table(unique(cbind(Train[,2], Train$count)))
  colnames(data) <- c("id", "id_count")
  data$id_count <- as.numeric(data$id_count)
  order_id_count <- sort(data$id_count, decreasing = TRUE)
  if(length(order_id_count) > 20){
    Train[, (newcategory) := ifelse(count > order_id_count[20], values, "others")]
    Train[, (name) := NULL]
  }
  else{
    Train[, (newcategory) := values]
    Train[, (name) := NULL]
  }
}

Train[, count := NULL]
Train[, values := NULL]

rows <- sample(nrow(Train))
randomTrain <- Train[rows,]
fwrite(randomTrain, "ProcessedTraining.csv")

randomTrain <- fread("ProcessedTraining.csv", header = TRUE)

# now randomly shuffle the training and val datasets
TrainingDataset <- randomTrain[1:3000000,]
fwrite(TrainingDataset, "newTrainingDataset.csv")
ValidationDataset <- randomTrain[3000001:4000000,]
fwrite(ValidationDataset, "newValidationDataset.csv")
ValidationDataset2 <- randomTrain[4000001:5000000,]
fwrite(ValidationDataset2, "newValidationDataset2.csv")

######################## Decision Tree Model ########################
library(tree)

# read in training data set
TrainData <- fread(file = 'newTrainingDataset.csv', header = TRUE)
TrainData <- as.data.frame(TrainData) # make train a data frame
TrainData[] <- lapply(TrainData, function(x) as.factor(x))

# read in validation data set
ValData <- fread(file = 'newValidationDataset.csv', header = TRUE)
ValData <- as.data.frame(ValData) # make val a data fame

# read in validation 2 data set
ValData2 <- fread(file = "newValidationDataset2.csv", header = TRUE)
ValData2 <- as.data.frame(ValData2) # make val 2 a data frame

# read in test data
TestData <- fread("ProcessedTestData.csv", header = TRUE)
TestData <- as.data.frame(TestData) # make test a data frame
TestData[] <- lapply(TestData, function(x) as.factor(x))

# split X and Y for training
XTrain <- TrainData[,-1]
YTrain <- TrainData$click

# split X and Y for validation
XVal <- ValData[,-1]
XVal[] <- lapply(XVal, function(x) as.factor(x))
YVal <- as.integer(ValData$click)
class(YVal)

# split X and Y for validation 2
XVal2 <- ValData2[,-1]
XVal2[] <- lapply(XVal2, function(x) as.factor(x))
YVal2 <- as.integer(ValData2$click)
class(YVal2)

# define the log loss function
LL <- function(Pred,YVal){
  ll <- -mean(YVal*log(Pred)+(1-YVal)*log(1-Pred))
  return(ll)
}

# train the tree model 
tc <- tree.control(nrow(TrainData),minsize=20000,mincut=5000)
tree <- tree(click ~ ., data=TrainData, control=tc, split = "gini")

# make predictions with validation data
predTree <- predict(tree, newdata=XVal)
predTree <- predTree[,2]

# calculate log loss for validation data
treell <- LL(predTree, YVal)

# make predictions with second validation data
predTreeVal <- predict(tree, newdata=XVal2)
predTreeVal <- predTreeVal[,2]

# calculate log loss for second validation data
treell2 <- LL(predTreeVal, YVal2)

# make predictions with test data
predTreeTest <- predict(tree, newdata=TestData)
predTreeTest <- predTreeTest[,2]

######################## Logistic Regression Model ########################
library(glmnet)

# split X and Y - need to use model.matrix for X variables
XTrain <- model.matrix(click~., TrainData)[,-1]
YTrain <- TrainData$click

XVal <- model.matrix(click~., ValData)[,-1]
YVal <- ValData$click

XVal2 <- model.matrix(click~., ValData2)[,-1]
YVal2 <- ValData2$click

labels <- as.vector(unique(TrainData$click))

class(YVal)

# find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(XTrain, YTrain, alpha = 1, family = "binomial") # use cv.glmnet to perform k-fold cross-validation and get lambda value

# fit the final model on the training data
LR <- glmnet(XTrain, YTrain, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min) # use alpha = 1 to perform lasso regularization and automatic feature selection 

# display regression coefficients
coef(LR)

# make predictions with the validation data set
predLR <- predict(LR, newx = XVal, type = "response")
predLR <- as.vector(predLR)
LRll <- LL(predLR,(as.integer(YVal)-1))

# make predictions with the second validation data set
predLRVal <- predict(LR, newx = XVal2, type = "response")
predLRVal<- as.vector(predLRVal)
LR112 <- LL(predLR,(as.integer(YVal2)-1))

# make predictions with test data
XTest <- TestData[,1:9] # use the variables as indicated from the cv.glmnet
XTest[] <- lapply(XTest, function(x) as.factor(X)) # make them factors

XTest$click <- c(rep(1,13000000),rep(0,15341))
XTest$click <- as.factor(XTest$click)
XTest <- as.data.frame(XTest)
XTest1 <- model.matrix(click~., XTest)[,-1] # make into model.matrix

# make predictions on test data
predLRTest <- predict(LR, newx = XTest1, type = "response")

######################## Neural Network Models ########################
# using all of the clean data, write out the data for the neural net in Python
write.table(TrainData,file="TrainDataForBCExample.csv",sep=",",row.names=F)
write.table(ValData,file="ValDataForBCExample.csv",sep=",",row.names=F)
write.table(TestData,file="TestDataForBCExample.csv",sep=",",row.names=F)

# read back in neural net output
tmp1 <- read.table("TrYHatFromBCNN.csv",header=T,sep=",")
tmp2 <- read.table("ValYHatFromBCNN.csv",header=T,sep=",")

# compare three nn models
llNN4 <- LL(YHatVal4, ValData$click) # log loss = 0.421
llNN10 <- LL(YHatVal10, ValData$click) # log loss = 0.456
llNNsm <- LL(YHatValSM, ValData$click) # log loss = 0.432

# it turns out Neural Net Model has optimal outcome with 4 layers of 4 neurons probability output

# calculate log loss on validation data 2
tmp3 <- read.table("ValYHatFromBCNN2.csv",header=T,sep=",")
llv2<--(1-ValData$`ValData$click`)*log(1-tmp3$YHatVal)-ValData$`ValData$click`*log(tmp3$YHatVal)
mean(llv2) # log loss = 0.4215

######################## Check Log Loss on Val 2 ########################
# read in predictions on validation data 2
dt<-read.table('DecisionTreePredictionsVal2.csv',header=T)
lg<-read.table('LRValidation2Predictions (1).csv',header=T)
nn<-read.table('ValYHatFromBCNN2.csv',header=T)
xgb<-read.table('ValYHatFromXGB2.csv',header=T)

# log loss function
LL <- function(Pred,YVal){
  ll <- -mean(YVal*log(Pred)+(1-YVal)*log(1-Pred))
  return(ll)
}

# average of all 4 models  
ave_4<-(dt+lg+nn+xgb)/4
LL(ave_4$predTree,YVal)
# log loss = 0.4086

# average of dt lg and nn 
ave_dln<-(dt+lg+nn)/3
LL(ave_dln$predTree,YVal)
# log loss = 0.4145

# average of dt lg and xgb
ave_dlx<-(dt+lg+xgb)/3
LL(ave_dlx$predTree,YVal)
# log loss = 0.4063

# average of dt nn and xgb
ave_dnx<-(dt+nn+xgb)/3
LL(ave_dnx$predTree,YVal)
# log loss = 0.4066

# average of lg nn and xgb
ave_lnx<-(lg+nn+xgb)/3
LL(ave_lnx$s0,YVal)
# log loss = 0.4100

# average of dt and xgb
ave_dx<-(dt+xgb)/2
LL(ave_dx$predTree,YVal)
# log loss = 0.4031

######################## Compile Predictions ########################
# need to read in Neural Net predictions since those were done in Python
predNNTest <- fread("TestYHatFromBCNN.csv", header = TRUE)
colnames(predNNTest) <- "predNNTest"

# need to read in XGBoost predictions since those were done in Python
predXGBTest <- fread("TestYHatFromXGB.csv", header = TRUE)
colnames(predXGBTest) <- "predXGBTest"

# put model predictions together into a data frame and we will use ensemble method to take the average of the 2 best models
TestDataPreds <- data.frame(predTreeTest, predLRTest, predNNTest, predXGBTest)

# take average of the 2 best models and add it to column P(click)
TestDataPreds$`P(click)` <- rowMeans(subset(TestDataPreds, select = c(predTreeTest, predXGBTest)))

# read in the submission file with correct data types
Data <- read.table("ProjectSubmission-TeamX.csv",colClasses=c("character","numeric"),header=T,sep=",")

# put the probabilities in Data[[2]]
Data[[2]] <- TestPredsFinal$`P(click)`

# round to 10 digits accuracy and prevent scientific notation
# this converts Data[[2]] to strings
Data[[2]] <- format(round(Data[[2]],10), scientific = FALSE)

# write out the data in the correct format
write.table(Data,file="ProjectSubmission-Team4.csv",quote=F,sep=",",
            row.names=F,col.names=c("id","P(click)"))
