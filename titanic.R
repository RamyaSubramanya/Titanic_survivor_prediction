setwd(dir = "D:\\DATA ANALYTICS\\CLASSIFICATION & REGRESSION\\Classification Practicals\\titanic")

#loading train and test data

train<-read.csv("train.csv")
test<-read.csv("test.csv")

#combining train and test data

install.packages("dplyr")
library(dplyr)

data<-bind_rows(train, test)

--------------------------------------------------------------------------------
#exploratory data analysis

str(data)
data.frame(colnames(data))
table(data$Cabin)

#dropping the variables name, ticket

data<-data[,-c(4,9)]

#renaming the variables

names(data)<-c("PassengerId", "Target_class", "Travel_class", "Gender", "Age",
               "Sibling&Spouse", "Parents&Children","Travel_fare","Cabin_no",
               "Embarked")

#to check if there are any missing values

colSums(is.na(data))

hist(data$Age)   #non-normal distributed
hist(data$Travel_fare)   #left skewed data 

#since values in both the variables are not normally distributed we will replace 
#the data using median imputation


data$Age[which(is.na(data$Age))]<-median(data$Age, na.rm = TRUE)

data$Travel_fare[which(is.na(data$Travel_fare))]<-median(data$Travel_fare, 
                                                         na.rm = TRUE)


--------------------------------------------------------------------------------
#convert categorical data into dummies
  
str(data)

table(data$Gender)
  
table(data$Cabin_no)

table(data$Embarked)

install.packages("fastDummies")
library(fastDummies)  
  
data<-dummy_cols(data, select_columns = "Gender", remove_first_dummy = TRUE, 
                 remove_selected_columns = TRUE)
  
data<-dummy_cols(data, select_columns = "Embarked", remove_first_dummy = TRUE, 
                 remove_selected_columns = TRUE) 
  

data$Cabin_no<-as.factor(data$Cabin_no)

data$Target_class<-as.factor(data$Target_class)

data<-data[,-c(8)]

---------------------------------------------------------------------------------

#split the data into train 

Traindata<-data[1:891,]
Testdata<-data[892:1309,]
  
--------------------------------------------------------------------------------

#Logistic regression

install.packages("caret")
library(caret)

reg1<-glm(Target_class~., data = Traindata, family = binomial)
summary(reg1)

p1<-predict(reg1, newdata = Traindata, type = "response")
p1<-ifelse(p1>0.50, "1", "0")
p1<-factor(p1)
confusionMatrix(p1, Traindata$Target_class, positive = "1")

#Sensitivity is 70%, accuracy is 80%, p-value<0.05,  kappa=0.5

--------------------------------------------------------------------------------

#to balance the data
  
install.packages("ROSE")
library(ROSE)

install.packages("janitor")
library(janitor)

Traindata<-clean_names(Traindata)
Testdata<-clean_names(Testdata)

table(Traindata$Target_class)
549*2

set.seed(123)
over<-ovun.sample(target_class~., data = Traindata, method = "over", N = 1098)$data
  
#logistic regression on over sampled data

reg2<-glm(target_class~., data = over, family = binomial)
summary(reg2)  
  
p2<-predict(reg2, Traindata, type = "response")
p2<-ifelse(p2>0.50, "1", "0")
p2<-factor(p2) 
confusionMatrix(p2, Traindata$target_class, positive = "1")
  

#After balancing the data, sensitivity increased to 76%, accuracy is 79%, kappa = 0.56, p-value<0.05

--------------------------------------------------------------------------------

#Decision Tree

install.packages("rpart")
library(rpart)
  
library(rpart.plot)
library(rattle)

?rpart
tree<-rpart(target_class~., data = over, method = "class")
tree

fancyRpartPlot(tree)

p3<-predict(tree, newdata = over, type = "class")
confusionMatrix(p3, over$target_class, positive = "1")

#With decision tree, sensitivity is 86%, accuracy is 84%, kappa=0.69, p-value<0.05


-----------------------------------------------------------------------------------
#Random Forest
  
install.packages("randomForest")  
library(randomForest)  

?randomForest

set.seed(123)
forest<-randomForest(target_class~., data = over)

plot(forest)

?tuneRF
tuneRF(over[,-2], over[,2], ntreeTry = 170, stepFactor = 2, 
       improve = 0.05, trace = TRUE, plot = TRUE)

#with ntree = 400, mtry = 6, we shall rebuild the randomforest

forest1<-randomForest(target_class~., data = over, mtry = 6, ntree = 170)

p4<-predict(forest1, newdata = over, type = "class")
confusionMatrix(p4, over$target_class, positive = "1")

varImpPlot(forest1)

#we get random forest poor sensitivity

--------------------------------------------------------------------------------
  
#decision tree 10-fold cross validation
?train
custom<-trainControl(method = "repeatedcv", number = 10, repeats = 5)
tree1<-train(target_class~., data = over, method = "rpart", 
             trControl = custom, tuneLength = 10)
  
tree1$results
#accuracy is high as 82.18%, complexity parameter is 0.010

plot(tree1)   #visualization of cp vs accuracy

p5<-predict(tree1, newdata = over)
confusionMatrix(p5, over$target_class, positive = "1")
  
#sensitivity is 84.88%, accuracy is 84%, kappa=0.68, p-value<0.05

--------------------------------------------------------------------------------
  
#predict the decision tree model on test data
  
p6<-predict(tree, newdata = Testdata, type = "class")
write.csv(p6, "prediction.csv")
