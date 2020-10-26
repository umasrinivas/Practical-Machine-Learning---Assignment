Introduction
------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways.

Loading Packages
----------------

``` r
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
```

Data Preparation
----------------

``` r
# set the url for the download
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlValid <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
training <- read.csv(url(urlTrain))
validation <- read.csv(url(urlValid))
training$classe <- as.factor(training$classe)

# create a partition with the training dataset
inTrain <- createDataPartition(y=training$classe,p=0.7,list=FALSE)
trainSet <- training[inTrain,]
testSet <- training[-inTrain,]
dim(trainSet)
```

    ## [1] 13737   160

``` r
dim(testSet)
```

    ## [1] 5885  160

``` r
# remove variables with near zero variance
NZV <- nearZeroVar(trainSet)
trainSet <- trainSet[,-NZV]
testSet <- testSet[,-NZV]

# remove variables that are mostly NA
allNA <- sapply(trainSet,function(x)mean(is.na(x)))>0.95
trainSet <- trainSet[,allNA==FALSE]
testSet <- testSet[,allNA==FALSE]

# remove identification only variables
trainSet <- trainSet[,-(1:5)]
testSet <- testSet[,-(1:5)]
dim(trainSet)
```

    ## [1] 13737    54

``` r
dim(testSet)
```

    ## [1] 5885   54

Correlation Analysis
--------------------

``` r
#library(corrplot)
corMatrix <- cor(trainSet[,-54])
corrplot(corMatrix,order="FPC",method="color",type="lower",tl.cex=0.8,tl.col=rgb(0,0,0))
```

![](final_assignment_files/figure-markdown_github/unnamed-chunk-3-1.png)

Prediction Model building
-------------------------

### Method - Random Forest

``` r
# model fit
set.seed(12345)
controlRF <- trainControl(method="cv",number=3,verboseIter=FALSE)
modFitRandForest <- train(classe~.,data=trainSet,method="rf",trControl=controlRF)
modFitRandForest$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.22%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3905    1    0    0    0 0.0002560164
    ## B    5 2649    3    1    0 0.0033860045
    ## C    0    5 2391    0    0 0.0020868114
    ## D    0    0    7 2245    0 0.0031083481
    ## E    0    0    1    7 2517 0.0031683168

``` r
# prediction on test set
predRandForest <- predict(modFitRandForest,testSet)
confMatRandForest <- confusionMatrix(predRandForest,testSet$classe)

# plot matrix results
plot(confMatRandForest$table,col=confMatRandForest$byClass,
     main=paste("Random Forest - Accuracy =",round(confMatRandForest$overall['Accuracy'],4)))
```

![](final_assignment_files/figure-markdown_github/unnamed-chunk-4-1.png)

### Method - Decision Trees

``` r
# model fit
set.seed(12345)
modFitDecTree <- rpart(classe~.,data=trainSet,method="class")
fancyRpartPlot(modFitDecTree)
```

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](final_assignment_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
# prediction on test data set
predDecTree <- predict(modFitDecTree,newdata=testSet,type="class")
confMatDecTree <- confusionMatrix(predDecTree,testSet$classe)
confMatDecTree
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1571  297   41  102   70
    ##          B   32  632   69   43   92
    ##          C   23  111  825   81   45
    ##          D   40   51   71  624   80
    ##          E    8   48   20  114  795
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7556          
    ##                  95% CI : (0.7445, 0.7666)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6883          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9385   0.5549   0.8041   0.6473   0.7348
    ## Specificity            0.8789   0.9503   0.9465   0.9508   0.9604
    ## Pos Pred Value         0.7549   0.7281   0.7604   0.7206   0.8071
    ## Neg Pred Value         0.9729   0.8989   0.9581   0.9323   0.9414
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2669   0.1074   0.1402   0.1060   0.1351
    ## Detection Prevalence   0.3536   0.1475   0.1844   0.1472   0.1674
    ## Balanced Accuracy      0.9087   0.7526   0.8753   0.7991   0.8476

``` r
# plot matrix results
plot(confMatDecTree$table,col=confMatDecTree$byClass,
     main = paste("Decision Tree - Accuracy =",round(confMatDecTree$overall['Accuracy'], 4)))
```

![](final_assignment_files/figure-markdown_github/unnamed-chunk-5-2.png)

### Method - Generalized Boosted Model

``` r
# model fit
set.seed(12345)
controlGBM <- trainControl(method="repeatedcv",number=5,repeats=1)
modFitGBM <- train(classe~.,data=trainSet,method="gbm",trControl=controlGBM,verbose=F)
modFitGBM$finalModel
```

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 53 had non-zero influence.

``` r
# prediction on test data set
predGBM <- predict(modFitGBM,newdata=testSet)
confMatGBM <- confusionMatrix(predGBM,testSet$classe)
confMatGBM
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673   10    0    0    0
    ##          B    1 1115    9    4    9
    ##          C    0   14 1013   15    2
    ##          D    0    0    3  945    9
    ##          E    0    0    1    0 1062
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9869          
    ##                  95% CI : (0.9837, 0.9897)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9834          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9789   0.9873   0.9803   0.9815
    ## Specificity            0.9976   0.9952   0.9936   0.9976   0.9998
    ## Pos Pred Value         0.9941   0.9798   0.9703   0.9875   0.9991
    ## Neg Pred Value         0.9998   0.9949   0.9973   0.9961   0.9959
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1895   0.1721   0.1606   0.1805
    ## Detection Prevalence   0.2860   0.1934   0.1774   0.1626   0.1806
    ## Balanced Accuracy      0.9985   0.9870   0.9905   0.9889   0.9907

``` r
# plot matrix results
plot(confMatGBM$table,col=confMatGBM$byClass,
     main=paste("GBM - Accuracy =",round(confMatGBM$overall['Accuracy'],4)))
```

![](final_assignment_files/figure-markdown_github/unnamed-chunk-6-1.png)

Applying best fit model
-----------------------

``` r
# Applying the Selected Model to the Validation Data
predValidation <- predict(modFitRandForest,validation)
predValidation
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
