require(DRR)
require(doParallel)
require(caret)
require(e1071)
require(randomForest)
require(MASS)
require(FNN)
require(RSNNS)
require(tidyr)
require(precprec)
require(OptimalCutpoints)
require(rpart)
require(ipred)
require(ELMR)
require(DescTools)
require(mlbench)
require(fastDummies)
require(neuralnet)
require(nnet)
require(NoiseFiltersR)
require(devtools)
detectCores()
cl <- makeCluster(max(1,detectCores()-1))
registerDoParallel(cl)
#install_github("davidavdav/ROC")
#library(ROC) # package for ROC, DET, AUC, EER

TIME <- Sys.time()
TIME
#rootFolder <- dirname(sys.frame(1)$ofile)
#setwd(rootFolder)

# converts all factors to numeric type, if levels>3 dummyVars could be considered
factorsNumeric <- function(d) modifyList(d, lapply(d[, sapply(d, is.factor)], as.numeric))

# Uncomment this line if you are creating final model
#OurData <- read.csv("loan.csv")

# Uncomment this line if you are testing script
OurData <- read.csv("../input/loan.csv",sep=",")
OurData
dim(OurData)
colnames(OurData)
str(OurData)
summary(OurData)

OurData$loan_amnt <-  as.numeric(levels(OurData$loan_amnt))[OurData$loan_amnt]

Desc(OurData$loan_amnt, main = "Loan amount distribution", plotit = TRUE)
Desc(OurData$term, main = "Loan term distribution", plotit = TRUE)
Desc(OurData$grade, main = "Loan grade distribution", plotit = TRUE)
Desc(OurData$emp_length, main = "Employement length distribution", plotit = TRUE)
Desc(OurData$annual_inc, main = "Anual income distribution", plotit = TRUE)
Desc(OurData$addr_state, main = "States distribution", plotit = TRUE)
Desc(OurData$application_type, main = "Application type distribution", plotit = TRUE)
Desc(OurData$loan_status, main = "Loan status", plotit = T)
Desc(OurData$purpose, main = "Loan purposes", plotit = TRUE)
Desc(OurData$title, main = "Loan titles", plotit = TRUE)


OurData <- OurData[,-which(colnames(OurData) %in% c("id","member_id","zip_code", "emp_title", "url", "desc","title","verification_status","issue_d","earliest_cr_line","last_pymnt_d","next_pymnt_d","pymnt_plan","initial_list_status","addr_state","last_credit_pull_d","mths_since_last_delinq", "verification_status_joint", "policy_code","dti","inq_last_12m.."))]

str(OurData)
dim(OurData)
lack_of_data_idx <- names(OurData)[colSums(!is.na(OurData)) < 887379*0.5]
lack_of_data_idx
OurData <- OurData[,-which(colnames(OurData) %in% lack_of_data_idx)]
str(OurData)
dim(OurData)
summary(OurData)

OurData <- OurData[complete.cases(OurData), ]
summary(OurData)
dim(OurData)



Desc(OurData$loan_amnt, main = "Loan amount distribution", plotit = TRUE)
Desc(OurData$term, main = "Loan term distribution", plotit = TRUE)
Desc(OurData$grade, main = "Loan grade distribution", plotit = TRUE)
Desc(OurData$emp_length, main = "Employement length distribution", plotit = TRUE)
Desc(OurData$annual_inc, main = "Anual income distribution", plotit = TRUE)
Desc(OurData$addr_state, main = "States distribution", plotit = TRUE)
Desc(OurData$application_type, main = "Application type distribution", plotit = TRUE)
Desc(OurData$loan_status, plotit = T)
Desc(OurData$purpose, main = "Loan purposes", plotit = TRUE)



levels(OurData$loan_status)

Current <- OurData[OurData$loan_status=="Current",]
Charged_off <- OurData[OurData$loan_status=="Charged Off",]
DNMTCP_CO <- OurData[OurData$loan_status %in% c("Does not meet the credit policy. Status:Charged Off"),]
DNMTCP_FP <- OurData[OurData$loan_status %in% c("Does not meet the credit policy. Status:Fully Paid"),]
Late <- OurData[OurData$loan_status %in% c("Late (16-30 days)","Late (31-120 days)"),]
Default <- OurData[OurData$loan_status=="Default",]
Fully_paid <- OurData[OurData$loan_status=="Fully Paid",]
In_Grace <- OurData[OurData$loan_status=="In Grace Period",]
Issued <- OurData[OurData$loan_status=="Issued",]

# rm(OurData)

Good_loan <- rbind(Fully_paid,DNMTCP_FP)

nrow(Good_loan)

Bad_loan <- rbind(Charged_off,DNMTCP_CO, Late,In_Grace)

nrow(Bad_loan)

print(paste0("Good loans number: ", nrow(Good_loan)))
print(paste0("Bad loans number: ", nrow(Bad_loan)))

print(paste0("Ratio between classes: ", round(nrow(Bad_loan)/nrow(Good_loan),3)))

print("Printing percent of classes in dataset")
print(paste0("Good loans: ", round(nrow(Good_loan)/(nrow(Good_loan)+nrow(Bad_loan))*100,2)," %"))
print(paste0("Bad loans: ", round(nrow(Bad_loan)/(nrow(Good_loan)+nrow(Bad_loan))*100,2)," %"))

rm(Charged_off)
rm(DNMTCP_CO)
rm(DNMTCP_FP)
rm(Late)
rm(Default)
rm(Fully_paid)
rm(In_Grace)
rm(Issued)

# Making smaller sample size for calculation
Good_loan <- Good_loan[sample(nrow(Good_loan), floor(nrow(Good_loan)*0.2)), ]
Bad_loan <- Bad_loan[sample(nrow(Bad_loan), floor(nrow(Bad_loan)*0.2)), ]


Good_loan$loan_status <- "False"
Bad_loan$loan_status <- "True"

myData <- rbind(Good_loan, Bad_loan)
names(myData)[names(myData) == 'loan_status'] <- 'Churn'
colY <- "Churn"
idxY <- which(colnames(myData) %in% colY)

dim(myData)

# create formulas - long for neuralnet, short for logit and LDA
nmd <- names(myData)
formulaLong <- as.formula(paste(paste(colY," ~",sep=""), paste(nmd[!nmd %in% colY], collapse = " + ")))
formulaShort <- as.formula(paste(paste(colY,".",sep="~")))  

myData$Churn <- as.factor(myData$Churn)
str(myData)
dim(myData)

Clean <- HARF(formulaLong, myData, agreementLevel=0.75, ntree=500)

myData<- Clean$cleanData


print(paste0("Good loans number: ", nrow(myData[myData$Churn=="False",])))
print(paste0("Bad loans number: ", nrow(myData[myData$Churn=="True",])))

print(paste0("Ratio between classes: ", round(nrow(myData[myData$Churn=="True",])/nrow(myData[myData$Churn=="False",]),3)))

print("Printing percent of classes in dataset")
print(paste0("Good loans: ", round(nrow(myData[myData$Churn=="False",])/(nrow(myData[myData$Churn=="False",])+nrow(myData[myData$Churn=="True",]))*100,2)," %"))

print(paste0("Bad loans: ",
             round(nrow(myData[myData$Churn=="True",])/(nrow(myData[myData$Churn=="False",])+nrow(myData[myData$Churn=="True",]))*100,2)," %"))


# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- caret::train(formulaShort, data=myData, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


str(myData)


# prepare normalization routine for a data set
procValues <- caret::preProcess(factorsNumeric(myData[,-idxY]), method = c("center", "scale"))

gc()

# random forest settings
mtry <- floor(sqrt(ncol(myData)-1))
ntree <- 500
ptree <- 0

# k-NN settings
knn_neighs <- 50
hiddenSize <- 20

k <- 3 # number of CV folds
myFolds <- createFolds(myData[,colY],k) # CV, stratified on label

# SVM settings
sigmas <- sigest(formulaShort, myData, frac = 1, scaled = TRUE)
grid <- expand.grid(sigma = round(seq(sigmas[1], sigmas[3], len=3),4), C = 2^c(-3:2))
ctrl <- trainControl(method="cv", number=10, classProbs=T, summaryFunction=twoClassSummary, allowParallel = T) # mnLogLoss, to use AUC set summaryFunction=twoClassSummary

gc()
myResults <- NULL

for (i in 1:k) {
  tstInd <- myFolds[[i]]
  
  trnIdx <- as.logical(rep(1,1,nrow(myData)))
  
  trnIdx[tstInd] <- FALSE
  
  trnInd <- which(trnIdx)
  
  target <- as.logical(myData[tstInd,idxY])
  
  trnDataProc <- predict(procValues, factorsNumeric(myData[trnInd,-idxY]))
  tstDataProc <- predict(procValues, factorsNumeric(myData[tstInd,-idxY]))
  
  
  cat(sprintf("\nCV fold %d out of %d / Random Forest\n", i, k))
  model_classwt <- prop.table(table(myData[trnInd,idxY]))
  rf_model <- tuneRF(myData[trnInd,-idxY], myData[trnInd,idxY], mtryStart = mtry, ntreeTry = ntree,
                     stepFactor = 2, improve = 0.01, plot=FALSE, doBest=TRUE,
                     classwt = model_classwt, cutoff = model_classwt,
                     strata = Y, replace = FALSE, importance=FALSE, do.trace = ptree)
  model <- rep("RF",length(target))
  soft <- predict(rf_model,myData[tstInd,-idxY],type="prob")
  score <- soft[,2]
  myResults <- rbind(myResults,data.frame(tstInd,model,score,target))
  
  Start = Sys.time()
  cat(sprintf("\nCV fold %d out of %d / Naive Bayes\n", i, k))
  nb_model <- naiveBayes(formulaShort, data=myData[trnInd,])
  model <- rep("NB",length(target))
  soft <- predict(nb_model,myData[tstInd,-idxY], type="raw")
  score <- soft[,2]
  myResults <- rbind(myResults,data.frame(tstInd, model,score,target))
  rm(nb_model)
  Time_to_complete= Sys.time() - Start
  Time_to_complete
  
  Start = Sys.time()
  cat(sprintf("\nCV fold %d out of %d / k-Nearest Neighbors\n", i, k))
  knn_model <- knn(trnDataProc, tstDataProc, myData[trnInd,idxY], k = knn_neighs, prob = TRUE, algorithm = "kd_tree")
  model <- rep("kNN",length(target))
  score <- 1-abs(as.numeric(knn_model)-1-attr(knn_model,"prob"))
  myResults <- rbind(myResults,data.frame(tstInd,model,score,target))
  rm(knn_model)
  
  
  
  # cat(sprintf("\nCV fold %d out of %d / Logistic Regression\n", i, k))
  # logit_model <- glm(formulaShort,family=binomial(link='logit'),data=myData[trnInd,])
  # model <- rep("logit",length(target))
  # score <- predict(logit_model,myData[tstInd,-idxY],type="response")
  # myResults <- rbind(myResults,data.frame(tstInd,model,score,target))
  # rm(logit_model)
  
  
  
  grid <- expand.grid(sigma = round(seq(sigmas[1], sigmas[3], len=3),4), C = 2^c(-3:2))
  cat(sprintf("\nCV fold %d out of %d / SVM with RBF kernel\n", i, k))
  svm_model <- caret::train(formulaShort, data = cbind(trnDataProc,Churn=myData[trnInd,idxY]),
                            method = "svmRadialSigma", metric="ROC",
                            tuneGrid = grid, trControl = ctrl)
  print(svm_model)
  model <- rep("SVM", length(target))
  soft <- predict(svm_model, tstDataProc, type="prob")
  score <- soft[,2]
  myResults <- rbind(myResults,data.frame(tstInd, model,score,target))
  rm(svm_model)
  
  
  grid <- expand.grid(size=c(5,10,15,20,c(5,5),c(10,10),c(20,20),c(5,10,5),c(10,20,10)), decay=c(0.00001, 0.0001, 0.001, 0.01, 0.1))
  cat(sprintf("\nCV fold %d out of %d / Neural Network \n", i, k))
  nn_model <- caret::train(formulaShort, data = cbind(trnDataProc,Churn=myData[trnInd,idxY]),
                           method = "nnet",tuneGrid = grid, trControl = ctrl)
  print(nn_model)
  model <- rep("NN", length(target))
  soft <- predict(nn_model, tstDataProc, type="prob")
  score <- soft[,2]
  myResults <- rbind(myResults,data.frame(tstInd, model,score,target))
  rm(nn_model)
  gc()
}

myModels <- levels(myResults[,"model"])
myScores <- spread(myResults, model, score)

# confusion matrix @ EER threshold
myF <- NULL
for (i in 1:length(myModels)) {
  opt.cut.result <- optimal.cutpoints(X = myModels[i], status = "target", tag.healthy = 0, methods = "SpEqualSe", data = myScores, trace = F)
  threshold <- opt.cut.result$SpEqualSe$Global$optimal.cutoff$cutoff
  confusionMatrix <- caret::confusionMatrix(as.factor(myScores[,myModels[i]]>=threshold),as.factor(myScores$target),positive="TRUE",mode="everything")
  cat(paste0(myModels[i],'\n'))
  print(confusionMatrix)
  myF <- c(myF,as.numeric(confusionMatrix$byClass['F1']))
}

# ROC curves
# myModelNames <- NULL
# i <- 1
# performance <- roc.plot(myResults[myResults[,"model"]==myModels[i],],i,traditional=TRUE)
# myModelNames[i] <- sprintf('%s AUC=%5.3f',myModels[i],1-performance['pAUC'])
# for (i in 2:length(myModels)) {
#   performance <- roc.plot(myResults[myResults[,"model"]==myModels[i],],i,traditional=TRUE)
#   myModelNames[i] <- sprintf('%s AUC=%5.3f',myModels[i],1-performance['pAUC'])
# }
# legend(0.3,0.5,myModelNames,lty=rep(1,1,length(myModels)),col=1:length(myModels))

# # DET curves
# myModelNames <- NULL
# det.plot(NULL,1,xmax=75,ymax=75)
# for (i in 1:length(myModels)) {
#   performance <- det.plot(myResults[myResults[,"model"]==myModels[i],],nr=i+1)
#   myModelNames[i] <- sprintf('%s EER=%5.2f%%',myModels[i],performance['eer'])
# }
# legend(-3.2,-1.24,myModelNames,lty=rep(1,1,length(myModels)),col=2:(length(myModels)+1))

# # Precision-Recall curves
myScores <- spread(myResults, model, score)
myLegend <- paste0(myModels, " F1=", format(myF,digits=3))
msmdat <- mmdata(myScores[,-c(1,2)], myScores[,2], posclass = TRUE, modnames = myLegend)
plot(autoplot(evalmod(msmdat), "PRC", type="b"))

# save scores for uploading to http://www.biosoft.hacettepe.edu.tr/easyROC/
myScores$target <- as.numeric(myScores$target)
write.csv(myScores,file="churn_scores.csv")

print(Sys.time() - TIME)

new <- caret::preProcess(factorsNumeric(Current), method = c("center", "scale"))
soft_pred <- predict(rf_model,Current,type="prob")
score_pred <- soft_pred[,2]
str(score_pred)
Desc(score_pred, main = "Loan amount distribution", plotit = TRUE)


current_pred <- predict(rf_model,Current,type="response")
str(current_pred)
Desc(current_pred, main = "Loan amount distribution")
plot(current_pred)