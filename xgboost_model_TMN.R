install.packages('caret')
install.packages('pROC')
install.packages("MLmetrics")
install.packages('precrec')


library(caret)
library(dplyr)
library(ggplot2)
library(ModelMetrics)
library(readr)
library(pROC)
library(MLmetrics)
library(precrec)

#import dataset : data <- read.csv2("X.csv", dec='.', sep=';')
data <- data_TCGA_phenotype_normalize
names(data)[1] <- "phenotype"


# Division dataset train/ test 80/20% 
set.seed(123)
train_ind <- sample(1:nrow(data), size = floor(80*nrow(data)/100))

tr <- data[train_ind,]
x_train <- tr[,-1]
te <- data[-train_ind,]
x_test <- te[,-1]

# dimension data
percentage <- prop.table(table(data$phenotype))*100
cbind(freq=table(data$phenotype), percentage=percentage)

# dimension train
percentage <- prop.table(table(tr$phenotype))*100
cbind(freq=table(tr$phenotype), percentage=percentage)

# dimension test
percentage <- prop.table(table(te$phenotype))*100
cbind(freq=table(te$phenotype), percentage=percentage)


# xgboost with Default Hyperparameters 

grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, 
  allowParallel = TRUE 
)

xgb_base <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE, )

#Number of Iterations and the Learning Rate
nrounds <- 1000

tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)


tune_control <- caret::trainControl(
  method = "cv", 
  number = 3, 
  verboseIter = FALSE, 
  allowParallel = TRUE 
)

xgb_tune <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE,
)

xgb_tune$bestTune


# Step 2: Maximum Depth and Minimum Child Weight

tune_grid2 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = ifelse(xgb_tune$bestTune$max_depth == 4,
                     c(xgb_tune$bestTune$max_depth:6),
                     xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3),
  subsample = 1
)

xgb_tune2 <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE,) 

xgb_tune2$bestTune

#Step 3: Column and Row Sampling

tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = tune_control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE,)

xgb_tune3$bestTune


#Step 4: Gamma
tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE,
)

xgb_tune4$bestTune


#Step 5: Reducing the Learning Rate
tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune5 <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE, )

xgb_tune5$bestTune


#Fitting the Model
final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
)

print(final_grid)

xgb_model <- caret::train(
  x = as.matrix(x_train),
  y = tr$phenotype,
  trControl = train_control,
  tuneGrid = final_grid,
  method = "xgbTree",
  verbose = TRUE, 
)

print(xgb_model)


#Evaluating the Model Performance on train set

xgbpredtrain <- predict(xgb_model, x_train)
print(xgbpredtrain)
caret::confusionMatrix (xgbpredtrain, tr$phenotype, positive = "0", mode = 'everything')
caret::confusionMatrix (xgbpredtrain, tr$phenotype, positive = "1", mode = 'everything')


#Evaluating the Model Performance on test set
xgbpred <- predict (xgb_model,x_test)
print(xgbpred)
summary(xgbpred)


# Confusion Matrix
caret::confusionMatrix (xgbpred, te$phenotype, positive = "0", mode = 'everything')
caret::confusionMatrix (xgbpred, te$phenotype, positive = "1", mode = 'everything')

# ROC

xgbpred <- predict (xgb_model,x_test, type = "prob")
print(xgbpred)
ROC1 <- roc(predictor= xgbpred$`1`, response= predfinal$phenotype, levels = (levels(predfinal$phenotype)))
ROC1$auc
plot(ROC1,main="xgboost ROC immune subtype", col = "red")
text(0.5,0.5, paste("AUC=", format(ROC1$auc, digits=5, scientific= FALSE)))

# AUCpr
PRAUC(y_pred = xgbpred$`1`, y_true = predfinal$phenotype)
PRAUC(y_pred = xgbpred$`0`, y_true = predfinal$phenotype)

precrec_obj <- evalmod(scores = xgbpred$`0`, labels = predfinal$phenotype)
autoplot(precrec_obj)

# features importance
caret_imp <- varImp(xgb_model) 
plot(caret_imp, top = 5)
ggplot(caret_imp,top = 5) +
  theme_minimal()

# external validation on datatest GHPS
test_valid <- data_COSMOS_final_01_normalise_ICC
names(test_valid)[1] <- "phenotype"
test_x_valid <- test_valid[,-1]

xgbpred_valid <- predict(xgb_model,test_x_valid)
print(xgbpred_valid)

# confusion matrix GHPS validation
library(caret)
test_valid$phenotype <- factor(test_valid$phenotype, levels = c("1","0"))
cm <- caret::confusionMatrix (xgbpred_valid, test_valid$phenotype, mode='everything',positive = "1")
caret::confusionMatrix (xgbpred_valid, test_valid$phenotype, mode='everything',positive = "0")


# ROC GHPS validation
library(pROC)
xgbpred_valid <- predict(xgb_model,test_x_valid, type="prob")

ROC1 <- roc(predictor= xgbpred_valid$`1`, response= test_valid$phenotype, levels = (levels(predfinal$phenotype)))
ROC1$auc
plot(ROC1,main="xgboost ROC", col = "red")
text(0.5,0.5, paste("AUC=", format(ROC1$auc, digits=5, scientific= FALSE)))

# AUCpr GHPS validation
PRAUC(y_pred = xgbpred_valid$`1`, y_true = test_valid$phenotype)
PRAUC(y_pred = xgbpred_valid$`0`, y_true = test_valid$phenotype)

#plot PR-AUC curve
precrec_obj <- evalmod(scores = xgbpred_valid$`1`, labels = test_valid$phenotype, mode="basic")
precrec_obj <- evalmod(scores = xgbpred_valid$`0`, labels = test_valid$phenotype)
autoplot(precrec_obj)

# Draw confusion matrix 
draw_confusion_matrix <- function(cm) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}
draw_confusion_matrix(cm)


