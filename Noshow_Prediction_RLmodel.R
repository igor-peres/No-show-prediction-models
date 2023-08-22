library(tidyverse)
library(lubridate)
library(caret)
library(mlbench)
library(Hmisc)
library(MLmetrics)
library(grid)
library(ModelMetrics)
library(givitiR)
library(DescTools)
library(corrplot)
library(pROC)

load("data.RData")

#SEPARANDO TREINO DO TESTE
save(data, file="data.RData")
set.seed(998)
inTraining <- createDataPartition(data$`No-show`,
                                  p = .8, list = FALSE)
training <- data[inTraining,]
testing  <- data[-inTraining,]

summary(training)

# load("training.RData")

#Correlation analysis
cramer_tab = PairApply(training,
                       CramerV, symmetric = TRUE)
cramer_tab[which(is.na(cramer_tab[,])==T)] = 0
corrplot.mixed(cramer_tab, tl.pos = "lt")


#Prediction

fitControl = trainControl(method = "cv", number = 10,
                          summaryFunction = twoClassSummary, 
                          classProbs = TRUE, savePredictions = T)

#Model = Logistic Regression
#Optimization Metric = ROC Curve 

set.seed(4760)
RL_model <- train(`No-show` ~   `Month` + `Sex` +  `Age` + 
                     + `Insurance Company` +  `Speciality`,
                  data=training,
                  method="glm",
                  family=binomial(link = "logit"),
                  metric="ROC ",
                  trControl=fitControl)
summary(RL_model$results)

 
# Evaluation of TESTING set -----

# load("testing.RData")
# load("Noshow_Prediction_RLmodel.RData")

Predicted <- data.frame(Predicted=predict(RL_model, newdata=testing, type = "prob"))
Observed = if_else(testing$`No-show` == "Faltou",1,0)

ModelMetrics::auc(predicted = Predicted$Predicted.Faltou, actual = Observed)

#Estimating best cut-off
ROC = roc(response= Observed,
          predictor =  Predicted$Predicted.Faltou,
          levels = c("0","1"),
          direction = "<")

cutoff_otimo = coords(ROC,x="best",best.method = "youden",
                      # best.weights = c(2,0.1),
                      ret= c("threshold","sensitivity","specificity","ppv","npv"),
                      transpose=F
)
cutoff_otimo

#cutoff_otimo$threshold = 0.07383589

ppv(predicted = Predicted$Predicted.Faltou, actual = Observed,
    cutoff = cutoff_otimo$threshold)
npv(predicted = Predicted$Predicted.Faltou, actual = Observed,
    cutoff = cutoff_otimo$threshold)
sensitivity(predicted = Predicted$Predicted.Faltou, actual = Observed,
            cutoff = cutoff_otimo$threshold)
specificity(predicted = Predicted$Predicted.Faltou, actual = Observed,
            cutoff = cutoff_otimo$threshold)

#Calibration Belt

cb = givitiCalibrationBelt(o = Observed,
                           e = Predicted$Predicted.Faltou, 
                           devel = "external"
)
plot(cb,main="Calibration Belt - RL",
     xlab="Predicted probabilities of no-show",
     ylab="Observed no-show")

