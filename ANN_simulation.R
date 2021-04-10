----------------------------------
  
Title: "Neural Network Simulations"
Autore: "Ruggero Cazzato"
Data: "09/04/2021"

----------------------------------

library(readxl);library(tidyverse)

#import trainset e testset
train <- read_xlsx("Dataset_finale.xlsx", sheet = 2)
test <- read_xlsx("Dataset_finale.xlsx", sheet = 3)

train <- train %>% 
  select(Default, x1, x2, x3, x4, x5, x6, x7, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18)

test <- test %>% 
  select(Default, x1, x2, x3, x4, x5, x6, x7, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18)

#scaling variables
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

train[] <- lapply(train, normalize)
test[] <- lapply(test, normalize)
#check for missing values
lapply(train, function(x) sum(is.na(train)))
lapply(test, function(x) sum(is.na(test)))

train<- sample_n(train, nrow(train))
test<- sample_n(test, nrow(test))

#model formula
ann.equation <- "Default ~ x1 + x2 + x3 + x4 + x6 + x7 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18"

#N neural network estimation where "N = neurons*times"
#"times" are the times a neural network is estimated for each neuron
ANN_simulation <- function(formula, train_set, test_set, neurons, times, threshold = 0.01, stepmax = 100000, 
                           rep = 1, startweights = NULL, learningrate.limit = NULL, 
                           learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
                           lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", 
                           err.fct = "sse", act.fct = "logistic", linear.output = TRUE, 
                           exclude = NULL, constant.weights = NULL, likelihood = FALSE) {
  
  total_simulation <- times*neurons
  
  for(i in 1:neurons) { 
    accuracy<-NULL
    sensitivity<-NULL 
    specificity<-NULL 
    seed<-NULL
    
    for(i in 1:total_simulation) {
      
      smp <- 1:1000000
      newseed<-sample(smp,1)
      set.seed(newseed)
      
      ann.models <- neuralnet::neuralnet(formula = formula, data=train_set, hidden = i,
                                         threshold = threshold, stepmax = stepmax, rep = rep,
                                         startweights = startweights, learningrate.limit = learningrate.limit, 
                                         learningrate.factor = learningrate.factor, learningrate = learningrate, 
                                         lifesign = lifesign, lifesign.step = lifesign.step, algorithm = algorithm, 
                                         err.fct = err.fct, act.fct = act.fct, linear.output = linear.output, 
                                         exclude = exclude, constant.weights = constant.weights, likelihood = likelihood)
      
      ann.predictions <- neuralnet::compute(ann.models, test_set)
      
      results <- data.frame(actual = test_set$Default, prediction = ann.predictions$net.result)
      roundedresults<-sapply(results, round, digits = 0)
      roundedresults<-data.frame(roundedresults)
      roundedresults$actual<-as.factor(roundedresults$actual)
      roundedresults$prediction<-as.factor(roundedresults$prediction)
      
      CM <- caret::confusionMatrix(roundedresults$prediction, roundedresults$actual, positive = "1")
      newaccuracy <- CM$overall[1]
      newsensitivity<-CM$byClass[1] 
      newspecificity<-CM$byClass[2]
      
      accuracy<-c(accuracy,newaccuracy)
      sensitivity<-c(sensitivity,newsensitivity)
      specificity<-c(specificity,newspecificity)
      seed<-c(seed,newseed)
      
    }
    
  } 
  
  table <- as.data.frame(matrix(nrow = neurons*times, ncol = 5))
  colnames(table) = c("neurons", "accuracy", "sensitivity", "specificity", "seed")
  table$neurons <- rep(1:neurons, each = times)
  table$accuracy <- accuracy
  table$sensitivity <- sensitivity
  table$specificity <- specificity
  table$seed <- seed
  simulation_table <<- table
 
}


ANN_simulation(formula = ann.equation, train_set = train, test_set = test,
               neurons = 10, times = 1, stepmax = 1e+09, threshold = 0.05,
               err.fct = "sse", act.fct = "logistic", linear.output = FALSE)

#the output is a table in which are recorded for each estimate: the number of neurons used,
#accuracy, sensitivity, specificity and the seed used to obtain that particular result.
#In this way the result can be easily replicated.



