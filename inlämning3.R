#Elev:      Marcus Stenroos
#Lärare:    Martin Gräslund
#Inlämning: 3


#install.packages("e1071")
#install.packages("dplyr")
#install.packages("randomForest")
#install.packages("caret")
#install.packages("C50")
#install.packages("irr")
#install.packages('libcoin', dependencies = T) #C50 paketet fungerade inte
#install.packages('C50', dependencies = T)
#install.packages("gmodels")
#install.packages("psych")


library(e1071)
library(dplyr)
library(randomForest)
library(caret)
library(irr)
library(C50)
library(gmodels)
library(psych)


#Importerar datasettet manuellt

str(Diabetes)

#Ändrar namn på kolumnen som är felaktig skriver
names(Diabetes)[1] = "Graviditeter"
Diabetes$Diabetes <- as.factor(Diabetes$Diabetes)

#Tar reda på medelvärdet för varje kolumn
mean_blodtryck <- sum(Diabetes$Blodtryck)/nrow(filter(Diabetes, Blodtryck!=0))
mean_glukos <- sum(Diabetes$Glukostolerans)/nrow(filter(Diabetes, Glukostolerans!=0))
mean_hud <- sum(Diabetes$Hudtjocklek)/nrow(filter(Diabetes, Hudtjocklek!=0))
mean_insulinin <- sum(Diabetes$insulininjektion)/nrow(filter(Diabetes, insulininjektion!=0))
mean_bmi <- sum(Diabetes$BMI)/nrow(filter(Diabetes, BMI!=0))
mean_familj <- sum(Diabetes$Diabetes.i.familj)/nrow(filter(Diabetes, Diabetes.i.familj!=0))
mean_alder <- sum(Diabetes$Alder)/nrow(filter(Diabetes, Alder!=0))

#Sätter in medelvärdet i respektive kolumn där raden innehåller en nolla
Diabetes$Blodtryck <- replace(Diabetes$Blodtryck, Diabetes$Blodtrycks == 0, mean_blodtryck)
Diabetes$Glukostolerans <- replace(Diabetes$Glukostolerans, Diabetes$Glukostolerans == 0, mean_glukos)
Diabetes$Hudtjocklek <- replace(Diabetes$Hudtjocklek, Diabetes$Hudtjocklek == 0, mean_hud)
Diabetes$insulininjektion <- replace(Diabetes$insulininjektion, Diabetes$insulininjektion == 0, mean_insulinin)
Diabetes$BMI <- replace(Diabetes$BMI, Diabetes$BMI == 0, mean_bmi)
Diabetes$Diabetes.i.familj <- replace(Diabetes$Diabetes.i.familj, Diabetes$Diabetes.i.familj == 0, mean_familj)
Diabetes$Alder <- replace(Diabetes$Alder, Diabetes$Alder == 0, mean_alder)


#Använder Carets inbyggda funktion createDataPartition för att skapa ett träning, och ett testset
in_train <- createDataPartition(Diabetes$Diabetes, p = 0.75, 
                                list = FALSE)
#Uppdelningen sker genom att "in_train" vektorn indikerar vilka rader som sparats där i som i sin tur
# flyttas över till diabetes_trains dataram. De raderna som inte finns med i "in_train" läggs in i diabetes_test.
diabetes_train <- Diabetes[in_train, ]
diabetes_test <- Diabetes[-in_train, ]



#Fixar k-folds dvs crossvalidation för att estimera datamodellens prestanda.
#Här väljer jag att blanda data 10 gånger med helt slumpmässiga uppdelningar av datamängden för att undvika att
#data slumpas med samma mått mer än en gång.
#Setseed funktionen används för att garantera att resultatet förblir konsekvent om samma kod skulle köras igen
set.seed(123)
folds <- createFolds(Diabetes$Diabetes, k = 10)

#För att se resultatet kan man köra koden nedan.
#str(folds)

#För att skapa ett tränings och testset för utvärdering och uppbyggande av en datamodell
diabetes_folds_test <- Diabetes[folds$Fold01, ]
diabetes_folds_train <- Diabetes[-folds$Fold01, ]


#I koden nedan delas data upp i tränings och testset, och ett c5,0 decisiontree byggs upp vilket genererar
#prediktioner från testdata och sedan jämförs de aktuella värden med de predikterade värden genom att använda kappa2 funktionen.
#Resultatet av kappa statistiken sparas o cv_result
cv_result <- lapply(folds, function(x){
  diabetes_train <- Diabetes[-x, ]
  diabetes_test <- Diabetes[x, ]
  diabetes_model <- C5.0 (Diabetes ~., data = diabetes_train)
  diabetes_pred <- predict(diabetes_model, diabetes_test)
  diabetes_actual <- diabetes_test$Diabetes
  kappa <- kappa2(data.frame(diabetes_actual, diabetes_pred))$value
  return(kappa)

})

#Resultatet av körningen kan göras genom kod nedan
#str(cv_result)

#Ett sista steg för 10-"blandning" för crossvalidation är att beräkna medelvärdet av alla 10 värden
#medel för kappa värdet = resultatet av uträkningen
mean(unlist(cv_result))*100



#Testar hur resultatet blir med Random Forest
#Bygger modellen
set.seed(300)
diabetes_rf <- randomForest(Diabetes ~ ., data = diabetes_folds_test)

#Använder Caret funktionen och ställer in hur data ska tränas
ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats = 10)

#Sätter upp tuningen för random forest mtry vilket definierar hur stora ängder data 
#som slumpas vid varje delning
grid_rf <- expand.grid(.mtry = c(2, 4, 8, 16))


#Vi flyttar det resulterande gallrandet till train-funktionen med ctrl-objektet som följer.
set.seed(300)
diabetes_modell_rf <- train(Diabetes ~ ., data = diabetes_folds_test, method = "rf",
              metric = "Kappa", trControl = ctrl,
              tuneGrid = grid_rf)

#Resultat av RF körning
diabetes_modell_rf

#Sedan jämförs resultatet med en boostad tree med olika iterationer
grid_c50 <- expand.grid(.model = "tree",
                        .trials = c(10, 20, 30, 40),
                        .winnow = "FALSE")

#set.seed(300)
m_c50 <- train(Diabetes ~ ., data = diabetes_folds_test, method = "C5.0",
                 metric = "Kappa", trControl = ctrl,
                 tuneGrid = grid_c50)

#Skriver ut innan förbättring
diabetes_modell_rf
#Skriver ut resultatet efter boosting
m_c50



#Bygger upp modellen med vanlig kostnad
cost_diabetes <- C5.0(diabetes_train[-9], diabetes_train$Diabetes)

#Gör en prediktion
diabetes_cost_pred_cheap <- predict(cost_diabetes, diabetes_test)

#Skriver ut resultatet
CrossTable(diabetes_test$Diabetes, diabetes_cost_pred_cheap,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))




#Skapar en cost matris för att se hur utfallet påverkas av det
matrix_dimensions <- list(c("0", "1"), c("0", "1"))
names(matrix_dimensions) <- c("Predicted", "Actual")

#nedan om man vill kontrollera att matrisen blev rätt
#matrix_dimensions


#Sätter in att "kostnaden" ökar med  5 ggr
error_cost <- matrix(c(0, 1, 5, 0), nrow = 2,
                     dimnames = matrix_dimensions)




#Höjer cost med 5
#Bygger upp modellen
higher_cost_diabetes <- C5.0(diabetes_train[-9], diabetes_train$Diabetes,
                             costs = error_cost)

#Gör en prediktion
diabetes_cost_pred <- predict(higher_cost_diabetes, diabetes_test)


#Skriver ut resultatet
CrossTable(diabetes_test$Diabetes, diabetes_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))





#Skapar ett kontrollobjekt med namnet ctrl som använder 10-faldig kryssvalidering och "oneSE"-funktionen, 
ctrl <- trainControl(method = "cv",
                     selectionFunction = "oneSE")


#Använder expand.grid funktion, vilket skapar dataramar från kombinationerna av alla värden som medföljer.
grid <- expand.grid(.model = "tree",
                    .trials = c(1, 5, 10, 15, 20, 25, 30, 35),
                    .winnow = "FALSE")


#CARET körning. Med kontrollistan som skapades tidigare som grund är vi redo att köra en grundligt anpassad train-funktion
# mha Caret. Vi ställer in Set.Seed för att säkerställa resultat som kan repeteras. 
set.seed(300)
diabetes_model_c5 <- train(Diabetes ~., data = Diabetes, method = "C5.0",
metric = "Kappa",
trControl = ctrl,
tuneGrid = grid)

#Skriver ut resultatet av Caret körningen
diabetes_model_c5



#För att få uppskattade sannolikheter för varje klass, använd parametern typ = "prob". Skriver här ut de första raderna av Diabetes klassen
head(predict(diabetes_model_c5, Diabetes, type = "prob"))

#Kollar vilken variabel som påverkar mest för att drabbas av diabetes
pairs.panels(Diabetes[c("Diabetes", "Graviditeter", "Glukostolerans", "Blodtryck", "Hudtjocklek", "insulininjektion", 
                        "BMI", "Diabetes.i.familj", "Alder")])




##################################################################################################
#Skräp och tester här nedan

#Diabetes$Graviditeter <-as.factor(Diabetes$Graviditeter)
#Diabetes$Glukostolerans <-as.factor(Diabetes$Glukostolerans)
#Diabetes$Blodtryck <-as.factor(Diabetes$Blodtryck)
#Diabetes$Hudtjocklek <-as.factor(Diabetes$Hudtjocklek)
#Diabetes$insulininjektion <-as.factor(Diabetes$insulininjektion)
#Diabetes$BMI <-as.factor(Diabetes$BMI)
#Diabetes$Diabetes.i.familj <-as.factor(Diabetes$Diabetes.i.familj)
#Diabetes$Alder <-as.factor(Diabetes$Alder)
#Diabetes$Diabetes <-as.factor(Diabetes$Diabetes)

#Kontrollerar att inga nollor finns
#levels(Diabetes$Graviditeter)
#levels(Diabetes$Glukostolerans)
#levels(Diabetes$Blodtryck)
#levels(Diabetes$Hudtjocklek)
#levels(Diabetes$insulininjektion)
#levels(Diabetes$BMI)
#levels(Diabetes$Diabetes.i.familj)
#levels(Diabetes$Alder)
#levels(Diabetes$Diabetes)
#str(Diabetes)

#Diabetes
#str(Diabetes)         
