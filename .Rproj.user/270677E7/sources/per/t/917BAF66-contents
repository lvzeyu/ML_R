library(haven)
library(labelled)
library(tidyverse)
library("parsnip")
library("neuralnet")
library(modeldata)
library(skimr)
library(tidymodels)
library(iml)
library(shapviz)

convert_labels_to_na <- function(variable) {
  labels <- labelled::get_value_labels(variable)
  
  # Check if variable is labelled
  if (is.null(labels)) {
    return(variable)
  }
  
  labels<-data.frame(labels)
  values_to_na <- c("非該当", "無回答", "わからない")
  na_values <- labels[rownames(labels) %in% values_to_na, ]
  variable[variable%in%na_values] <- NA
  return(variable)
}

PY100 <- read_dta("Data/PY100/PY100.dta")


unmarriage<-PY100%>%
  subset(ZQ50==1)

wave1<-unmarriage%>%
  select(2:211,295:430,509:537)

wave1["marriage_status"]<-unmarriage$IQ45

for (colname in colnames(wave1)) {
  wave1[[colname]] <- convert_labels_to_na(wave1[[colname]])
}

wave1$marriage_status<-ifelse(wave1$marriage_status==2,0,1)

wave1<-wave1%>%
  drop_na("marriage_status")

wave1 <- wave1[, apply(wave1, 2, function(x) sum(is.na(x)) <= 500)]


wave1 <- wave1 %>%
  mutate_all(~ifelse(is.na(.), mean(., na.rm = TRUE), .))

wave1$marriage_status<-as.factor(wave1$marriage_status)


train<-wave1[1:800,]
test<-wave1[800:nrow(wave1),]

rf_spec = rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")

xgb_spec <- boost_tree() %>%
  set_mode("classification") %>%
  set_engine("xgboost")
knn_spec <- nearest_neighbor(neighbors = 5) %>%
  set_mode("classification") %>%
  set_engine("kknn")
nn_spec <- mlp(hidden_units = 10) %>%
  set_mode("classification") %>%
  set_engine("keras")


rec = recipe(marriage_status~.,data=wave1) %>%
  step_dummy(marriage_status, -marriage_status)



xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_spec)

rf_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

knn_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(knn_spec)

nn_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(nn_spec)


xgb_fit <- xgb_wf %>%
  fit(data = train)

rf_fit <- rf_wf %>%
  fit(data = train)

knn_fit <- knn_wf %>%
  fit(data = train)

nn_fit <- nn_wf %>%
  fit(data = train)


xgb_res <- predict(xgb_fit, test) %>%
  bind_cols(test) %>%
  metrics(truth = marriage_status, estimate = .pred_class)

rf_res <- predict(rf_fit, test) %>%
  bind_cols(test) %>%
  metrics(truth = marriage_status, estimate = .pred_class)

predict_func <- function(model, newdata) {
  predict(model, data = newdata, type = "response")$predictions
}


trained_rf <- extract_fit_parsnip(rf_fit)$fit

wave1_predictors <- wave1 %>%
  select(-marriage_status)

library(iml)

dia_small_prep <- bake(
  prep(rec), 
  has_role("marriage_status"),
  new_data = wave1_predictors, 
  composition = "matrix"
)


shap <- shapviz(extract_fit_parsnip(xgb_fit), X_pred = dia_small_prep, X = wave1_predictors)

