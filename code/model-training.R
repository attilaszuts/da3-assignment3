# setup -------------------------------------------------------------------
rm(list=ls())
# General
library(tidyverse)
library(skimr)
# Modeling
library(caret)
# Visualization
library(cowplot)
library(party)
library(GGally)
library(pROC)
library(knitr)
library(rattle)
source('code/helper.R')


# loading data ------------------------------------------------------------

df <- read.csv('data/clean/hotels-europe_price.csv')

# skim(df)

# pre-processing ----------------------------------------------------------

cities <- c("Berlin", "Munich", "Vienna", "Budapest", "Prague", "Warsaw")

df <- df %>% filter(city %in% cities)

# choose the one with the most variables
table(df$year, df$month, df$weekend)

data.frame(table(df$accommodation_type)) %>% arrange(desc(Freq)) %>% filter(Freq != 0)

# filter for 2017, december, weekend, one night, hotel
hotels <- c("Hotel", "Hostel", "Apart-hotel")
df <- df %>% filter(
  year == 2017, 
  month == 12, 
  weekend == 0, 
  nnights == 1,
  accommodation_type %in% hotels)

# skim(df)

# there is an outlier based on stars
# view(df[df$stars==4 & df$price > 3000,])
df <- df[df$price < 2000,]

# drop cols that are not needed
drop_vars <- c("year", "month", "weekend", "holiday", "country", "city_actual", "center1label", "center2label", "neighbourhood", "distance_alter", "nnights", "rating", "rating_reviewcount")
df <- df[, !(names(df) %in% drop_vars)]

# create flag for missing values
df <- df %>% 
  mutate(
    flag_stars = ifelse(is.na(stars), 1, 0),
    flag_ratingta = ifelse(is.na(ratingta), 1, 0),
    flag_ratingta_count = ifelse(is.na(ratingta_count), 1, 0)
  )

df <- df %>% 
  mutate(
    price_ln = log(price)
  )
# variables ---------------------------------------------------------------

target <- c("price")
target_ln <- c("price_ln")

preds_basic <- c("city", "distance", "stars", "flag_stars") 
preds_rate <- c("ratingta", "ratingta_count", "flag_ratingta", "flag_ratingta_count")
preds_extra <- c("scarce_room", "offer", "offer_cat", "accommodation_type")

formula_reg <- formula(paste0(c(target_ln, paste0(c(preds_basic, preds_rate), collapse = " + ")), collapse = " ~ "))
formula_enet <- formula(paste0(c(target_ln, paste0(c(preds_basic, preds_rate, preds_extra), collapse = " + ")), collapse = " ~ "))
formula_tree <- formula(paste0(c(target_ln, paste0(c(preds_basic, preds_rate, preds_extra), collapse = " + ")), collapse = " ~ "))

# EDA ---------------------------------------------------------------------

pl_ggcorrs <- ggcorr(df) + theme_bw()

pl_ggpairs <- ggpairs(df, columns = c("price_ln", "ratingta", "ratingta_count", "stars")) + theme_bw()

pl_price_hist <- df %>% 
  ggplot(aes(price)) + 
  geom_histogram(fill = "white", alpha = 0.4, color = "black", binwidth = 50) +
  labs(x = "Price", 
       y = "Absolute frequency") + 
  theme_bw()

pl_price_hist_log <- df %>% 
  ggplot(aes(log(price))) + 
  geom_histogram(fill = "white", alpha = 0.4, color = "black") +
  labs(x = "Log(Price)", 
       y = "Absolute frequency") + 
  theme_bw()


pl_price_stars <- df %>% 
  ggplot(aes(stars, log(price))) + 
  geom_point() + 
  geom_smooth(method = "loess") +
  labs(x = "Log(Price)", 
       y = "Absolute frequency") + 
  theme_bw()

pl_price_rating <- df %>% 
  ggplot(aes(ratingta, log(price))) + 
  geom_point() + 
  geom_smooth(method = "loess") +
  labs(x = "Log(Price)", 
       y = "Absolute frequency") + 
  theme_bw()

pl_price_distance <- df %>% 
  ggplot(aes(distance, log(price))) + 
  geom_point() + 
  geom_smooth(method = "loess") +
  labs(x = "Log(Price)", 
       y = "Absolute frequency") + 
  theme_bw()

pl_price_city <- df %>% 
  ggplot(aes(city, price)) + 
  geom_boxplot() + 
  labs(x = "City",
       y = "Price") + 
  theme_bw()

# train-test split --------------------------------------------------------
set.seed(1234)
training_ratio <- 0.7
train_indices <- createDataPartition(
  y = df[["price"]],
  times = 1,
  p = training_ratio,
  list = FALSE
) %>% as.vector()
data_train <- df[train_indices, ]
data_test <- df[-train_indices, ]

# setup results
models <- list()
runtimes <- list()

# setup CV
trctrl <- trainControl(
  method = "cv", 
  number = 10,
  verboseIter = T
)

# multiple regression -----------------------------------------------------

set.seed(1234)
start <- Sys.time()
model_reg <- train(
  formula_reg,
  method = "lm",
  data = data_train,
  trControl = trctrl,
  na.action = na.omit,
  metric = "RMSE"
)
end <- Sys.time()

# Save model props
models[["reg"]] <- model_reg
runtimes[["reg"]] <- end-start

# Notification
time <- as.numeric(round(end - start, 2), unit = "secs")
modelname <- "**multiple regression model**"
send_message(my_text = paste0(user, ", your ", modelname, " has finished training! It took: ", time, " seconds."))


model_reg

# elastic net -------------------------------------------------------------
enet_tune <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = seq(0.01, 0.1, by = 0.01)
)


set.seed(1234)
start <- Sys.time()
model_enet <- train(
  formula_enet,
  method = "glmnet",
  data = data_train,
  preProcess = c("center", "scale"),
  trControl = trctrl,
  tuneGrid = enet_tune,
  na.action = na.omit,
  metric = "RMSE"
)
end <- Sys.time()

# Save model props
models[["enet"]] <- model_enet
runtimes[["enet"]] <- end-start

# Notification
time <- as.numeric(round(end - start, 2), unit = "secs")
modelname <- "**elastic net model**"
send_message(my_text = paste0(user, ", your ", modelname, " has finished training! It took: ", time, " seconds."))

model_enet

# CART --------------------------------------------------------------------
cart_tune <- expand.grid(
  cp = seq(0.001, 0.02, by = 0.0005)
)

set.seed(1234)
start <- Sys.time()
model_cart <- train(
  formula_tree,
  method = "rpart",
  data = data_train,
  trControl = trctrl,
  tuneGrid = cart_tune,
  na.action = na.omit,
  metric = "RMSE"
)
end <- Sys.time()


# Save model props
models[["cart"]] <- model_cart
runtimes[["cart"]] <- end-start

# Notification
time <- as.numeric(round(end - start, 2), unit = "secs")
modelname <- "**regression tree (CART) model**"
send_message(my_text = paste0(user, ", your ", modelname, " has finished training! It took: ", time, " seconds."))

model_cart

# random forest -----------------------------------------------------------

forest_tune <- expand.grid(
  mtry = seq(20, 60, by = 5)
)

set.seed(1234)
start <- Sys.time()
model_rf <- train(
  formula_tree,
  method = "rf",
  data = data_train,
  trControl = trctrl,
  tuneGrid = forest_tune,
  na.action = na.omit,
  metric = "RMSE"
)
end <- Sys.time()

plot(model_rf)
# Save model props
models[["rf"]] <- model_rf
runtimes[["rf"]] <- end-start

# Notification
time <- as.numeric(round(end - start, 2), unit = "secs")
modelname <- "**random forest**"
send_message(my_text = paste0(user, ", your ", modelname, " has finished training! It took: ", time, " seconds."))

model_rf

# boosting ----------------------------------------------------------------

gbm_tune <- expand.grid(
  n.trees = 100,
  interaction.depth = (1:10),
  shrinkage = 0.1,
  n.minobsinnode = c(3, 5, 10)
)

set.seed(1234)
start <- Sys.time()
model_gbm <- train(
  formula_tree,
  method = "gbm",
  data = data_train,
  trControl = trctrl,
  tuneGrid = gbm_tune,
  na.action = na.omit,
  metric = "RMSE"
)
end <- Sys.time()


plot(model_gbm)
# Save model props
models[["gbm"]] <- model_gbm
runtimes[["gbm"]] <- end-start

# Notification
time <- as.numeric(round(end - start, 2), unit = "secs")
modelname <- "**Gradient Boosting Machine**"
send_message(my_text = paste0(user, ", your ", modelname, " has finished training! It took: ", time, " seconds."))


model_gbm

# model selection ---------------------------------------------------------

cv_rmse <- sort(
  unlist(
    lapply(models, function(model){
      mean(model$resample$RMSE)
    }
      )), 
  decreasing = F)

top_n <- 3 
top_models <- models[names(cv_rmse[1:top_n])]

rmses <- list()
for (model in 1:length(models)) {
  modelname <- names(models)[model]
  rmses[modelname] <- mean(models[[model]]$resample$RMSE)
}
rmses


# plots and tables -------------------------------------------------------------------

# Random forest
pl_model_rf <- plot(model_rf)

plot(varImp(model_rf))

varimpdf <- as.data.frame(model_rf$finalModel$importance) %>% filter(IncNodePurity > 0) %>% arrange(desc(IncNodePurity)) 
varimpdf['vars'] <- rownames(varimpdf)
rownames(varimpdf) <- NULL
colnames(varimpdf)[1] <- "purity"
pl_model_rf_varimp <- varimpdf %>% 
  ggplot(aes(reorder(vars, purity), purity)) + 
  geom_bar(stat = "identity", fill = "white", color = "black") + 
  coord_flip() + 
  theme_bw() +
  labs(x = "Predictor",
       y = "Importance (%)")

# GBM
pl_model_gbm <- plot(model_gbm)

# elastic net
pl_model_enet <- plot(model_enet)

# CART
pl_model_cart <- plot(model_cart)

pl_model_cart_tree <- fancyRpartPlot(model_cart$finalModel)

# test model performance --------------------------------------------------

data_test <- na.omit(data_test)

test_rmses <- list()
iterator <- 0
for (model in models) {
  iterator <- iterator + 1
  model_name <- names(models)[iterator]
  pred <- predict(model, newdata = data_test)
  test_rmses[model_name] <- RMSE(pred, data_test$price_ln)
}
test_rmses

test_results <- data.frame(
  model = names(test_rmses),
  rmse = round(as.numeric(as.vector(bind_rows(test_rmses)[1,])), 4)
)
write_csv(test_results, "out/test_results.csv")

resample_profile <- resamples(
  list("linear" = model_reg,
       "elastic" = model_enet, 
       "random-forest" = model_rf,
       "GBM" = model_gbm,
       "cart" = model_cart)
)

pl_resample <- dotplot(resample_profile)

resample_profile

# export plots ------------------------------------------------------------

# export plots
plot_ind <- grep("^pl_", ls())
export_obj <- ls()[plot_ind]

for (plot in export_obj) {
  tryCatch({ggsave(paste0("out/", plot, ".png"), eval(parse(text = plot)), )},
           error = function(e) {print(paste0('error with plot: ', plot))},
           finally = {})
}

plot_out <- function(plot_obj) {
  name <- deparse(substitute(plot_obj))
  png(file=paste0("out/", name, ".png"),
      width=800, height=720) 
  plot(plot_obj)
  dev.off()
}

plot_out(pl_model_cart)
plot_out(pl_model_enet)
plot_out(pl_model_gbm)
plot_out(pl_model_rf)
plot_out(pl_resample)
