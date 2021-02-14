# define webhook function
send_message <- function(webhookurl = url, my_text = text) {
  library(jsonlite)
  library(rvest)
  
  headers = c(
    `Content-type` = 'application/json'
  )
  tryCatch({
    data= toJSON(list("content"= my_text), auto_unbox = T)
    res <- httr::POST(url = webhookurl, httr::add_headers(.headers=headers), body = data)
  },
  warning = function(w) {
    print('warning!')
  }, error = function(e) {
    print('error!')
  }, finally = {
  }
  )
}


# read in credentials
creds <- read.delim('/mnt/DATA/Projects/BA-20-21/danalysis-3/data/creds.txt', sep = ',') # change the path to your creds.txt file
url <- as.character(creds[creds$key == 'webhookurl', 2])
user <- as.character(creds[creds$key == 'user', 2])

text <- paste0('hello ', user, ' your model has finished training!')

