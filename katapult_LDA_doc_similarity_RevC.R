# 
# Jan 2023

# clear workspace
rm(list=ls())
gc()

if(!require('kableExtra')) install.packages('kableExtra')
if(!require('DT')) install.packages('DT')
if(!require('wordcloud')) install.packages('wordcloud')
if(!require('pals')) install.packages('pals')
# if(!require('lda')) install.packages('lda')
if(!require('flextable')) install.packages('flextable')
if(!require('textmineR')) install.packages('textmineR')
if(!require('ggwordcloud')) install.packages('ggwordcloud')
library(corpus)
library(tidyverse)
library(tidytext)
library(tm)
library(topicmodels)
# library(textmineR)
library(reshape2)
library(ggplot2)
# library(lda)
library(ldatuning)
library(kableExtra)
library(hunspell)
library(DT)
library(pals)
library(flextable)
library(parallel)


# wrap a function to get Spearman rank correlation coefficient
get_rank_correlation <- function(results, benchmark, doc_id){
  
  print(paste('doc_id:',doc_id))
  
  rho <- list()
  p <- list()
  
  # organise model results
  results_df <- results %>% as.data.frame()
  row.names(results_df) <- NULL
  rank_by_model <- results_df %>% select_if(is.character) %>% 
    rownames_to_column(var = 'rank.model') %>% 
    lapply(function(x){gsub(".txt","",x)}) %>% 
    as_tibble()
  colnames(rank_by_model) <- sapply(strsplit(colnames(rank_by_model), '_'), '[', 1)
  rank_by_model$rank.model <- as.numeric(rank_by_model$rank.model)
  
  # get rho
  col1 <- rank_by_model[, c('rank.model', doc_id)]
  col1 <- arrange(col1, across(names(col1)[2]))
  col2 <- filter(benchmark, path==doc_id) %>% arrange(organization)
  rank.corr <- cor.test(col1$rank.model, col2$rank.bm, method = 'spearman', exact = FALSE)
  rho[[doc_id]] <- rank.corr$estimate[[1]]
  p[[doc_id]] <- rank.corr$p.value
  
  return(list(rho=rho, p=p))
}


# wrap a function to fit 1 LDA model plus 1 iteration of all query docs and get rank comparison result
one_iter <- function(dtm, query_corpus, K, a, n_iter=2000) {
  
  # this function first build a LDA model on organization dtm, then for each policy doc
  # get its dtm and posterior off the LDA model, merge with organization topic distribution
  # calculate cosine distance from the policy doc to each of the organizations
  # sort the rank, compare with benchmark rank order
  # by running a Spearman's rank correlation coefficient test (allowing ties)
  #
  # dtm: organization dtm, not policy
  # query_corpus: the corpus that wants to infer the model to get posterior topic distribution
  # K: number of topics
  # a: initial alpha value
  # n_iter: number of iterations, default 2000
  #
  # note: not setting beta, but using estimate.beta=TRUE in the control list
  
  print(paste('INFO: a=',a,'K=',K))
  
  # fit a model
  m <- topicmodels::LDA(dtm # this is organization dtm
                        , k = K
                        , method = 'Gibbs'
                        # , beta = b # initial value
                        , control = list(seed = 420
                                         , alpha = a # initial value
                                         # , estimate.alpha = TRUE # default TRUE
                                         , estimate.beta = TRUE # default TRUE
                                         , iter = n_iter # default 2000
                                         , thin = ceiling(n_iter/10) # omit in-between Gibbs iterations
                                         , burnin = ceiling(n_iter/10) # omit beginning Gibbs iterations
                                         , best = TRUE # return max (posterior) likelihood
                        ))
  # extract org gamma
  org_gamma <- tidy(m, matrix = "gamma")
  
  results <- list()
  rho <- list()
  
  # for given alpha and K, loop through each policy doc
  for (i in seq(1:length(query_corpus))) {
    # get policy doc dtm
    pol_dtm <- DocumentTermMatrix(query_corpus[i], control = list(weighting = weightTf))
    
    # infer the fitted model to get posterior for policy doc
    policy_posterior <- topicmodels::posterior(m, pol_dtm)
    
    # get this policy name 
    policy_name <- tidy(pol_dtm)$document[1]
    
    # extract policy gamma
    policy_gamma <- as_tibble(policy_posterior$topics) %>%
      tidyr::pivot_longer(cols = 1:K, names_to = "topic", values_to = "gamma") %>% 
      mutate(document = policy_name) %>%
      mutate(topic = as.integer(topic)) %>%
      relocate(document, .before = topic)
    
    # merge policy gamma results with org gamma
    merged_policy_org <- rbind(policy_gamma, org_gamma)
    merged_policy_org <- arrange(merged_policy_org, topic)
    
    # create a matrix for use in Cosine Similarity
    matrix_ <- pivot_wider(merged_policy_org
                           , names_from = document
                           , values_from = gamma) %>% 
      mutate(topic = NULL) %>% 
      as.matrix()
    
    # cal cosine similarity and sort results
    cosine_results <- lsa::cosine(matrix_)
    cosine_results_sorted <- sort(cosine_results[, policy_name],dec=T) %>% 
      data.frame(similarity = .)
    
    # save to list (skip 1st row which is "1")
    print(paste('INFO: processing', policy_name))
    results[[policy_name]]=rownames_to_column(cosine_results_sorted
                                              , var="organization")[2:(length(org_corpus)+1),]
    
    # rank_corr_coef <- 0.2
    
    # get rank correlation 
    # Spearman's rank correlation test between benchmark and current model's rank results
    doc_ID <- strsplit(policy_name, '_')[[1]][1]
    print(paste("INFO: document id:", doc_ID))
    rank_corr_coef <- get_rank_correlation(results=results
                                           , benchmark = benchmark
                                           , doc_id = doc_ID)
    rho <- rank_corr_coef$rho
    
  }
  print('INFO: finished one iteration of the query corpus')
  corr_coef <- mean(unlist(rho))
  return(list(alpha=a, K=K, model=m, similarity=results, rho=rho, corr_coef=corr_coef))
}





## -------------------------------------------

# Grid Search with learning curve on similarity gap


setwd("G:/My Drive/U/2022_WORK_UNISA/WORK_UNISA/Katapult")
removeSpecialChars <- function(x) gsub("[^a-zA-Z0-9 ]","",x)


# 1. organization
## get corpus
organization <- ("docs/organization/organization_train_data")
org_corpus <- tm::Corpus(DirSource(directory = organization, pattern = "*.txt"))
org_corpus <- tm_map(org_corpus, removePunctuation)
org_corpus <- tm_map(org_corpus, removeSpecialChars)
org_corpus <- tm_map(org_corpus, content_transformer(tolower))
org_corpus <- tm_map(org_corpus, removeNumbers)
org_corpus <- tm_map(org_corpus, removeWords, stopwords("english"))
org_corpus <- tm_map(org_corpus, stripWhitespace)
org_corpus <- tm_map(org_corpus, stemDocument)

## get dtm
org_dtm <- DocumentTermMatrix(org_corpus, control = list(weighting = weightTf))

## Optinal: 
## drop common words that occur in more than 95% of all docs
## This is to give a collection that is common enough (occurred in at least 2 docs) 
## and unique enough (not shared by all docs)
word_freq <- findFreqTerms(org_dtm, highfreq = nrow(org_dtm)*0.95)
## length(word_freq)
## apply freq to dtm
org_dtm_communique <- org_dtm[, word_freq]
## inspect(org_dtm)


# 2. policy
## get corpus
policy <- ("docs/policy")
policy_corpus <- Corpus(DirSource(directory = policy, pattern = "*.txt"))
policy_corpus <- tm_map(policy_corpus, removePunctuation)
policy_corpus <- tm_map(policy_corpus, removeSpecialChars)
policy_corpus <- tm_map(policy_corpus, content_transformer(tolower))
policy_corpus <- tm_map(policy_corpus, removeNumbers)
policy_corpus <- tm_map(policy_corpus, removeWords, stopwords("english"))
policy_corpus <- tm_map(policy_corpus, stripWhitespace)
policy_corpus <- tm_map(policy_corpus, stemDocument)


# 3. benchmark
bm_list <- list.files(path = "docs/benchmark"
                      , pattern = "*.csv"
                      , full.names = T) 

benchmark <- read_csv(bm_list, id='path')
benchmark$path <- gsub('(docs/benchmark/)|.csv', '', benchmark$path)
# benchmark


## ---------------------------------------------

# Grid search

n_iter <- 2000
alpha_range <- 0.1*c(1:5, 10, 20, 30, 35)
k_range <- c(12,14,16)
df <- data.frame()
similarity <- list()
i <- 1

for(a in alpha_range){
  for(K in k_range){
    # for given alpha K pair, run one iteration
    iter <- one_iter(dtm=org_dtm, query_corpus = policy_corpus, K=K, a=a, n_iter=n_iter)
    
    # get results from one iteration
    similarity[[i]] <- c(alpha=a,K=K,sim=iter$similarity, rho=iter$rho)
    i <- i+1
    corr_coef <- iter$corr_coef
    
    # save to dataframe
    df <- rbind(df, as.data.frame(list(alpha=a,K=K,corr_coef=corr_coef)))
  }
}

# visualise grid search results
df$K <- as.factor(df$K)
plt <- ggplot(df, aes(x=alpha, y=corr_coef, group=K)) +
  geom_line(aes(linetype=K)) +
  geom_point(aes(shape=K), size=2) + 
  ggtitle('Average rank correlation coeffiient by alpha and number of topics')
plt

# extract optimal hyperp values
best.row <- df[which.max(df$corr_coef),]
best.alpha <- best.row[[1]]
best.K <- best.row[[2]]

# print
best.alpha
best.K
df

## ------- END Grid Search
