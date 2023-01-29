# 
# Jan 2023

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
library(textmineR)
library(reshape2)
library(ggplot2)
library(wordcloud)
library(ggwordcloud)
library(lda)
library(ldatuning)
library(kableExtra)
library(hunspell)
library(DT)
library(pals)
library(SnowballC)
library(flextable)
library(parallel)

setwd("G:/My Drive/U/2022_WORK_UNISA/WORK_UNISA/Katapult")
removeSpecialChars <- function(x) gsub("[^a-zA-Z0-9 ]","",x)

organization <- ("docs/organization_train_data")
corpus_org <- tm::Corpus(DirSource(directory = organization, pattern = "*.txt"))
corpus_org <- tm_map(corpus_org, removePunctuation)
corpus_org <- tm_map(corpus_org, removeSpecialChars)
corpus_org <- tm_map(corpus_org, content_transformer(tolower))
corpus_org <- tm_map(corpus_org, removeNumbers)
corpus_org <- tm_map(corpus_org, removeWords, stopwords("english"))
corpus_org <- tm_map(corpus_org, stripWhitespace)
corpus_org <- tm_map(corpus_org, stemDocument)

# Build document matrix
dtm_org <- DocumentTermMatrix(corpus_org, control = list(weighting = weightTf))
inspect(dtm_org)

# Drop common words that occur in more than 95% of all docs
# This is to give a collection that is common enough (occurred in at least 2 docs) 
# and unique enough (not shared by all docs)
word_freq <- findFreqTerms(dtm_org, highfreq = nrow(dtm_org)*0.95)
length(word_freq)

# apply freq to dtm
dtm_org <- dtm_org[, word_freq]
inspect(dtm_org)


# find optimal K
# TIME WARNING: 1 min(s) for 17 documents (2100 terms)
optimal.topics <- FindTopicsNumber(
  dtm_org,
  topics = 2:40,
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"), 
  method = "Gibbs",
  control = list(seed = 420),
  mc.cores = 2L,
  verbose = FALSE
)
optimal.topics
FindTopicsNumber_plot(optimal.topics)

# extract K
cond1 <- min(optimal.topics$Arun2010 - optimal.topics$CaoJuan2009)
# cond2 <- ratio to length(corpus_org)
# K <- meet cond1 and cond2

K <- 14


# model using package topcimodelsR
lda_org2 <- topicmodelsR::LDA(dtm_org
                , k = K
                , method = 'Gibbs'
                , control = list(seed = 420
                                 , alpha = 0.5
                                 , iter = 2000
                                 , burnin = 200
                                 , best = TRUE)
                )


# orgnization
# output 1 Topic - term
terms(lda_org2, 8) %>% as.data.frame()


org_topics <- tidy(lda_org2, matrix("beta"))
org_topics

org_top_terms <- org_topics %>% group_by(topic) %>%
  slice_max(beta, n=10) %>% ungroup() %>% arrange(topic, -beta)
org_top_terms

org_top_terms %>% mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = 'free') +
  scale_y_reordered() + 
  ggtitle('Word-topic probabilities'
          , subtitle = 'slice_max top 10 terms for each topic from the full beta list')


## output 2 Document - topic
## lda_org$theta %>% as.data.frame() %>% set_names(paste('Topic', 1:10))

org_gamma <- tidy(lda_org2, matrix = "gamma")
org_gamma


# policy
policy <- ("docs/policy")
corpus_pol <- VCorpus(DirSource(directory = policy, pattern = "*.txt"))
corpus_pol <- tm_map(corpus_pol, removePunctuation)
corpus_pol <- tm_map(corpus_pol, content_transformer(tolower))
corpus_pol <- tm_map(corpus_pol, removeNumbers)
corpus_pol <- tm_map(corpus_pol, removeWords, stopwords("english"))
corpus_pol <- tm_map(corpus_pol, stripWhitespace)
corpus_pol <- tm_map(corpus_pol, stemDocument)



# prediction
# loop through each policy doc in the policy corpus
results <- list()
for (i in seq(1:length(corpus_pol))) {
  # get new doc dtm
  dtm_pol <- DocumentTermMatrix(corpus_pol[i], control = list(weighting = weightTf))
  # get posterior for the new doc
  posterior_query_docs <- topicmodels::posterior(lda_org2, dtm_pol)
  # 
  policy_ready <- tidy(dtm_pol)$document[1]
  policy_gamma <- as_tibble(posterior_query_docs$topics) %>%
    tidyr::pivot_longer(cols = 1:10, names_to = "topic", values_to = "gamma") %>% 
    mutate(document = policy_ready) %>%
    mutate(topic = as.integer(topic)) %>%
    relocate(document, .before = topic)
  
  # Add the policy gamma results to the website gamma results
  merged_policy_org <- rbind(policy_gamma, org_gamma)
  merged_policy_org <- arrange(merged_policy_org, topic)
  
  # Create a matrix for use in Cosine Similarity
  matrix_ <- pivot_wider(merged_policy_org
                         , names_from = document
                         , values_from = gamma) %>% mutate(topic = NULL)
  matrix_ <- as.matrix(matrix_)
  
  # Run Cosine Similarity and Sort results
  cosine_results <- lsa::cosine(matrix_)
  
  # Display results
  cosine_results_sorted <- sort(cosine_results[, policy_ready],dec=T)
  
  cosine_results_sorted <- cosine_results_sorted %>% data.frame(similarity = .)
  results[[policy_ready]]=rownames_to_column(cosine_results_sorted
                                             , var="organization")[2:(length(org_corpus)+2),]
}

# organise model results
results_df <- results %>% as.data.frame()
row.names(results_df) <- NULL
rank_by_model <- results_df[, sapply(results_df, is.character)] %>% 
  rownames_to_column(var = 'rank.model') %>% 
  lapply(function(x){gsub(".txt","",x)}) %>% 
  as_tibble()
colnames(rank_by_model) <- sapply(strsplit(colnames(rank_by_model), '_'), '[', 1)
rank_by_model$rank.model <- as.numeric(rank_by_model$rank.model)
rank_by_model
 

col1 <- rank_by_model[, c('rank.model', 'ID0')] %>% arrange(ID0)
col2 <- filter(benchmark, path=='ID0') %>% arrange(Organization)
rank.corr <- cor.test(col1, col2, method = 'spearman')

## ----------------------------------------------
## ----------------------------------------------


# Grid Search with learning curve on similarity gap


setwd("G:/My Drive/U/2022_WORK_UNISA/WORK_UNISA/Katapult")
removeSpecialChars <- function(x) gsub("[^a-zA-Z0-9 ]","",x)


# 1. organization
# get corpus
organization <- ("docs/organization/organization_train_data")
org_corpus <- tm::Corpus(DirSource(directory = organization, pattern = "*.txt"))
org_corpus <- tm_map(org_corpus, removePunctuation)
org_corpus <- tm_map(org_corpus, removeSpecialChars)
org_corpus <- tm_map(org_corpus, content_transformer(tolower))
org_corpus <- tm_map(org_corpus, removeNumbers)
org_corpus <- tm_map(org_corpus, removeWords, stopwords("english"))
org_corpus <- tm_map(org_corpus, stripWhitespace)
org_corpus <- tm_map(org_corpus, stemDocument)

# get dtm
org_dtm <- DocumentTermMatrix(org_corpus, control = list(weighting = weightTf))

# Optinal: 
# drop common words that occur in more than 95% of all docs
# This is to give a collection that is common enough (occurred in at least 2 docs) 
# and unique enough (not shared by all docs)
word_freq <- findFreqTerms(org_dtm, highfreq = nrow(org_dtm)*0.95)
# length(word_freq)
# apply freq to dtm
org_dtm_communique <- org_dtm[, word_freq]
# inspect(org_dtm)


# 2. policy
# get corpus
policy <- ("docs/policy")
policy_corpus <- Corpus(DirSource(directory = policy, pattern = "*.txt"))
policy_corpus <- tm_map(policy_corpus, removePunctuation)
policy_corpus <- tm_map(policy_corpus, removeSpecialChars)
policy_corpus <- tm_map(policy_corpus, content_transformer(tolower))
policy_corpus <- tm_map(policy_corpus, removeNumbers)
policy_corpus <- tm_map(policy_corpus, removeWords, stopwords("english"))
policy_corpus <- tm_map(policy_corpus, stripWhitespace)
policy_corpus <- tm_map(policy_corpus, stemDocument)


# 3. load benchmark
bm_list <- list.files(path = "docs/benchmark"
                         , pattern = "*.csv"
                         , full.names = T) 

benchmark <- read_csv(bm_list, id='path')
benchmark$path <- gsub('(docs/benchmark/)|.csv', '', benchmark$path)
# benchmark$path <- sub(".csv", "", benchmark$path)
benchmark

# 4. wrap a function for 1 LDA model build plus 1 iteration of all query docs
one_iter <- function(dtm, query_corpus, K, a, b, n_iter=2000) {
  # 
  # this function first build a LDA model on organization dtm, then for each policy doc
  # get its dtm and posterior off the LDA model, 
  # dtm: organization dtm, not policy
  # query_corpus: the corpus that wants to infer the model to get posterior topic distribution
  # K: number of topics
  # a: initial alpha value
  # b: initial beta value
  # iter: number of iterations, default 2000
  
  print(paste('INFO: a=',a,' beta=',b))
  
  # fit a model
  m <- topicmodels::LDA(dtm # this is organization dtm
                        , k = K
                        , method = 'Gibbs'
                        , estimate.alpha = TRUE # default TRUE
                        , beta = b # initial value
                        , control = list(seed = 420
                                         , alpha = a # initial value
                                         , estimate.beta = TRUE # default TRUE
                                         , iter = n_iter # default 2000
                                         , thin = ceiling(n_iter/10) # omit in-between Gibbs iterations
                                         , burnin = ceiling(n_iter/10) # omit beginning Gibbs iterations
                                         , best = TRUE # return max (posterior) likelihood
                                         ))
  # extract org gamma
  org_gamma <- tidy(m, matrix = "gamma")
  
  results <- list()
  
  # loop through each policy doc
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
    # matrix_ <- as.matrix(matrix_)
    
    # cal cosine similarity and sort results
    cosine_results <- lsa::cosine(matrix_)
    cosine_results_sorted <- sort(cosine_results[, policy_name],dec=T) %>% 
      data.frame(similarity = .)
    # cosine_results_sorted <- cosine_results_sorted %>% data.frame(similarity = .)
    
    # save top5 to list (skip 1st row which is "1")
    print(paste('INFO: processing ', policy_name))
    results[[policy_name]]=rownames_to_column(cosine_results_sorted
                                               , var="organization")[2:(K+1),]
    
    # get the gap score (Spearman's rank correlation test between bm and current rank results)
    cor_test <- cor.test(benchmark, results[[policy_name]])
    
    # return(list(alpha=a, beta=b, model=m, similarity=results, gap=gap_score))
  }
  print('INFO: finished one iteration of the query corpus')
  return(list(alpha=a, beta=b, model=m, similarity=results, gap=gap_score))
}

# 4. grid search a and b
# each a/b pair will run one_iter function once
alpha_range <- 0.1*c(1:5) # 10, 20, 30, 35
# beta_range <- 0.1 *c(0.1, 1, 5) # 10
beta_range <- 0.1
df <- data.frame()
K <- 14
n_iter <- 2000

similarity <- list()
i <- 1
for(a in alpha_range){
  for(b in beta_range){
    iter <- one_iter(dtm=org_dtm, query_corpus = policy_corpus, K=K, a=a, b=b, n_iter=n_iter)
    similarity[[i]] <- c(alpha=a,beta=b,sim=iter$similarity)
    i <- i+1
    gap_score <- iter$gap
    df <- rbind(df, as.data.frame(list(alpha=a,beta=b,gap=gap_score)))
  }
}
  
df
similarity

ggplot2::ggplot(df)

# results <- one_iter(a, b)
  
