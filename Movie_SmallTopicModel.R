#
# Script: Movie_SmallTopicModel.R
#
# R script for for analyzing movie similarities using only selected movies and keywords
# meant for illustration purposes only
#
# Requires the following files:
#  opus_movies.csv              Movie characteristics of wide releases from 2006-2014
#  opus_movielens_tags.csv      Keywords that describe the movie from MovieLens
#
# The data included for this exercise is for internal use only and
# may not be posted or distributed further.
# Specifically the file opus_movies.txt 
# is provided by The Numbers (http://www.the-numbers.com),
# powered by OpusData (http://www.opusdata.com).
# The opus_movielens_tags.txt is available from Movielens
# which is located at http://grouplens.org/datasets/movielens/latest
#




##################### setup environment  ######################

# setup libraries
if (!require(lattice)) {install.packages("lattice"); library(lattice)}
if (!require(NLP)) {install.packages("NLP"); library(NLP)}
if (!require(topicmodels)) {install.packages("topicmodels"); library(topicmodels)}
if (!require(tm)) {install.packages("tm"); library(tm)}
if (!require(slam)) {install.packages("slam"); library(slam)}




##################### input the data  ######################

## read in the data

# in RStudio select Menu Bar --> Session --> Set Working Directory --> To Source File Directory
# or automatically set working directory to be that of the script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # only works in Rstudio scripts
# alternatively set the working directory manually
#setwd("~/Documents/class/marketing analytics/cases/movies")  # !! edit and uncomment this line, if needed !!

# read in movie datasets
movies=read.delim("opus_movies.txt",header=T)  # the Opus movie data
tags=read.delim("opus_movielens_tags.txt",header=T)  # just the tags from movielens


## make modifications to the dataset

# change data formats
tags$odid=as.factor(tags$odid)


## transform the terms into a structure that can be used for topic modeling

# use this definition of mterms for movielens tags
# put data in sparse matrix form using simple_triplet_matrix as needed by LDA
mterms=simple_triplet_matrix(i=as.integer(tags$odid),j=as.integer(tags$tag),v=tags$count,
                             dimnames=list(levels(tags$odid),levels(tags$tag)))
# let's create a list of a smaller list of movies produced by paramount
movielist=movies$odid[movies$production_company1=="Paramount Pictures"]
# let's create a short list of terms to save
shorttermlist=c("action","comic book","animation")
# keep only the subset of a few terms
mterms=mterms[rownames(mterms) %in% movielist,shorttermlist]
# also delete any movies that do not have any terms
mterms=mterms[apply(mterms,1,sum)>0,]
# let's update the list of movies and their names since some might have just been deleted
movielist=rownames(mterms)
movienames=as.character(movies$display_name[movies$odid %in% rownames(mterms)])
# let's print out our matrix
as.matrix(mterms)
# let's lookup the movie names
movies[movies$odid %in% rownames(mterms),c("odid","display_name")]
# compute totals for mterms
lmterms=apply(mterms,1,sum)   # compute the sum of each of the rows (# of terms per movie)
lwterms=apply(mterms,2,sum)   # compute the sum of each of the columns (# of times word used)

# prepare a subset of movies with just the movies in our list
umovies=movies[movies$odid %in% as.integer(movielist),]   # create a subset of the movies that have terms




##################### for reference compute kmeans cluster  ######################

# estimate kmeans with two topics
(grpKmeans=kmeans(mterms,centers=2))

# summarize the centroids
grpKcenter=t(grpKmeans$centers)
parallelplot(t(grpKcenter))

# print a table with the movies assigned to each cluster
for (i in 1:2) {
  print(paste("* * * Movies in Cluster #",i," * * *"))
  print(movienames[grpKmeans$cluster==i])
}



##################### estimate an LDA topic model using keywords  ######################

## our first step is to estimate the topic model using LDA

# setup the parameters for LDA control vector
burnin=1000     # number of initial iterations to discard for Gibbs sampler (for slow processors use 500)
iter=5000       # number of iterations to use for estimation  (for slow processors use 1000)
thin=50         # only save every 50th iteration to save on storage
seed=list(203,5,63,101,765)  # random number generator seeds
nstart=5        # number of repeated random starts
best=TRUE       # only return the model with maximum posterior likelihood

# estimate a series of LDA models (each run can take a few minutes depending upon your processor)
ClusterOUT = LDA(mterms,2,method="Gibbs",control=list(nstart=nstart,seed=seed,best=best,burnin=burnin,iter=iter,thin=thin))


## now that we have saved the LDA results to our ClusterOUT object we want to
## extract the topic information and look at them

# probability of topic assignments (each movie has its own unique profile)
# rows are movies and columns are topics
ClustAssign = ClusterOUT@gamma   # this is a matrix with the row as the movie and column as the topic
rownames(ClustAssign)=movienames  # set the movie titles as the row names
dim(ClustAssign)  # check the dimension of the cluster (movies X topics)
head(ClustAssign,n=10)   # show the actual topic probabilities associated with the first 10 movies

# matrix with probabilities of each term per topic
ClustTopics = exp(ClusterOUT@beta)     # notice that we use "@" to access elements in the object and not "$" since this is an S4 object
colnames(ClustTopics)=colnames(mterms) # the columns are the terms
dim(ClustTopics)                       # check dimensions of the topics
print(ClustTopics)                     # print out clusters (topics in rows and terms in columns)


## let's work on understanding the cluster based upon the movies

# visualize the distribution of topics across the movies
boxplot(ClustAssign,xlab="Topic",ylab="Probability of Topic across Movies")

# print a table with the movies assigned to each cluster
ClustBest = apply(ClustAssign,1,which.max)  # determine the best guess of a cluster, a vector with best guess
for (i in 1:2) {
  print(paste("* * * Movies in Cluster #",i," * * *"))
  print(movienames[ClustBest==i])
}


## another way to understand the topics is through their associations with the keywords

# show the terms and associated topics
parallelplot(ClustTopics,main="Topic associated with selected Terms")

# show the topics associated with a selected movie
imovie=1
barplot(ClustAssign[imovie,],names.arg=1:ncol(ClustAssign),main=paste("Topics Associated with selected movie",umovies$display_name[imovie]))


## last we can use our model to compute a best guess

# determine the best guess for each movie/term combination
ClustGuess=(ClustAssign%*%ClustTopics)*lmterms

# we can compare the predictions for a selected movie
imovie=1
mcompare=cbind(ClustGuess[imovie,],as.vector(mterms[imovie,]))
print(mcompare)

# or we can print the predictions for all movies
as.matrix(cbind(ClustGuess,mterms))

# compare kmeans solutions with the topic model
# remember that kmeans assignments are deterministic, while topic models are probabilistic
# so this cross tab only considers the matches between the most likely
xtabs(~grpKmeans$cluster+ClustBest)

