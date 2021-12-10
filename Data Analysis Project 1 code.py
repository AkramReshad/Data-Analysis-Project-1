#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: akram
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import sqrt
import pandas as pd
from scipy.stats import ttest_ind, shapiro, linregress, ttest_ind_from_stats, ks_2samp, f_oneway


##### GATHERING DATA AND SEPARATING INTO SMALLER DATAFRAMES ####

movies_df = pd.read_csv (r'/Users/akram/Documents/Graduate School/NYU Fall 2021/Intro to Data Science/Data Analysis Project 1/movieReplicationSet.csv')
movies_df = movies_df.rename({'Rambo: First Blood Part II' : 'Rambo: First Blood Part II (1985)'}, axis = 1) #The only movie without a year
rows,cols = movies_df.shape
movie_ratings = movies_df.iloc[:,:400]
sensation_assessment = movies_df.iloc[:,401:420]
personality_assessment = movies_df.iloc[:,420:464]
movie_experience = movies_df.iloc[:,465:474]
gender = movies_df.iloc[:,474]
only_child = movies_df.iloc[:,475]
enjoy_movies_alone= movies_df.iloc[:,476]

mean_rating = movie_ratings.mean() # the mean rating of each column



############### QUESTION 1 ###############

num_of_ratings = movie_ratings.count() 

mean_rating = movie_ratings.mean() # the mean rating of each column
corr_between_num_of_Rating_and_mean = num_of_ratings.corr(mean_rating) # the Correlation between # of ratings and average rating
X = num_of_ratings.values.reshape(-1,1)
Y = mean_rating.values.reshape(-1,1)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X,Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.figure()
plt.scatter(X,Y)
plt.plot(X,Y_pred, color = 'red')
plt.xlabel("Number Of Ratings")
plt.ylabel("Mean of Ratings")
plt.title('Mean of Ratings vs Number of Rating')
plt.show()
median_num_of_ratings = num_of_ratings.median() # median # of rating

greater_than_median_rating_data = movie_ratings.loc[:, num_of_ratings > median_num_of_ratings] # getting the movies with number of ratings > 197.5
less_than_median_rating_data =  movie_ratings.loc[:, num_of_ratings <= median_num_of_ratings] # getting the movies with number of ratings < 197.5
greater_than_average = greater_than_median_rating_data.stack().mean()
greater_than_std = greater_than_median_rating_data.stack().std()
less_than_average = less_than_median_rating_data.stack().mean()
less_than_std = less_than_median_rating_data.stack().std()

#Since our Variance is different we will use the Welch test with N1 = 90214 N2 = 22000
N1 = 90214
N2 = 22000
SE1 = greater_than_std/sqrt(N1)
SE2 = less_than_std/sqrt(N2)
t_cal = (greater_than_average - less_than_average)/sqrt(SE1**2 + SE2**2)
df =(SE1**2 + SE2**2)**2/(SE1**4/(N1-1) + SE2**4/(N2-1))

t_1, p_1 = ttest_ind(greater_than_median_rating_data.stack(),less_than_median_rating_data.stack(), equal_var = False)
print('For Question 1:')
print('Our t-statistic', t_1, 'with p value', p_1)
print('We reject our Null Hypothesis')
print("")




############### QUESTION 2 ###############
years_list = []

for name in movie_ratings.columns:
    years_list.append(int(name[-5:-1]))

years_df = pd.DataFrame(years_list, columns=['Year'])
years_df.index = movie_ratings.columns

#movie_rating_with_year = movie_ratings
#movie_rating_with_year.loc['Year'] = years_list
#movie_rating_with_year = movie_rating_with_year.sort_values(by = 'Year', axis = 1)
#print(movie_rating_with_year)

median_year = int(years_df.median())

older_than_median_year_data = movie_ratings.loc[:, years_df['Year'] > median_year] #getting the movies with year > 1999
younger_than_median_year_data =  movie_ratings.loc[:, years_df['Year'] <= median_year] # getting the movies with year <= 1999
older_than_average = older_than_median_year_data.stack().mean()
older_than_std = older_than_median_year_data.stack().std()
younger_than_average = younger_than_median_year_data.stack().mean()
younger_than_std = younger_than_median_year_data.stack().std()
N1A = len(older_than_median_year_data.stack())
N2A = len(younger_than_median_year_data.stack())
SE1A = older_than_std/sqrt(N1A)
SE2A = younger_than_std/sqrt(N2A)

t_cal = (older_than_average - younger_than_average)/((sqrt(((N1A-1)*older_than_std**2 + (N2A-1)*younger_than_std**2)/(N1A+N2A-2))) * sqrt(1/N1A + 1/N2A))
t_2,p_2 = ttest_ind(older_than_median_year_data.stack(), younger_than_median_year_data.stack(), equal_var = True)
#t,p = ttest_ind_from_stats(older_than_average, older_than_std, N1A, younger_than_average, younger_than_std, N2A)
print('For Question 2:')
print('The t statistic = ', t_2, 'with p value = ', p_2)
print('We reject our Null Hypothesis')
years_df = years_df.assign(Means = mean_rating)

X = years_df['Year'].values.reshape(-1,1)
#X= years_index.values.reshape(-1,1)
Y = years_df['Means'].values.reshape(-1,1)
corr_years_mean =  years_df['Year'].corr(years_df['Means'])
linear_regressor = LinearRegression()  # create object for the class
reg = linear_regressor.fit(X,Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
print(linregress(years_df['Year'].values,years_df['Means'].values)) # returns slope, intercept, and rvalue
plt.figure()
plt.scatter(X,Y)
plt.plot(X,Y_pred, color = 'black')
plt.xlabel("Year")
plt.ylabel("Mean of Ratings")
plt.title('Mean of Ratings vs Year')
plt.show()
print('')



############### QUESTION 3 ###############

Shrek_and_gender = pd.concat([movie_ratings['Shrek (2001)'],gender], axis = 1)
Shrek_and_gender_Ns = Shrek_and_gender.groupby(by = 'Gender identity (1 = female; 2 = male; 3 = self-described)', axis = 0 ).count()
Shrek_and_gender_means = Shrek_and_gender.groupby(by = 'Gender identity (1 = female; 2 = male; 3 = self-described)', axis = 0 ).mean()
Shrek_and_gender_stds = Shrek_and_gender.groupby(by = 'Gender identity (1 = female; 2 = male; 3 = self-described)', axis = 0 ).std()
females = [Shrek_and_gender_means.iloc[0,0], Shrek_and_gender_stds.iloc[0,0], Shrek_and_gender_Ns.iloc[0,0]]
males = [Shrek_and_gender_means.iloc[1,0], Shrek_and_gender_stds.iloc[1,0], Shrek_and_gender_Ns.iloc[1,0]]

t_3,p_3 = ttest_ind_from_stats(females[0], females[1], females[2], males[0], males[1], males[2])
print("Question 3:")
print('Female mean, std, and count', females)
print('Male mean, std, and count', males)
print('Our test statistic is', t_3, 'with p-value', p_3)
print('We fail to reject our Null Hypothesis')
print('')


############### QUESTION 4 ###############

movie_ratings_and_gender = pd.concat([movie_ratings, gender], axis = 1)
movie_ratings_and_gender_Ns= movie_ratings_and_gender.groupby( by = 'Gender identity (1 = female; 2 = male; 3 = self-described)', axis = 0).count()
movie_ratings_and_gender_means= movie_ratings_and_gender.groupby( by = 'Gender identity (1 = female; 2 = male; 3 = self-described)', axis = 0).mean()
movie_ratings_and_gender_stds = movie_ratings_and_gender.groupby( by = 'Gender identity (1 = female; 2 = male; 3 = self-described)', axis = 0).std()
differences = []

for i in range(0,400):
    t_4,p_4 = ttest_ind_from_stats(movie_ratings_and_gender_means.iloc[0,i], movie_ratings_and_gender_stds.iloc[0,i], movie_ratings_and_gender_Ns.iloc[0,i],
                               movie_ratings_and_gender_means.iloc[1,i], movie_ratings_and_gender_stds.iloc[1,i], movie_ratings_and_gender_Ns.iloc[1,i], 
                               equal_var=False)
    differences.append( p_4 < .0025 )
print('Question 4:')
print('The proportion of movies that are rated differently by male and female viewers', (sum(differences)/400)*100, '%')
print('')




############### QUESTION 5 ###############

lk_and_siblings = pd.concat([movie_ratings['The Lion King (1994)'], only_child], axis = 1)
lk_and_siblings = lk_and_siblings[lk_and_siblings['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'] != -1]
lk_and_siblings_Ns = lk_and_siblings.groupby(by = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', axis = 0).count()
lk_and_siblings_means = lk_and_siblings.groupby(by = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', axis = 0).mean()
lk_and_siblings_stds = lk_and_siblings.groupby(by = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', axis = 0).std()

t_5,p_5 = ttest_ind_from_stats(lk_and_siblings_means.iloc[0,0], lk_and_siblings_stds.iloc[0,0],lk_and_siblings_Ns.iloc[0,0],
                           lk_and_siblings_means.iloc[1,0], lk_and_siblings_stds.iloc[1,0], lk_and_siblings_Ns.iloc[1,0], 
                           equal_var=False)
print('For Question 5:')
print('Our Test statistic is', t_5, 'with p-value',p_5)
print('We fail to reject our Null Hypothesis')
print('')

############### QUESTION 6 ###############

movie_ratings_and_siblings = pd.concat([movie_ratings, only_child], axis = 1)
movie_ratings_and_siblings = movie_ratings_and_siblings[movie_ratings_and_siblings['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'] != -1]
movie_ratings_and_siblings_Ns = movie_ratings_and_siblings.groupby(by = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', axis = 0).count()
movie_ratings_and_siblings_means = movie_ratings_and_siblings.groupby(by = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', axis = 0).mean()
movie_ratings_and_siblings_stds = movie_ratings_and_siblings.groupby(by = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', axis = 0).std()
differences = []

for i in range(0,400):
    t_6,p_6 = ttest_ind_from_stats(movie_ratings_and_siblings_means.iloc[0,i], movie_ratings_and_siblings_stds.iloc[0,i], movie_ratings_and_siblings_Ns.iloc[0,i],
                               movie_ratings_and_siblings_means.iloc[1,i], movie_ratings_and_siblings_stds.iloc[1,i], movie_ratings_and_siblings_Ns.iloc[1,i], 
                               equal_var=False)
    #if p < .0025:
     #   print(movie_ratings_and_siblings.columns[i])
    differences.append( p_6 < .0025 )
print('For Question 6:')
print('The proportion of movies that exhibit an “only child effect”', (sum(differences)/400)*100, '%')
print('')


############### QUESTION 7 ###############

wow_and_social = pd.concat([movie_ratings['The Wolf of Wall Street (2013)'], enjoy_movies_alone], axis = 1) # Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)
wow_and_social = wow_and_social[wow_and_social['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] != -1]
wow_and_social_Ns = wow_and_social.groupby(by = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', axis = 0).count()
wow_and_social_means = wow_and_social.groupby(by = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', axis = 0).mean()
wow_and_social_stds = wow_and_social.groupby(by = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', axis = 0).std()

t_7,p_7 = ttest_ind_from_stats(wow_and_social_means.iloc[0,0], wow_and_social_stds.iloc[0,0],wow_and_social_Ns.iloc[0,0],
                           wow_and_social_means.iloc[1,0], wow_and_social_stds.iloc[1,0], wow_and_social_Ns.iloc[1,0], 
                           equal_var=False)
print('For Question 7:')
print('Our Test statistic is', t_7, 'with p-value',p_7)
print('We fail to reject our Null Hypothesis')
print('')


############### QUESTION 8 ###############

ratings_social = pd.concat([movie_ratings, enjoy_movies_alone], axis = 1)
ratings_social = ratings_social[ratings_social['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] != -1]
ratings_social_Ns = ratings_social.groupby(by = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', axis = 0).count()
ratings_social_means = ratings_social.groupby(by = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', axis = 0).mean()
ratings_social_stds = ratings_social.groupby(by = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', axis = 0).std()
differences = []
for i in range(0,400):
    t_8,p_8 = ttest_ind_from_stats(ratings_social_means.iloc[0,i], ratings_social_stds.iloc[0,i], ratings_social_Ns.iloc[0,i],
                              ratings_social_means.iloc[1,i], ratings_social_stds.iloc[1,i], ratings_social_Ns.iloc[1,i], 
                              equal_var=False)

    differences.append( p_8 < .0025 )
print('For Question 8:')
print('The proportion of movies that exhibit a “social watching” effect is', (sum(differences)/400)*100, '%')
print('')


############### QUESTION 9 ###############

k_9, p_9=  ks_2samp(movie_ratings['Home Alone (1990)'],movie_ratings['Finding Nemo (2003)'])
#plt.scatter(range(0,len(movie_ratings['Home Alone (1990)'])),movie_ratings['Home Alone (1990)'])
#plt.scatter(range(0,len(movie_ratings['Finding Nemo (2003)'])),movie_ratings['Finding Nemo (2003)'], color = 'orange')
plt.figure()
plt.hist(movie_ratings['Home Alone (1990)'], alpha = 0.5, color = 'blue')
plt.hist(movie_ratings['Finding Nemo (2003)'], alpha = .3, color = 'orange')
colors = {'Nemo':'orange', 'Home alone':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = .5) for label in labels]
plt.legend(handles, labels)

plt.show()
print('For Question 9:')
print('Our Kolmogorov-Smirnov Test statistic is', k_9, 'with p value', p_9)
print('We reject our Null Hypothesis')
print('')



############### QUESTION 10 ###############

from collections import defaultdict

franchisesList = ['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones', 'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']

franchises = defaultdict(list)

for col in movie_ratings.columns:
    for franchise in franchisesList:
        if franchise in col:
            franchises[franchise].append(col)
            
print('For Question 10:')
star_wars_rating = pd.concat([movie_ratings[franchises['Star Wars'][0]], movie_ratings[franchises['Star Wars'][1]],
                              movie_ratings[franchises['Star Wars'][2]], movie_ratings[franchises['Star Wars'][3]],
                              movie_ratings[franchises['Star Wars'][4]], movie_ratings[franchises['Star Wars'][5]]], axis = 1) 
star_wars = star_wars_rating.dropna()
f_star, p_star = f_oneway(star_wars.iloc[0], star_wars.iloc[1], star_wars.iloc[2],
                          star_wars.iloc[3], star_wars.iloc[4], star_wars.iloc[5])
print( 'Star Wars F Value = ',f_star, 'with p value', p_star)

harry_rating = pd.concat([movie_ratings[franchises['Harry Potter'][0]], movie_ratings[franchises['Harry Potter'][1]],
                          movie_ratings[franchises['Harry Potter'][2]], movie_ratings[franchises['Harry Potter'][3]]], axis = 1 )
harry_potter = harry_rating.dropna()

f_harry, p_harry = f_oneway(harry_potter.iloc[0], harry_potter.iloc[1],
                          harry_potter.iloc[2], harry_potter.iloc[3])    
        
print('Harry Potter F Value = ', f_harry, 'with p value', p_harry)


matrix_rating = pd.concat([movie_ratings[franchises['The Matrix'][0]], movie_ratings[franchises['The Matrix'][1]],
                          movie_ratings[franchises['The Matrix'][2]]], axis = 1 )
the_matrix = matrix_rating.dropna()

f_matrix, p_matrix = f_oneway(the_matrix.iloc[0], the_matrix.iloc[1], the_matrix.iloc[2]) 
         
print('The Matrix F Value = ', f_matrix, 'with p value', p_matrix)

Indiana_rating = pd.concat([movie_ratings[franchises['Indiana Jones'][0]], movie_ratings[franchises['Indiana Jones'][1]],
                          movie_ratings[franchises['Indiana Jones'][2]], movie_ratings[franchises['Indiana Jones'][3]]], axis = 1 )
indiana_jones = Indiana_rating.dropna()

f_indiana, p_indiana = f_oneway(indiana_jones.iloc[0], indiana_jones.iloc[1],
                          indiana_jones.iloc[2], indiana_jones.iloc[3])    
        
print('Indiana Jones F Value = ', f_indiana, 'with p value', p_indiana)

jurassic_rating = pd.concat([movie_ratings[franchises['Jurassic Park'][0]], movie_ratings[franchises['Jurassic Park'][1]],
                          movie_ratings[franchises['Jurassic Park'][2]]], axis = 1 )
Jurassic_Park = jurassic_rating.dropna()

f_jurassic, p_jurassic = f_oneway(Jurassic_Park.iloc[0], Jurassic_Park.iloc[1], Jurassic_Park.iloc[2]) 
         
print('Jurassic Park F Value = ', f_jurassic, 'with p value', p_jurassic)

pirates_rating = pd.concat([movie_ratings[franchises['Pirates of the Caribbean'][0]], movie_ratings[franchises['Pirates of the Caribbean'][1]],
                          movie_ratings[franchises['Pirates of the Caribbean'][2]]], axis = 1 )
Pirates_oC = pirates_rating.dropna()

f_pirates, p_pirates = f_oneway(Pirates_oC.iloc[0], Pirates_oC.iloc[1], Pirates_oC.iloc[2]) 
         
print('Pirates of the Caribbean F Value = ', f_pirates, 'with p value', p_pirates)

toys_rating = pd.concat([movie_ratings[franchises['Toy Story'][0]], movie_ratings[franchises['Toy Story'][1]],
                          movie_ratings[franchises['Toy Story'][2]]], axis = 1 )
Toy_Story = toys_rating.dropna()

f_toys, p_toys = f_oneway(Toy_Story.iloc[0], Toy_Story.iloc[1], Toy_Story.iloc[2]) 
         
print('Toy Story F Value = ', f_toys, 'with p value', p_toys)


bat_rating = pd.concat([movie_ratings[franchises['Batman'][0]], movie_ratings[franchises['Batman'][1]],
                          movie_ratings[franchises['Batman'][2]]], axis = 1 )
Batman = bat_rating.dropna()

f_bat, p_bat = f_oneway(Batman.iloc[0], Batman.iloc[1], Batman.iloc[2]) 
         
print('Batman F Value = ', f_bat, 'with p value', p_bat)

p_values = pd.Series([p_star, p_harry, p_matrix, p_indiana, p_jurassic, p_pirates, p_toys, p_bat])

print('We only reject our null Hypothesis for Harry Potter')




