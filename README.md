# Multiple Linear Regression
## Bike Sharing Assignment

#### Problem Statement:

A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.


A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands

Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 




## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
#### Business Goal:

We are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
1.	From your analysis of the categorical variables from the dataset, what could you infer about their effect on the dependent variable?
Observations from above boxplots for categorical variables:
•	The year box plots indicates that more bikes are rent during 2019.
•	The season box plots indicates that more bikes are rent during fall season.
•	The working day and holiday box plots indicate that more bikes are rent during normal working days than on weekends or holidays.
•	The month box plots indicates that more bikes are rent during september month.
•	The weekday box plots indicates that more bikes are rent during saturday.
•	The weathersit box plots indicates that more bikes are rent during Clear, Few clouds, Partly cloudy weather.

2.	Why is it important to use drop_first=True during dummy variable creation?
drop_first=True is important to use, as it helps in reducing the extra column created during dummy variable creation. Hence it reduces the correlations created among dummy variables.

3.	Looking at the pair-plot among the numerical variables, which one has the highest correlation with the target variable?
By looking at the pair plot temp variable has the highest (0.63) correlation with target variable 'cnt'.

4.	How did you validate the assumptions of Linear Regression after building the model on the training set?
•	The Dependent variable and Independent variable must have a linear relationship. And A simple pairplot of the dataframe can help us see if the Independent variables exhibit linear relationship with the Dependent Variable.
•	No Autocorrelation in residuals. statsmodels’ linear regression summary gives us the DW value amongst other useful insights.
•	No Heteroskedasticity. Residual vs Fitted values plot can tell if Heteroskedasticity is present or not.
If the plot shows a funnel shape pattern, then we say that Heteroskedasticity is present.
•	No Perfect Multicollinearity. In case of very less variables, one could use heatmap, but that isn’t so feasible in case of large number of columns.
Another common way to check would be by calculating VIF (Variance Inflation Factor) values.
If VIF=1, Very Less Multicollinearity
VIF<5, Moderate Multicollinearity
VIF>5 , Extreme Multicollinearity (This is what we have to avoid)
•	Residuals must be normally distributed. Use Distribution plot on the residuals and see if it is normally distributed.


5.	Based on the final model, which are the top 3 features contributing significantly towards explaining the demand of the shared bikes?
The Top 3 features contributing significantly towards the demands of share bikes are:
•	weathersit_Light_Snow(negative correlation).
•	yr_2019(Positive correlation).
•	temp(Positive correlation).


<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.


## Contact
Created by [@githubusername] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->