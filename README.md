
# Kickstarter Success Prediction

In this predictive analytics project, I've developed a model to predict the success or failure of a given Kickstarter fundraising campaign using a gradient boosted random forest classifier

## Context & Summary of Results
[Kickstarter](https://www.kickstarter.com/about) is a crowdfunding platform with a focus on bringing creative projects to life.

As part of a data mining course, I was tasked with developing a classification model which would be evaluated solely on the accuracy of its predictions on a completely unseen dataset. The model could only use attributes which would be known at the time a project is published on the platform. 

Upon evaluation, my final CV score was 72.1% - near the average of 72.5% across 84 analytics students. During my own model creation process, I achieved ~72% accuracy during training and ~73% accuracy on a test set. Overall, this demonstrated the appropriate balance between bias and variance that is characteristic of a stable and generalizable model. 


## Process
At a high level, these are the steps I took in developing this model. This repo currently has the code for my final model. I hope to upload some of the behind-the-scenes code while developing that model once I've gotten that cleaned up. 

#### Preprocessing
* Removed irrelevant variables, including IDs, time stamps, and variables not known at project publication (e.g., staff pick)
* Computed a new 'goal_usd' variable which expressed the goal in the same currency as other key monetary variables 
* Eliminated multicollinearity among variables
* Eliminated outliers in terms of fundraising goal and days from project creation to project publishing on the platform

#### Feature Selection


#### Model Selection & Hyperparameter Tuning



