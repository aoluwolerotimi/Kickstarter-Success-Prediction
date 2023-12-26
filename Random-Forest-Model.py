# =============================================================================
# MODEL BUILDING
# =============================================================================
### IMPORTING THE LIBRARIES ###
import pandas as pd
from scipy.stats import iqr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


### PREPROCESSING ###

# Importing the dataset
dataset = pd.read_excel('/Users/aoluwolerotimi/Datasets/Kickstarter.xlsx') # Change this to your local path

df = dataset.copy()
# Preserve only the two states of interest for the model 
df = df[df['state'].isin(['failed', 'successful'])]

# compute new variable goal_usd by applying static_usd_rate to goal in local currency
df['goal_usd'] = df['goal'] * df['static_usd_rate']
# round goal_usd to 2 decimal places
df['goal_usd'] = df['goal_usd'].round(2)

# drop irrelevant variables from original dataframe as determined during prior EDA
irrelevant = [
    'id', 'name', 'pledged', 'currency', 'deadline', 'state_changed_at',
    'created_at', 'launched_at', 'staff_pick', 'backers_count', 'usd_pledged',
    'spotlight', 'state_changed_at_weekday', 'created_at_weekday', 'deadline_day',
    'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
    'state_changed_at_hr', 'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
    'launched_at_day', 'launched_at_hr', 'launch_to_state_change_days', 'goal', 'static_usd_rate', 'disable_communication'
]

df.drop(irrelevant, axis=1, inplace=True)

# impute "Uncategorized" for missing values in category
df['category'].fillna('Uncategorized', inplace=True)

# check for outliers in goal_usd using modified IQR method
# calculate IQR
iqr_value = iqr(df['goal_usd'])
# calculate upper and lower bounds
lower = df['goal_usd'].quantile(0.25) - (2.5 * iqr_value)
upper = df['goal_usd'].quantile(0.75) + (2.5 * iqr_value)
# record outliers
ol_goal_usd = df[(df['goal_usd'] < lower) | (df['goal_usd'] > upper)]


# check for outliers in create_to_launch_days using modified IQR method
# calculate IQR
iqr_value = iqr(df['create_to_launch_days'])
# calculate upper and lower bounds
lower = df['create_to_launch_days'].quantile(0.25) - (2.5 * iqr_value)
upper = df['create_to_launch_days'].quantile(0.75) + (2.5 * iqr_value)
# check percentage of outliers
df[(df['create_to_launch_days'] < lower) | (df['create_to_launch_days'] > upper)].shape[0] / df.shape[0]
# record outliers
ol_ctl_days = df[(df['create_to_launch_days'] < lower) | (df['create_to_launch_days'] > upper)]

all_outliers = pd.concat([ol_goal_usd, ol_ctl_days], axis=0, ignore_index=False)
all_outliers = all_outliers.drop_duplicates()

# drop outliers
df.drop(all_outliers.index, axis=0, inplace=True)

# noting categorical and numerical variables
c_var = ['country', 'category', 'deadline_weekday', 'launched_at_weekday', 'deadline_month', 'launched_at_month'] # categoricals
n_var = ['name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean','deadline_yr', 'launched_at_yr', 'create_to_launch_days', 
         'launch_to_deadline_days', 'goal_usd'] # numericals

# drop highly correlated variables (name_len, blurb_len, and launched_at_yr) from original dataframe as determined during prior EDA
df.drop(['name_len', 'blurb_len', 'launched_at_yr'], axis=1, inplace=True)
for feature in ['name_len', 'blurb_len', 'launched_at_yr']:
    n_var.remove(feature)
    
    
### FEATURE ENGINEERING ###

# Set aside target variable and predictors
y = df['state'].copy()
y.rename('successful', inplace=True)
y = y.map({'successful': 1, 'failed': 0})


# create df "x" which has all but "state" column
X_base = df.copy()
X_base.drop('state', axis=1, inplace=True)
# retain only the variables which hit feature importance threshold
base_features = ['goal_usd', 'create_to_launch_days', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days', 'deadline_yr', 'category']  
X_base = X_base[base_features]

X_mapped = X_base.copy()

# mapping dictionary for umbrella categories
u_categories = {
    'Gadgets': 'Tech_Hardware',
    'Uncategorized': 'Other',
    'Experimental': 'Arts',
    'Plays': 'Arts',
    'Spaces': 'Other',
    'Web': 'Tech_Software',
    'Apps': 'Tech_Software',
    'Wearables': 'Tech_Hardware',
    'Software': 'Tech_Software',
    'Festivals': 'Arts',
    'Hardware': 'Tech_Hardware',
    'Robots': 'Tech_Hardware',
    'Makerspaces': 'Arts',
    'Musical': 'Arts',
    'Immersive': 'Arts',
    'Flight': 'Tech_Hardware',
    'Sound': 'Tech_Hardware',
    'Academic': 'Other',
    'Places': 'Other',
    'Thrillers': 'Arts',
    'Webseries': 'Arts',
    'Blues': 'Arts',
    'Shorts': 'Arts'
}
# perform mapping 


X_mapped['category'] = X_mapped['category'].map(u_categories)
X_mapped = pd.get_dummies(X_mapped, columns = ['category'])
rf_features = ['goal_usd', 'create_to_launch_days', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days', 'category_Tech_Software', 'category_Arts']
X_mapped = X_mapped[rf_features]


### MODEL INITIALIZATION & FITTING - GRADIENT BOOSTED RANDOM FOREST ###

# Initialize the Gradient Boosting Classifier with the best parameters as determined during prior hyperparameter tuning
best_gbt = GradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=3,
    max_features= None,  # max_features = n_features
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=100,
    subsample= 1.0,  # subsample = n_samples
    random_state=0
)

# Fit the model with the data
best_gbt.fit(X_mapped, y)



# =============================================================================
# RUNNING MODEL ON UNSEEN TEST SET
# =============================================================================

### PREPROCESSING ###

# Importing the dataset
dataset2 = pd.read_excel('/Users/aoluwolerotimi/Datasets/Kickstarter-Grading-Sample.xlsx')

df2 = dataset.copy()
df2 = df2[df2['state'].isin(['failed', 'successful'])]

# compute new variable goal_usd by applying static_usd_rate to goal in local currency
df2['goal_usd'] = df2['goal'] * df2['static_usd_rate']
# round goal_usd to 2 decimal places
df2['goal_usd'] = df2['goal_usd'].round(2)

# drop irrelevant variables from original dataframe
irrelevant = [
    'id', 'name', 'pledged', 'currency', 'deadline', 'state_changed_at',
    'created_at', 'launched_at', 'staff_pick', 'backers_count', 'usd_pledged',
    'spotlight', 'state_changed_at_weekday', 'created_at_weekday', 'deadline_day',
    'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
    'state_changed_at_hr', 'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
    'launched_at_day', 'launched_at_hr', 'launch_to_state_change_days', 'goal', 'static_usd_rate', 'disable_communication'
]

df2.drop(irrelevant, axis=1, inplace=True)

# impute "Uncategorized" for missing values in category
df2['category'].fillna('Uncategorized', inplace=True)

# calculate IQR
iqr_value = iqr(df2['goal_usd'])
# calculate upper and lower bounds
lower = df2['goal_usd'].quantile(0.25) - (2.5 * iqr_value)
upper = df2['goal_usd'].quantile(0.75) + (2.5 * iqr_value)
ol_goal_usd = df2[(df2['goal_usd'] < lower) | (df2['goal_usd'] > upper)]

# calculate IQR
iqr_value = iqr(df2['create_to_launch_days'])
# calculate upper and lower bounds
lower = df2['create_to_launch_days'].quantile(0.25) - (2.5 * iqr_value)
upper = df2['create_to_launch_days'].quantile(0.75) + (2.5 * iqr_value)
# record outliers
ol_ctl_days = df2[(df2['create_to_launch_days'] < lower) | (df2['create_to_launch_days'] > upper)]

all_outliers = pd.concat([ol_goal_usd, ol_ctl_days], axis=0, ignore_index=False)
all_outliers = all_outliers.drop_duplicates()

# drop outliers
df2.drop(all_outliers.index, axis=0, inplace=True)

# noting categorical and numerical variables
c_var = ['country', 'category', 'deadline_weekday', 'launched_at_weekday', 'deadline_month', 'launched_at_month'] # categoricals
n_var = ['name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean','deadline_yr', 'launched_at_yr', 'create_to_launch_days', 
         'launch_to_deadline_days', 'goal_usd'] # numericals

# drop highly correlated variables (name_len, blurb_len, and launched_at_yr) from original dataframe
df2.drop(['name_len', 'blurb_len', 'launched_at_yr'], axis=1, inplace=True)

for feature in ['name_len', 'blurb_len', 'launched_at_yr']:
    n_var.remove(feature)
    
    
### FEATURE ENGINEERING ###
# Set aside target variable and predictors
y_g = df2['state'].copy()
y_g.rename('successful', inplace=True)
y_g = y_g.map({'successful': 1, 'failed': 0})


# create df "x" which has all but "state" column
X_base_g = df2.copy()
X_base_g.drop('state', axis=1, inplace=True)
# retain only the variables which hit feature importance threshold
base_features = ['goal_usd', 'create_to_launch_days', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days', 'deadline_yr', 'category'] 
X_base_g = X_base_g[base_features]

X_mapped_g = X_base_g.copy()

# mapping dictionary for umbrella categories
u_categories = {
    'Gadgets': 'Tech_Hardware',
    'Uncategorized': 'Other',
    'Experimental': 'Arts',
    'Plays': 'Arts',
    'Spaces': 'Other',
    'Web': 'Tech_Software',
    'Apps': 'Tech_Software',
    'Wearables': 'Tech_Hardware',
    'Software': 'Tech_Software',
    'Festivals': 'Arts',
    'Hardware': 'Tech_Hardware',
    'Robots': 'Tech_Hardware',
    'Makerspaces': 'Arts',
    'Musical': 'Arts',
    'Immersive': 'Arts',
    'Flight': 'Tech_Hardware',
    'Sound': 'Tech_Hardware',
    'Academic': 'Other',
    'Places': 'Other',
    'Thrillers': 'Arts',
    'Webseries': 'Arts',
    'Blues': 'Arts',
    'Shorts': 'Arts'
}
# perform mapping 


X_mapped_g['category'] = X_mapped_g['category'].map(u_categories)
X_mapped_g = pd.get_dummies(X_mapped_g, columns = ['category'])
rf_features = ['goal_usd', 'create_to_launch_days', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days', 'category_Tech_Software', 'category_Arts']
X_mapped_g = X_mapped_g[rf_features]
    
### PREDICTION AND ACCURACY CHECK ###
y_g_pred = best_gbt.predict(X_mapped_g)
a_score = accuracy_score(y_g, y_g_pred)

print(f"Accuracy score is {a_score}")