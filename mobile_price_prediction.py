'''
1. Business Understanding
       A new Mobile phone manufaturer needs to decide how to price their phones to be competetive in the market. Thus the objective in this example is to decide
       (predict) price-segment of Mobile phones based on various sepcifications of the phone. A Mobile phone's price should be linked to the features or 
       specifications it provides. Brand Value of the manufacturer also play a critical role but that applies only for established players in the market. 
       Hence in our case since it's a new manufacturer, we can focus on features or specs of the phone to decide the price segment. 
       The features may include the following:    
        Battery Power in mAH           
        Bluetooth                
        Clock speed of its processor          
        Dual SIM                    
        Front Camera MegaPixel             
        whether 4G enabled            
        Internal Memory in GB         
        Depth in cm                
        Weight in gm               
        No. of processor cores           
        Primary Camera MegaPixel            
        Pixel Resolution height        
        Pixel Resolution width          
        RAM in MB                     
        Screen Height                   
        Screen Width                 
        whether it has TouchScreen             
        whether it has WiFi

    2. Sources of data - URL: https://raw.githubusercontent.com/pranav-09/mobile-price-data/main/mobile_price_data.csv
        
    3. Analytics task performed - Predictive Analytics.

'''
# Downloading data
from urllib import request
URL = "https://raw.githubusercontent.com/pranav-09/mobile-price-data/main/mobile_price_data.csv"
response = request.urlretrieve(URL, "mobile_price_data.csv")


# Code for converting the above downloaded data into a dataframe

import pandas as pd
df=pd.read_csv("mobile_price_data.csv")


# Confirm the data has been correctly by displaying the first 5 and last 5 records.

print(df.head())
print(df.tail())


# Display the column headings, statistical information, description and statistical summary of the data.

df.info()
print("\n\nShape of the dataset is ", df.shape)
df.describe()

'''
# Writing your observations from the above. 
# 1. Size of the dataset - Size of dataset = 2000 rows and 21 columns
# 2. What type of data attributes are there?
        Types of Data Attributes:

No.      Column             Attribute type  
---  ------         --------------
 0   battery_power  Ratio                       
 1   blue           Nominal                              
 2   clock_speed    Ratio                             
 3   dual_sim       Nominal                         
 4   fc             Ratio                         
 5   four_g         Nominal                              
 6   int_memory     Ratio                                  
 7   m_dep          Ratio                                
 8   mobile_wt      Ratio                                 
 9   n_cores        Ratio                             
 10  pc             Ratio                                          
 11  px_height      Ratio                             
 12  px_width       Ratio                                 
 13  ram            Ratio                                    
 14  sc_h           Ratio                       
 15  sc_w           Ratio                        
 16  talk_time      Ratio                      
 17  three_g        Nominal                        
 18  touch_screen   Nominal                          
 19  wifi           Nominal                                  
 20  Price_Range    Nominal  

'''
# Check for duplicate data
print("Count of Duplicate Records is ",df.duplicated().sum())

# Check for missing data
print("\n\nCount of Missing Values", 
      "\n--------------------------\n", df.isnull().sum(),"\n")

# Check for data inconsistencies in categorical variables
print(df['Price_Range'].value_counts(),"\n")
print(df['blue'].value_counts(),"\n")
print(df['dual_sim'].value_counts(),"\n")
print(df['three_g'].value_counts(),"\n")
print(df['four_g'].value_counts(),"\n")
print(df['touch_screen'].value_counts(),"\n")


# Removing Duplicates
df.drop_duplicates()

# Handling missing data
import matplotlib.pyplot as plt
import seaborn as sns

# Draw the Box-plot for battery_poer to check for skewed data, outliers
sns.boxplot(x=df['battery_power'])


# Since data is not skewed, we will use mean to impute missing values for battery_power
df['battery_power'] = df['battery_power'].fillna(df['battery_power'].mean())

#validating that there are no more missing values for battery_power
df['battery_power'].isnull().sum()


# Draw the Box-plot for clock_speed to check for skewed data, outliers
sns.boxplot(x=df['clock_speed'])

# Since data is skewed, we will use mean to impute missing values for clock_speed
df['clock_speed'] = df['clock_speed'].fillna(df['clock_speed'].median())

#validating that there are no more missing values for clock_speed
df['clock_speed'].isnull().sum()


# Handling data inconsisencies
# We have seen above the value counts for column blue,
# dual_sim, three_g, four_g, touch_screen, wifi
# are both Y/N and 1/0 with majority being 1/0

df['blue']=df['blue'].map({'1':1,'0':0, 'Y':1, 'N':0})
df['dual_sim']=df['dual_sim'].map({'1':1,'0':0, 'Y':1, 'N':0})
df['three_g']=df['three_g'].map({'1':1,'0':0, 'Y':1, 'N':0})
df['four_g']=df['four_g'].map({'1':1,'0':0, 'Y':1, 'N':0})
df['touch_screen']=df['touch_screen'].map({'1':1,'0':0, 'Y':1, 'N':0})
df['wifi']=df['wifi'].map({'1':1,'0':0, 'Y':1, 'N':0})

# Display to ensure that inconsitencies have now been removed

print(df['blue'].value_counts(),"\n")
print(df['dual_sim'].value_counts(),"\n")
print(df['three_g'].value_counts(),"\n")
print(df['four_g'].value_counts(),"\n")
print(df['touch_screen'].value_counts(),"\n")
print(df['wifi'].value_counts(),"\n")


# Display list of values of categorical field Price_Range

print(df['Price_Range'].unique())

# Apply Label encoding to Price_Range

from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
df['Price_Range']=labelencoder.fit_transform(df['Price_Range'])

# Display list of of encoded values for field Price_Range

print(df['Price_Range'].unique())

'''
Method to handle duplicate data
----------------------------------------------
We checked above and found that the data does not have any duplicates, 
by the use of duplicated() function.
However if we have duplicates in the data we can use drop_duplicates().

Method to handle missing data
--------------------------------------------
Records with missing data can be removed if it has missing data for 
most of the columns. Similarly if a column has missing data for most 
of the records then we can simply drop that column. In our case 
missing data appears in  just a couple of columns and only on a handful 
of observations, so we will resort to imputing instead of removing the 
missing values.

Imputing means replacing with some value, which ideally does not 
introduce any bias in the data. Hence before imputing we checked whether 
the distribution of the given attribute (column) is skewed or not. In 
case the distribution of an attribute is not skewed, we prefer to use 
mean value in place of missing values as we can see above for 
battery_power. In case the distribution is skewed, then median is a 
better choice for imputing missing values, as we can see above for 
clock_speed.

Method to remove data inconsistencies
--------------------------------------------------------
We noted above that some of the nominal attributes had inconsistent 
values, e.g. Majority of the observations have a 1 / 0  values but 
for a handful of observation it has Y / N values for following 
attributes:
blue
dual_sim
threee_g
four_g

To make the data consistent we have used the map method to map the 
'Y', 'N' values to 1 and 0 respectively. Also we converted '1' and '0'
string values to int 1 and 0 respectively in the same step to make it 
both consistent as well as numeric.
'''
# Target variable i.e. the one we plan to predict is Price_Range

Label = df['Price_Range']
print(Label)

# Features are all other variables in the data

Features = df.loc[:,df.columns != 'Price_Range']
Features

# Draw Scatter Plot for each of the Features with the Label

features_list = list(Features.columns.values)

fig, axes = plt.subplots(5,4, figsize=(20,30)) 

for feature, ax in zip(features_list, axes.ravel()):
    ax.set_title(feature)
    sns.scatterplot(x=feature,y='Price_Range',data=df,alpha=0.2,ax=ax)


'''
We are using the following plots to help with our EDA:

1. Pair Plot - beacuse it shows: 
        - the distribution for each feature on the daigonal, and 
        - the scatter plot between rest of the features.
        Related numeric features can be easily identified through 
        pairplot.

2. Heat Map of Correlation - as it shows 
        - the correlation among various features, as well as 
        - the correlation between each feature with the label
        This again makes for a very visually intuitive plot which 
        helps us see in a glance whic all features are most 
        correlated to label and hence a good set of features. 
        Also it help us see if there are features whihc are highly 
        correlated to one another and hence are not adding much 
        value as independednt features.
'''

# Draw PairPlot of all Numeirc Features (Ratio attributes) with one-another

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(25,25)})
sns.pairplot(Features, 
             vars=["battery_power","clock_speed","fc","int_memory",
                   "m_dep","mobile_wt","n_cores","pc","px_height",
                   "px_width","ram","sc_h","sc_w","talk_time"])

'''
As we can see in the above pairplot:         
    - features 3 and 8 are correlated (i.e. fc and pc)     
    - features 9 and 10 are correlated (i.e. px_height and px_width)      
    - features 12 and 13 are correlated (i.e. sc_h and sc_w)       
    
    This makes sense as front-camera (fc) and primary-camera (pc) 
    megapixels being related is intuitive. Smilarly Height (px_height) 
    and Width (px_width) of the Pixels being related makes sense.
    And Height (sc_h) and Width (sc_w) of the Screen being related 
    makes sense.
    
    Hence the abov Pair-Plot helps us find these relations in the 
    data and after plotting this we know that we should either 
    derive a new feature by combining these features (e.g. 
    Screen_Size = sc_h * sc_w etc.) or use only one from each of 
    such related pairs. For the sake of simplicity we will drop 
    one feature from each related pair and proceed to drop pc, 
    px_width, sc_w from the Feature list.
''' 

# Draw Heat Map of Correlation of all variable pairs

corr = df.corr()
sns.set(rc={'figure.figsize':(7,5)})
sns.heatmap(corr, cmap='mako')

'''
As we can clearly see in the above Correlation Heatmap:           
 - pc and fc are highly coreelated to each-other (same 
 observation as in pairplot above)             
 - three_g and four_g are highly correlated to each other               
 - px_width and px_height are highly correlated to each 
   other (same observation as in pairplot above)              
 - sc_w and sc_h are highly correlated to each other 
   (same observation as in pairplot above)                   
 - feature ram is highly correlated to label Price_Range 
   and hence can act as a very good feature in predicting 
   the label          
'''

# Dropping one feature each, form all correlated pairs as described above

Features = Features.drop(['fc','three_g','px_height','sc_h'],axis=1)
print(Features.shape)
Features.head()

Features.info()


# Apply Filter Methods for Feature Selection

from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import mutual_info_classif 
from sklearn.feature_selection import f_classif

# Mutual InfoFilter Score for all Features

fs_MutualInfo = SelectKBest(score_func=mutual_info_classif, k='all')
fs_MutualInfo.fit(Features, Label)
Features_MutualInfo = fs_MutualInfo.transform(Features)
for i in range(len(fs_MutualInfo.scores_)):
    print("Feature ", i, " has MutualInfo Gain ", fs_MutualInfo.scores_[i])

print("\n")

# Fisher Score for all Features

fs_Fisher = SelectKBest(score_func=f_classif, k='all')
fs_Fisher.fit(Features, Label)
Features_Fisher = fs_Fisher.transform(Features)
for i in range(len(fs_Fisher.scores_)):
    print("Feature ", i, " has Fisher Score ", fs_Fisher.scores_[i])

# Finaly use Fisher Score (ANOVA) Filter Method  to get best 5 Features

print("\nHence Finally selecting 5 best Features using Fisher Score")
fs_final = SelectKBest(score_func=f_classif, k=5)
Features_selected = fs_final.fit_transform(Features, Label)
Features_selected


'''
As noted towards end of Section 4.2 above, on the final feature list 
we have now, some features are nominal while others are ratio.        

No.    Column           Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   battery_power  2000 non-null   float64 - Ratio                  
 1   blue           2000 non-null   int64   - Nominal                  
 2   clock_speed    2000 non-null   float64 - Ratio                 
 3   dual_sim       2000 non-null   int64   - Nominal                   
 4   four_g         2000 non-null   int64   - Nominal                   
 5   int_memory     2000 non-null   int64  - Ratio                     
 6   m_dep          2000 non-null   float64- Ratio                         
 7   mobile_wt      2000 non-null   int64  - Ratio                       
 8   n_cores        2000 non-null   int64  - Ratio                      
 9   pc             2000 non-null   int64  - Ratio                     
 10  px_width       2000 non-null   int64  - Ratio                         
 11  ram            2000 non-null   int64  - Ratio                       
 12  sc_w           2000 non-null   int64  - Ratio                             
 13  talk_time      2000 non-null   int64  - Ratio                        
 14  touch_screen   2000 non-null   int64  - Nominal                     
 15  wifi           2000 non-null   int64  - Nominal                          
  
We know that if Feature and label both are nominal (categorical)       
then we can use Mutual Info or Chi Square. Hence we will use               
Mutual Info Gain for Features 1,3,4,14,15.          

Also, since we know that if Features are Numeric (e.g. ratio)                         
and Label is nominal (categorical) then we use ANOVA (Fisher score).                 
Hence we will use Fisher Test for Features 0,2,5,6,7,8,9,10,11,12,13               
 
Looking at the scores from Fisher Test in above section we note that 
Features with highest scores are 11,0,10,7,5.

While lookingat the Mutual Info Gain, we note that Features with                 
highest mutual info gain are 11,9,10,0,12,                         

We note that for both of the above methods, 3 out of the best 5               
features selected are common - namely features 11,0,10                   

And the feature where they disagree are Feature 9,12 suggested by                   
Mutual Info Vs. Feature 7,5 suggested by Fisher.                        

Since featrues 9,12,7 and 5 are all numeric, and Fisher is better                     
suited for Numeric features, we will go with Fisher's selection of                
best 5 Features namely -11,0,10,7,5 i.e.              

 0   battery_power            
 5   int_memory                    
 7   mobile_wt                       
 10  px_width                          
 11  ram                                      
'''

'''
As described above in Section 1, the objective is to predict 
price segment for a given mobile handset with specific 
attributes. This is claearly a classification problem. 
Hence we can use Decision Tree as it is one of the most 
powerful classifier methods.
'''

# Split entire dataset into 25 % Test Data and 75 % Train Data
from sklearn.model_selection import train_test_split

Feature_train, Feature_test, Label_train, Label_test=train_test_split(
                                                        Features_selected, 
                                                        Label, 
                                                        test_size = 0.25, 
                                                        random_state = 11)

# Apply Decision Tree with GINI

from sklearn.tree import DecisionTreeClassifier

dtree_gini = DecisionTreeClassifier(criterion = "gini", 
                                    random_state = 11, 
                                    max_depth=3, 
                                    min_samples_leaf=10)
dtree_gini.fit(Feature_train, Label_train)
Label_pred_dtree_gini = dtree_gini.predict(Feature_test)

# Apply Decision Tree with Entropy

dtree_entropy = DecisionTreeClassifier(criterion = "entropy", 
                                       random_state = 11, 
                                       max_depth=3, 
                                       min_samples_leaf=10)
dtree_entropy.fit(Feature_train, Label_train)
Label_pred_dtree_entropy = dtree_entropy.predict(Feature_test)


'''
Since we are dealing with a classification problem, we can use 
K Nearest Neighbors to predict our label, as it is one of the 
common classifier methods.
'''
# Apply KNN 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(Feature_train, Label_train)
Label_pred_knn = knn.predict(Feature_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("DESCISION TREE CLASSIFIER USING GINI\n",
      "-----------------------------------")
print("\nConfusion Matrix: \n\n", 
      confusion_matrix(Label_test, 
                       Label_pred_dtree_gini))
print("\nAccuracy : ", 
      accuracy_score(Label_test,
                     Label_pred_dtree_gini)*100) 
print("\nReport : \n", 
      classification_report(Label_test, 
                            Label_pred_dtree_gini))

print("DESCISION TREE CLASSIFIER USING ENTROPY\n",
      "--------------------------------------")
print("\nConfusion Matrix: \n\n", 
      confusion_matrix(Label_test, 
                       Label_pred_dtree_entropy))
print("\nAccuracy : ", 
      accuracy_score(Label_test,
                     Label_pred_dtree_entropy)*100) 
print("\nReport : \n", 
      classification_report(Label_test, 
                            Label_pred_dtree_entropy))

print("KNN CLASSIFIER\n",
      "-------------")
print("\nConfusion Matrix: \n\n", 
      confusion_matrix(Label_test, 
                       Label_pred_knn))
print("\nAccuracy : ", 
      accuracy_score(Label_test,
                     Label_pred_knn)*100) 
print("\nReport : \n", 
      classification_report(Label_test, 
                            Label_pred_knn))



'''
As noted in the step above, KNN model has a better performance than                               
Decision Tree in this case. Hence we can use KNN model to predict                       
the Label in our case.                                       
      
In other words, for any new mobile phone with given set of        
specifications, we can use KNN model defined above to decide which            
price segment it should be assigned to.                                                          

The specifications that we should use as inputs to this model are      
the features which we identified to be the most significant               
features namely:                 
 battery_power            
 int_memory                    
 mobile_wt                       
 px_width                          
 ram                     
                 
Thus, for any new mobile that this manufacturer wants to launch,       
they can use the RAM size, Battery Power in mAh, Mobile Weight          
in gm, Pixel width and Internal Memory specs of the mobile to         
determine which price segment it should belong to and price it             
accordingly, in order to be competitive.                  

In terms of our experience / learning during this business problem                
we can summarize as following points:                   
- We should start with identifying the business problem and            
  define it well.                
- We can expect data to have issues like missing, incomplete,         
  inconsistent and duplicates                                    
- Scatter plot of each of the features with label helps us have            
  a first glance into which feature may be useful                    
- PairPlot can help identify correlation among different features,               
   but it can be tricky if the number of features is large. we took                
   us a while to plot this one just fine.                              
- Correlation HeatMap is much simpler than pair plot but still                 
    manages to show the correlation among different features as well                
    as between each feature with the label. We can see features which            
    are highly correlated with one another (and hence can be dropped),            
    as well as features which are highly correlated with label and                
    hence useful in predicting the label effectively (e.g. ram in our                 
    case).                                        
- If we had more time, we could have gone for creation of new features                
  based on the features which are related, instead of dropping some of                
  them.                                              
- Filter methods should be carefully chosen for Feature selection based             
    on the attribute type of features and Label. We chose Fisher (ANOVA)                
    for ratio type features and nominal label. On the other hand some                 
    features were nominal too so we chose Mutual Info Filter as well.                   
    We displayed scores from both of these methods and then analysed the                  
    5 best manually.                      
    Once we were convinced that the highest scores from both filter             
    methods were numeric(ratio) type then we were confident in using               
    simply the Fisher Score Filter method for feature selection.                                     
- For choice of model, Decision Tree being a simple and easy model to           
    use in a classification problem, we chose that with 2 variations of              
    the criterion - Gini and Entropy. The other model we chose was KNN.                 
    KNN can be tricky in terms of what n_neighbors value we use. Here we               
    chose 4 as the n_neighbor  because we know that our Label has four               
    different classes.                   
- For measuring the performance, we have used the simplest measures           
    which are applicable in a classification problem, namely confusion         
    matrix, accuracy, precision, recall and f1-scores.                        
- FInaly we noted that KNN model performed better in terms of precision,                        
    recall and accuracy as compared to Decision Tree (both with Gini and                       
    with Entropy).                                       
- This was a gret learning experience which allowed us to apply the         
    learnings we have from this course to a real-life business problem                 
    and to learn a few addtional details and nuances along the way.                 
    '''
