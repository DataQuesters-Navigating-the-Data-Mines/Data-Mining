# F21DL Data-Mining and Machine Learning

This repository contains our data mining project for Heriot-Watt university, analysing a dataset to answer specific questions and displaying the results, through the use of python and jupyter notebooks.

## Topic: Analysing the Health Impact of COVID-19 on Emergency Patients: Integrating Patient Records and Chest X-rays.

By analysing this data, we hope to be able to understand the impact COVID-19 has on the health of patients, as well as the potential outcomes in an emegerency setting.

## Questions:

    a. Prevalence and Severity of COVID-19:
        Question: What is the prevalence and severity of COVID-19 among emergency patients in Mexico?
        Objective: Analyze the distribution and severity of COVID-19 cases in the dataset.

    b. Age-Related Impact:
        Question: How does the impact of COVID-19 vary across different age groups (10-20, 20-30, ..., 50-60, 60+)?
        Objective: Investigate the correlation between age and COVID-19 severity.

    c. Other Infections:
        Question: Are there patterns indicating the co-occurrence of other infections with COVID-19?
        Objective: Explore the presence of other infections in conjunction with COVID-19.

    d. Duration of COVID-19 Impact:
        Question: What is the average duration of COVID-19 symptoms, and does it differ by age group?
        Objective: Calculate the average duration of symptoms and analyze variations.

    e. Impact on Pre-existing Health Conditions:
        Question: How does COVID-19 impact patients with pre-existing health conditions?
        Objective: Examine the exacerbation or alteration of existing health conditions post-COVID.

    f. Machine Learning for Severity Prediction:
        Question: Can a CNN model effectively predict the severity of COVID-19 based on chest X-rays?
        Objective: Train and validate a CNN model using pre- and post-COVID chest X-rays.

## Hypothesis

    Age and Severity:
        Hypothesis: Younger patients (10-30 years) will generally experience milder forms of COVID-19, while older patients (60+) may have a higher likelihood of severe outcomes.

    Impact on Pre-existing Conditions:
        Hypothesis: Patients with pre-existing health conditions will experience a more significant impact from COVID-19 compared to those without underlying conditions.

    Duration of Symptoms:
        Hypothesis: The duration of COVID-19 symptoms will be longer for older patients compared to younger ones.

    Co-occurrence of Infections:
        Hypothesis: There may be a correlation between the presence of other infections and the severity of COVID-19 cases.

    Effectiveness of CNN Model:
        Hypothesis: A Convolutional Neural Network (CNN) model trained on chest X-rays can effectively predict the severity of COVID-19 cases with a high level of accuracy.

## Assumptions

    Data Quality:
        Assumption: The hospital emergency patient records and chest X-rays are accurate, complete, and representative of the population.

    Consistency in Diagnoses:
        Assumption: The diagnostic criteria for COVID-19 and other health conditions have remained consistent throughout the data collection period.

    Generalization of CNN Model:
        Assumption: The CNN model trained on the selected chest X-ray datasets is generalizable to the broader population and can provide meaningful predictions.

    Ethical Use of Data:
        Assumption: The data used in the analysis has been obtained and is being used in compliance with ethical standards, ensuring patient privacy and confidentiality.

    Impact of Interventions:
        Assumption: The recorded treatments and interventions post-COVID are accurately represented in the patient records, and their impact can be assessed based on the available data.

## Steps Taken

    1. Selected the data set. This was obtained from Kaggle: https://www.kaggle.com/datasets/meirnizri/covid19-dataset using a dataset from the Mexican Government.
      1.1 Selected python version and libraries
        1.1.1 Python 3.9.1
        1.1.2 Pandas 2.1.3
        1.1.3 matpotlib 3.8.1
        1.1.4 sklearn 1.3.2        
    
    2. Sharib cleaned the dataset, following the steps in the R1 pre-processing file. The data was cleaned so that certain values were attributed to specific variables, such as gender.
      2.1 Data set loaded
      2.2 Viewed the top 10 rows of the dataset
      2.3 Viewed the entire dataset
      2.4 Checked for duplicates
      2.5 Printed the datatypes
      2.6 Checked for null or missing values
      2.7 Checked the real status of missing values
      2.8 Created a heatmap to represent null values
      2.9 Check for strings in "DATE_DIED"
      2.10 Created column to assign a value if the person is dead
      2.11 Checked number of unique values in the dead column
      2.12 Replaced the specific values in the "DATE_DIED" column
      2.13 Checked modified data
      2.14 Converted the "DATE_DIED" column to datetime objects
      2.15 Counted null values in "DATE_DIED" column
      2.16 Summarised the data
      2.17 Counted number of values in "AGE" column that are over 110 
      2.18 Checked for number of rows containing "1" for Gender, which covers female patients
      2.19 Checked the number of those rows that are also pregnant
      2.20 Counted the number of males who also have the pregnant tag
      2.21 Reassigned gender values to 2 = male and 1 = female
      2.22 Counted modified values
      2.23 Counted frequency of unique values in "ICU" column
      2.24 Analysed the missing values, and came to the conclusion that the "missing" values coded as 97 were not hospitalised, while the missingv alues reprsented by 99 were hospitalised. In both cases, these values cannot be determined based on the available data
      2.25 Counted frequency of the values in the "INTUBED" column
      2.26 Replaced values as anyone nothospitalised also cannot have being intubed.
      2.27 Summarised the new data
      2.28 Undid the modifications to "DATE_DIED" due to updated data
      2.29 Counted null values
      2.30 Generated heatmap of null values
      2.31 Summarised all data, rounded to 3 decimals
      2.32 Finally print the final dataset to a CSV

    3. Next is the Exploratory Data Analysis
      3.1 Created a large heatmap, excluding the "DATE_DIED" column
      3.2 Removed certain columns to create a new dataframe, "df_med"
      3.3 Counted duplicate rows in "df_med"
      3.4 Counted number of occurrences of "1" in the "DEAD" column
      3.5 Created a bar chart showing how many died
      3.6 Created a pie chart showing the percentage of how many died
      3.7 Created a new dataframe, "df_dead" that contains only the data of those who died
      3.8 Counted the frequency of unique values in the "CLASSIFICATION_FINAL" column of "df_dead"
      3.9 Created function to determine if a patient is a COVID-19 carrier
      3.10 Applied this function to create a new column in "df_dead" to show if the patient is a carrier or not
      3.11 Created a pie chart to show how many of those that died had COVID-19
      3.12 Applied this function to the entire dataset
      3.13 Created pie chart to show how many patients as a whole had COVID-19
      3.14 Created new dataframe, "Covid_deaths" containing only those who had COVID-19
      3.15 Created a histogram visualising the distributionb of ages in the main dataframe
      3.16 Created a line plot graph to explore how age affected the likelyhood of having COVID-19
      3.17 Created a bar graph showing whether or not gender had any impact on the frequency of catching COVID-19
      3.18 Created a bar graph showing the frequency of COVID-19 in women whow ere pregnant vs not-pregnant
      3.19 Created pie chart showing the percentage of pregnant females who had and did not have COVID-19
      3.20 Created new dataframe, "df_diseases"
      3.21 Created multiple bar charts showing which diseases frequently were paired with COVID-19
      3.22 Created multiple bar charts showing the distrbution of these other diseases amongst patients
      3.23 Created pie chart of what percentage of patients were hospitalised.
      3.24 Created a pie chart to show what percentage of those who were hospitalised died.
      3.25 Created a second dataframe, "df_diseases2", based only upon those hospitalised
      3.26 Created a pie chart showing how many of those hospitalised died
      3.27 COmpared how many of those hospitalised also were sent to the ICU
      3.28 Created pie chart showing what percentage of those hospitalised were sent to the ICU
      3.29 Compared how many of those in ICU had COVID-19 vs did not have COVID-19
      3.30 Created pie chart showing what percentage of those int he ICU had COVID-19 vs did not have COVID-19
      3.31 Counted and created a pie chart of those who died in the ICU.
      3.32 Created a histogram visualising the trend of deaths over time
      
    4. Abdul handled the KMean clustering next, using the clean dataset generated above.
      4.1 Loaded the dataset
      4.2 Performed Z normalisation and saved the dataset to a new CSV file
      4.3 Found the optimal number of clusters
      4.4 Generated the Elbow plot graph
      4.5 Added cluster labels
      4.6 Performed the actual KMeans clustering
      4.7 Created a scatter plot graph
      4.8 Set the graphs limits and titles
      4.9 Display the graph

    5. Pravek handled the CNN prediction

    6. Muhammad handled the Neural network
      6.1 Read in the dataset
      6.2 Extracted the features and target variables
      6.3 Indentified the numerical columns
      6.4 Extracted the relevant features from the "DATE_DIED" column
      6.5 Dropped the original "DATE_DIED" column
      6.6 One-hot encoded variables
      6.7 Split the data into training and testing sets
      6.8 Indentified the numerical columns once One-shot encoding is complete
      6.9 Use  acolumn transformer to standardise the numerical features
      6.10 Transformed and fit the data into the model
      6.11 Built the neural network model
      6.12  Compiled the model
      6.13 Train the model
      6.14 Evaluate the model on the test set
      6.15 Post findings
      6.16 Save the model

    7. Sharib handled the decision tree
      7.1 Created a new dataframe, "df_model"
      7.2 Calculated the shape of "df_model"
      7.3 Displayed the shape of "df_model"
      7.4 Determined the number of missing values
      7.5 Removed all rows with missing values
      7.6 Created heatmap to visualise the presence of missing values
      7.7 Counted frequency of each value in "Covid_or_not" column
      7.8 Used a heatmap to visualised the correlation between features in "df" 
      7.9 Removed unnecessary columns from "df_model"
      7.10 Split the data into features and target variables
      7.11 Split into testing and training sets
      7.12 Created a heatmap for correlated features in the training set
      7.13 Created a function to correlate the fatures based upon a specified threshold
      7.14 Remove "Covid_or_not" from both training and testing sets
      7.15 Create a bar graph visualising the target variablebefore handling imbalance
      7.16 Train the model using logistic regression
      7.17 Print the classification for the logistic regression predictions using the test data
      7.18 Train a decision tree classifier, calculating and printing the train and test accuracy scores
      7.19 Print the classification report again

    8. Ross handled updating the repository and admin work
      8.1 Updated the repository to match the brief
      8.2 Filled out documentation for each week as honestly and accurately as possible with what is available
      8.3 Updated the README(s) to explain steps taken as well as what the repository is for
      8.4 Submitted the final ZIP file
      
  
