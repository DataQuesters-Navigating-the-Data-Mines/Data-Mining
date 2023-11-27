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
      
    3. Abdul handled the KMean clustering next, using the clean dataset generated above.
      3.1 Loaded the dataset
      3.2 Performed Z normalisation and saved the dataset to a new CSV file
      3.3 Found the optimal number of clusters
      3.4 Generated the Elbow plot graph
      3.5 Added cluster labels
      3.6 Performed the actual KMeans clustering
      3.7 Created a scatter plot graph
      3.8 Set the graphs limits and titles
      3.9 Displayed the graph
  
