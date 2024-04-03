# ABTS
Agent Based Travel Scheduler library to predict individual travel schedules


## License
ABTS / version 0.0.3
- install:

```python
!pip install ABTS
import ABTS as abts
```

###### This algorithm is a simulation model for predicting individual travel schedules, named as ABTS (Agent-Based Travel Scheduler). It utilizes four parameters for prediction, and it can decompose past travel data into trip purposes and age groups using historical OD (Origin-Destination) data. Future predictions are also possible. More detailed descriptions of the model can be found in the following two papers.

- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process


###### Figure 1 shows the framework of ABTS. ABTS utilizes a three-step process to generate the initial outcome of individual travel schedules: 1) Comprehensive Travel Classifier, 2) Land-Use Estimator, and 3) Individual Travel Schedule. The output of ABTS includes individual daily trip chains that consists of each trip purpose, occurrence time, destination, duration, and mode, tailored according to age group.

<div style="text-align: center;">
    <img src="/ABTS/image/Figure1_Framework_of_ABTS.png" alt="Figure1. Framework_of_ABTS" width="800"/>
    <p>Table 1. Classification in ABTS</p>
</div>


###### In the comprehensive travel classifier, trip purposes, trip modes, and age groups are classified based on their respective criteria as shown in Table 1. For example, trip purposes are divided into 9 major trip purposes and 13 sub trip purposes.


<div style="text-align: center;">
    <img src="/ABTS/image/Classification_method_in_ABTS.png" alt="Table 1. Classification in ABTS" width="650"/>
    <p>Table 1. Classification in ABTS</p>
</div>


###### In the Land-Use estimator, land-use data is generated. We have constructed the land-use into 13 categories, similar to the sub trip purposes in Table 1, and organized them into shp files. The source of the data is described in the 0. Preprocessing stage below.

---


# Run the model with example data

###### Here, we will run the process of the Individual Travel Schedule Generator using example data. The case area is Milwaukee County, and we will demonstrate the process of decomposing and predicting past travels by trip purpose and age for September 2020.


## 0. Preprocessing (using sample simulation in library)

###### Here, we show the preprocessing steps for 'NHTS' data and 'SafeGraph' data in order to run the ABTS model. The original data can be downloaded from the two websites described below, and the data used in this example is also downloaded from these sites. However, it seems that SafeGraph Neighborhood data is currently available from other sources, yet remains accessible.

- NHTS data: NHTS data: The source is the Federal Highway Administration (FHWA), and it uses data from the 2017 National Household Travel Survey (https://nhts.ornl.gov/downloads). 

- SafeGraph data: The source is SafeGraph, and the data is named Neighborhood Patterns (https://docs.safegraph.com/docs/neighborhood-patterns). This data represents the movements of people by CBG (Census Block Group) as OD (Origin-Destination) for a randomly selected week of the month, recording how many people moved from one CBG to another. However, it does not contain information on trip purposes or ages, collecting only counts. We used this data to create a probabilistic model and applied it to an algorithm for selecting Destination CBGs based on people's trip purposes.


### 0.0. Data import

###### We uploaded the origin data on Google drive (https://drive.google.com/drive/folders/1YlsNPeTIC-0hjgYNnatuhmIGWQVSvg0-?usp=sharing)

- 'trippub.csv': NHTS data (2017)
- 'Milwaukee_parcels.shp': Milwaukee landuse data
- 'neighbor_2020_09.xlsx': SafeGraph Neighborhood data in Milwaukee (2020/09)

``` python
path = "[Your Path where you save three datasets]"
trippub = pd.read_csv(path + 'trippub.csv') # NHTS
neighbor_2020_09 =  pd.read_excel(path + 'neighbor_2020_09.xlsx') # SafeGraph
landUse = gpd.read_file(path + 'Milwaukee_parcels.shp') # Landuse
```

### 0.1. Preprocess NHTS data
#### 0.1.1. Organize columns of NHTS data
###### This function reorganizes the NHTS dataset by selecting columns for analysis, mapping certain categorical codes to more meaningful values, and introducing new columns to better represent the data. It focuses on clarifying trip purposes, transportation modes, and travel days based on NHTS coding schemes.

```python
trippub_organized = abts.organize_columns(trippub, print_progress = True)
display(trippub_organized.head())
```

|    |   HOUSEID |   PERSONID |   PERSONID_new |   HHSTFIPS |   R_AGE_IMP | R_AGE_new   |   WHYFROM |   WHYTO |   TRPMILES |   DWELTIME |   STRTTIME |   ENDTIME |   TRAVDAY | TRAVDAY_new   |   TDAYDATE | TRPPURP_new   | TRPTRANS_new   |
|---:|----------:|-----------:|---------------:|-----------:|------------:|:------------|----------:|--------:|-----------:|-----------:|-----------:|----------:|----------:|:--------------|-----------:|:--------------|:---------------|
|  0 |  30000007 |          1 |      300000071 |         37 |          67 | Seniors     |         1 |      19 |      5.244 |        295 |       1000 |      1015 |         2 | Weekday       |     201608 | S_d_r         | Car            |
|  1 |  30000007 |          1 |      300000071 |         37 |          67 | Seniors     |        19 |       1 |      5.149 |         -9 |       1510 |      1530 |         2 | Weekday       |     201608 | Home          | Car            |
|  2 |  30000007 |          2 |      300000072 |         37 |          66 | Seniors     |         3 |       1 |     84.004 |        540 |        700 |       900 |         2 | Weekday       |     201608 | Home          | Car            |
|  3 |  30000007 |          2 |      300000072 |         37 |          66 | Seniors     |         1 |       3 |     81.628 |         -9 |       1800 |      2030 |         2 | Weekday       |     201608 | Work          | Car            |
|  4 |  30000007 |          3 |      300000073 |         37 |          28 | Adult       |         1 |       8 |      2.25  |        330 |        845 |       900 |         2 | Weekday       |     201608 | S_d_r         | Car            |

- PERSONID_new: Newly assigned unique ID
- HHSTFIPS: Milwaukee code
- R_AGE_IMP: Age
- R_AGE_new: new classification of age group
- Information on the remaining columns can be found in the Format Library provided at https://nhts.ornl.gov/downloads.


#### 0.1.2. Preprocess NHTS
###### Cleans and preprocesses the NHTS dataset (After running organize_columns function) to correct data anomalies and enhance data quality for analysis. The preprocessing steps include adjusting dwelling times, calculating travel times, and ensuring data consistency across the travel records.


```python
repaired_NHTS = abts.preprocess_NHTS(trippub_organized, print_progress = True) # will take long time
repaired_NHTS.to_csv('[yourPath]' + 'repaired_NHTS.csv') # save
display(repaired_NHTS.head())
```

|    |   Unnamed: 0 |    uniqID |   age | Day_Type   | Trip_pur   |   sta_T_hms |   arr_T_hms |   end_T_hms |   Dwell_T_min |   Trip_T_min |   sta_T_min |   arr_T_min |   end_T_min | age_class   |
|---:|-------------:|----------:|------:|:-----------|:-----------|------------:|------------:|------------:|--------------:|-------------:|------------:|------------:|------------:|:------------|
|  0 |            0 | 300000071 |    67 | Weekday    | Home       |           0 |           0 |        1000 |           600 |            0 |           0 |           0 |         600 | Seniors     |
|  1 |            1 | 300000071 |    67 | Weekday    | S_d_r      |        1000 |        1015 |        1510 |           295 |           15 |         600 |         615 |         910 | Seniors     |
|  2 |            2 | 300000071 |    67 | Weekday    | Home       |        1510 |        1530 |        2400 |           510 |           20 |         910 |         930 |        1440 | Seniors     |
|  3 |            3 | 300000073 |    28 | Weekday    | Home       |           0 |           0 |         845 |           525 |            0 |           0 |           0 |         525 | Adult       |
|  4 |            4 | 300000073 |    28 | Weekday    | S_d_r      |         845 |         900 |        1430 |           330 |           15 |         525 |         540 |         870 | Adult       |

- age: Origin age
- Day_Type: Weekday or Weekend
- Trip_pur: trip purpose
- sta_T_hms: Trip start time. (1000 -> 10:00, 1510 -> 15:10)
- arr_T_hms: Trip arrival time
- end_T_hms: End time of activity
- Dwell_T_min: Dwell time (duration - minutes)
- Trip_T_min: Travel time (duration - minutes)
- sta_T_min: Trip start time corresponding to 'sta_T_hms' (600 -> 10:00, 910 -> 15:10)
- arr_T_min: Trip arrival time corresponding to 'arr_T_hms'
- end_T_min: End time of activity corresponding to 'end_T_hms'
- age_class: age groups



#### 0.1.3. Preprocessing for tripmode
###### Processes the NHTS dataset to create a new trip purpose category and calculates the probability of trip mode choices by age and newly defined trip purpose. This function aims to simplify the analysis of trip behaviors across different demographics and trip purposes by mapping detailed purposes into broader categories.

```python
trip_mode_prop_all = abts.preprocess_NHTS_tripMode(trippub_organized) 
trip_mode_prop_all.to_csv('[yourPath]' + 'trip_mode_prop_all.csv')  # save
display(trip_mode_prop_all.head())
```

|    | age_class   | Trip_pur   | Trip_mode   |   nominator |   denominator |   Trip_modeP |
|---:|:------------|:-----------|:------------|------------:|--------------:|-------------:|
|  0 | Adult       | D_shop     | Bicy        |         126 |         15158 |   0.00831244 |
|  1 | Adult       | D_shop     | Car         |       13528 |         15158 |   0.892466   |
|  2 | Adult       | D_shop     | PTrans      |         210 |         15158 |   0.0138541  |
|  3 | Adult       | D_shop     | Walk        |        1294 |         15158 |   0.0853675  |
|  4 | Adult       | Home       | Bicy        |         727 |         46155 |   0.0157513  |

- Trip_modeP: Probability of choosing trip mode for each age class and trip purpose



### 0.2. Preprocess SafeGraph data
#### 0.2.1. DO to OD, Covert Destination area to Origin area
###### Converts Destination-Origin (DO) data into Origin-Destination (OD) data using the input SafeGraph Neighborhood data. The function processes columns related to work behavior and device home areas during weekdays and weekends, transforming them into a format that specifies how many devices move from one area to another.

```python
neighbor_2020_09_OD = abts.DOtoOD(neighbor_2020_09, print_progress = True)
display(neighbor_2020_09_OD.heaD())
```

|    | area | work_behavior_from_area | weekday_device_from_area_home | weekend_device_from_area_home |
|---:|-----:|:------------------------|:-----------------------------|:-----------------------------------|
|  3 | 550791863001 | {'550791863001': 9, '550790144002': 4,...} | {'550791863001': 43, '550791402011': 4,...} | {'550791863001': 11, '550790148001': 4,...} |
|  4 | 550790194001 | {'550791503013': 4, '550790146002': 4,...} | {'550790194001': 28, '550791009002': 6,... } | {'550790194001': 16, '550791009002': 5,...} |


- area: Census Block Groups (CBG), origin home area
- work_behavior_from_area: # people trip to workplace from 'area' (Workplace: # people)
- weekday_device_from_area_home: # people trip from 'area' to certain CBG in weekday (certain CBG: # people)
- weekend_device_from_area_home: # people trip from 'area' to certain CBG in weekend


#### 0.2.2. Computing probability of trips from origin cbg to dest cbg
###### Computes the probability of moving from an origin CBG to a destination CBG, factoring in the spatial attractiveness weight <i>W<sub>s</sub></i> and the distance sensitivity index <i>W<sub>d</sub></i>.

###### Please refer to the Eqution 10 that describe how to compute this probability.


```python
prob_trips_2020_09_ws05_wd025 = abts.compute_probabilityByk_Ws_Wd(neighbor_2020_09_OD, landUse, W_s = 0.5, W_d = 0.25)
display(prob_trips_2020_09_ws05_wd025.head())
```


#### 0.2.3. Combine all probability tables by their month

###### The process involves pre-creating many combinations of <i>W<sub>s</sub></i> and <i>W<sub>d</sub></i>, and then concatenating them into one. This varies depending on how many combinations of <i>W<sub>s</sub></i> and <i>W<sub>d</sub></i> the user creates, hence no code has been provided for it. In this example, the combinations are set as <i>W<sub>s</sub></i> = 0.5,1.5,1 and <i>W<sub>d</sub></i> = 0.25,0.5,0.75, and they are combined into one under the name 'prob_2020_09_combined'.


#### 0.2.4. fill empty probability
###### Fills empty probability distributions for trip purposes in a row of a input DataFrame with alternative values based on a predefined hierarchy of preferences. This ensures that each trip purpose category has a valid probability distribution, either specific or borrowed from a related category.

```python
prob_2020_09_combined = abts.prob_2020_09_combined.apply(fill_values, axis = 1)
prob_2020_09_combined.to_csv('[yourPath]' + 'prob_2020_09_combined.xlsx') # save
display(prob_2020_09_combined.head())
```


|      | area | weekday_Work | weekend_Work | weekday_School | weekday_University | weekday_Dailycare |  weekday_Religion | weekday_Large_shop | weekday_Etc_shop | weekday_Meals | weekday_V_fr_rel | weekday_Rec_lei | weekday_Serv_trip | weekday_Others | weekend_School | weekend_University | weekend_Dailycare | weekend_Religion |  weekend_Large_shop | weekend_Etc_shop | weekend_Meals | weekend_V_fr_rel | weekend_Rec_lei | weekend_Serv_trip |  weekend_Others  | Ws |  Wd |
|-----:|-------:|:----------|:------------|:---------|:--------|:----|:-------|:------|:--------|:----|:-----|:---------|:-----------------|:---------|:----------|:------------|:----------|:---------|:------------|:------------------|:----------------|:------------|:---------|:-----|:--------------|-----:|-----:|
| 1153 | 550790012001 | {'550790012002': 0.23171, '550790019004': 0.27989,...} | {'550790012002': 0.23171, '550790019004': 0.27989,...} | {'550790040001': 0.05985, '550790008001': 0.086...} | {'550791863001': 0.07454, '550790601012': 0.35423,...} | {'550791863001': 0.00127, '550790601021': 0.04264,...} | {'550790076001': 0.00092, '550790039003': 0.00814,...} | {'550790001021': 0.00877, '550790602001': 0.02857,...} | {'550791863001': 0.00024, '550791009002': 0.00016,...} | {'550791863001': 0.00059, '550791009002': 0.00018,...} | {'550791863001': 0.00013, '550790194001': 3e-05,...} | {'550791863001': 0.00051, '550790194001': 3e-05,...} | {'550791863001': 0.00089, '550790194001': 5e-05,...} | {'550791863001': 0.00039, '550791009002': 6e-05,...} | {'550790008001': 0.20682, '550790034001': 0.08553,...} | {'550791863002': 0.24308, '550791853001': 0.75692} | {'550790011001': 0.37633, '550790001012': 0.0137,...} | {'550790008001': 0.09629, '550790034001': 0.03982,...} | {'550790602001': 0.04491, '550790501011': 0.00957,...} | {'550790014002': 0.02371, '550790034001': 0.01567,...} | {'550790034001': 0.02218, '550790069002': 0.003,...} | {'550790014002': 0.01961, '550790008001': 0.05287,...} | {'550790014002': 0.02669, '550790008001': 0.04266,...} | {'550790014002': 0.04832, '550790008001': 0.0546,...} | {'550790014002': 0.00924, '550790008001': 0.02088,...} |  0.5 | 0.5  |
| 5417 | 550790067002 | {'550790067001': 1.0} | {'550790067001': 1.0} | {'550791854002': 0.0472, '550791857001': 0.17183,...} | {'550791863002': 0.58456, '550791853001': 0.41544} | {'550791854002': 0.31535, '550790011001': 0.14428,...} | {'550791854002': 0.04848, '550790065003': 0.06424,...} | {'550791402011': 0.00147, '550791401001': 0.00067,...} | {'550791854002': 0.04594, '550791402011': 0.00069,...} | {'550791854002': 0.01903, '550791402011': 0.00057,...} | {'550791854002': 0.01833, '550791402011': 0.0024, ...} | {'550791854002': 0.01252, '550791402011': 0.00038,...} | {'550791854002': 0.05368, '550791402011': 0.00162,...} | {'550791854002': 0.05892, '550791402011': 0.00068,...} | {'550790602001': 0.07441, '550790070004': 0.29906,...} | {'550790141001': 0.73842, '550791853001': 0.26158,...} | {'550790352001': 1.0} | {'550790024001': 0.01743, '550790070004': 0.21597...} | {'550790602001': 0.9111, '550790044001': 0.0889} | {'550791009002': 0.00877, '550790602001': 0.57703,...} | {'550791009002': 0.00354, '550790108002': 0.02074,...} | {'550791009002': 0.00591, '550790108002': 0.07082,...} | {'550791009002': 0.00499, '550790024001': 0.10415,...} | {'550791009002': 0.0049, '550790602001': 0.31572,...} | {'550791009002': 0.00464, '550790024001': 0.02351,...} |  1   | 0.25 |
| 4588 | 550791007002 | {'550791501002': 9e-05, '550790164002': 0.00022...} | {'550791501002': 9e-05, '550790164002': 0.00022,...} | {'550791401001': 0.00034, '550790125002': 0.02147,...} | {'550791501002': 0.0, '550791008001': 0.01605,...} | {'550791201022': 0.00167, '550791002001': 0.01508,...} | {'550790301002': 0.0, '550791201022': 0.00145,...} | {'550791201022': 0.00345, '550791401001': 7e-05,...} | {'550791009002': 0.39036, '550791201022': 0.00109,...} | {'550791009002': 0.1375, '550790076002': 0.0,...} | {'550791009002': 0.05497, '550790301002': 0.0,...} | {'550791009002': 0.05592, '550790301002': 0.0,...} | {'550791009002': 0.05516, '550790076002': 0.0,...} | {'550791009002': 0.06552, '550791201022': 0.00164,...} | {'550791301002': 0.03874, '550791401001': 0.00045,...} | {'550791008001': 1.0} | {'550791201022': 0.00619, '550791503013': 1e-05,...} | {'550791301002': 0.00017, '550791201022': 0.00331,...} | {'550791201022': 0.00902, '550791401001': 0.00018,...} | {'550791009002': 0.42921, '550791301002': 0.01761,...} | {'550791009002': 0.24759, '550791201022': 0.00241,...} | {'550791009002': 0.04555, '550791301002': 0.00112,...} | {'550791009002': 0.14939, '550791301002': 0.00938,...} | {'550791009002': 0.13345, '550791301002': 0.00096,...} | {'550791009002': 0.10349, '550791301002': 2e-05,...} |  1.5 | 0.75 |

- The output dataframe exhibits the probability of going to each destination by trip purpose for both weekdays and weekends. Each cell in the dataframe has a key representing the destination CBG and a value representing the probability. These defined probability values are used later to determine the destination.

---


## 1. Execution for ABTS
###### In this section, we intend to run the ABTS using the preprocessed data mentioned above. Since the preprocessed data is provided within this library, you can just import those example datasets to run the model.

### 1.0. Data import

```python
import ABTS as abts

prob_2020_09_combined = abts.prob_2020_09_combined # Combined Probability of travels from O to D in Sep, 2020
repaired_NHTS = abts.repaired_NHTS # preprocessed NHTS
trip_mode = abts.trip_mode # trip mode
cbg = abts.cbg # CBG shp file
network_road = abts.network_road # road network data in Milwaukee

```
###### abts.cbg is a downloaded dataset from the Census. abts.network_road dataset is imported using the code stated below. 

```python
import osmnx as ox
network_road = ox.graph_from_place('Milwaukee, Wisconsin, USA', network_type='drive')
```

### 1.1. Trip Occurence Builder
###### In the trip occurrence builder, the probability and total number of trips for each individual are computed using the NHTS data. The daily trip occurrence probability, <i>P</i>(θ), is determined for each day type <i>d</i>, age group <i>a</i>, and trip purpose <i>t</i> for an individual, employing a Naïve Bayes probability model. Day types are differentiated into weekdays and weekends. The parameter <i>W<sub>t</sub></i><sup>(<i>t</i>,<i>d</i>)</sup> modifies the occurrence probability for each trip purpose, where <i>t</i> and <i>d</i> are represented as superscripts, indicating that different values might be allocated based on the trip purpose <i>t</i> and the day type <i>d</i>.


<div style="text-align: center">
    <img src="/ABTS/image/EQ1_prob_t_trip.png" alt="Equation 1. The probability of a person within age group ‘a’ having ‘t’ trip occurs in a single day." height="250"/>
    <p>Equation 1. The probability of a person within age group ‘a’ having ‘t’ trip occurs in a single day.</p>
</div>

###### Following this, the total number of trips <i>k</i> for each agent <i>i</i> is determined as shown in equation 2. Here, <i>W<sub>k</sub></i> serves as a parameter that fine-tunes the daily trip count for each individual. <i>k<sub>(t,i)</sub></i> is modeled to follow a multinomial distribution, where this distribution is sampled for each trip purpose <i>t</i> based on the probability distribution <i>W<sub>t</sub></i>∙<i>p<sub>(1,a,d)</sub></i>, which is conditioned on the age group <i>a</i>, day type <i>d</i>, and trip purpose <i>t</i>. Furthermore, to ensure alignment with the actual data, the aggregate of all daily trip counts <i>k<sub>(t,i)</sub></i> is set to match <i>θ<sub>i</sub></i>.

<div style="text-align: center">
    <img src="/ABTS/image/EQ2_count_t_trips.png" alt="Equation 2. Counting ‘t’ trips occurring ‘k’ times for a single individual ‘i’ in a day." height="225"/>
    <p>Equation 2. Counting ‘t’ trips occurring ‘k’ times for a single individual ‘i’ in a day.</p>
</div>



#### 1.1.0. Data staging
###### You can run this data staging only once for generating multiple travel schedules in the same area. We set the study area to Milwaukee, so you can just save the results of here to create multiple schedules in Milwaukee.


##### 1.1.0.1. The probability of a person with age ‘a’ having ‘t’ trips occurs in a single day
###### Calculate the probability of trip purpose by age group and day type (Weekday or Weekend) using Naive Bayes.

```python
naive_result = abts.naive_bayes_prob_with_day(repaired_NHTS, age_col = 'age_class', tripPurpose_col= 'Trip_pur', travday_col= 'Day_Type')

naive_result.head()
```

|    | Age   | Trip_pur   | Day_Type   |       Prob |
|---:|:------|:-----------|:-----------|-----------:|
| 67 | Child | Others     | Weekend    | 0.0123082  |
| 18 | Adult | Home       | Weekday    | 0.46988    |
| 84 | Teen  | Others     | Weekday    | 0.00842771 |
| 19 | Adult | Home       | Weekend    | 0.483808   |
| 21 | Adult | S_d_r      | Weekend    | 0.0666646  |

- Now, the results exhibit probability of travels by trip purpose for each age group and day type.


##### 1.1.0.2. The number of trips based on unique IDs, age, day type, and trip purpose derived from NHTS data
###### Generate a distribution of trips based on unique IDs, age, day type, and trip purpose (from origin NHTS data).

```python
trip_count = abts.generate_trip_distribution(repaired_NHTS, id_age = 'age_class', id_col = 'uniqID', day_type_col = 'Day_Type', trip_purpose_col = 'Trip_pur')

trip_count.head()
```

|        |    uniqID | age_class   | Day_Type   | Trip_pur   |   count |
|-------:|----------:|:------------|:-----------|:-----------|--------:|
|  72503 | 301276494 | Child       | Weekday    | Home       |       2 |
| 240927 | 304228581 | Seniors     | Weekend    | Home       |       2 |
| 362687 | 401351691 | Adult       | Weekday    | D_shop     |       1 |
| 561851 | 407165011 | Adult       | Weekday    | Home       |       3 |
|  76059 | 301334554 | Teen        | Weekday    | Home       |       1 |

- This table displayes the actual counts of travels collected by each uniq person (each recepient of survey). We will use the results from this 'data staging' to generate simulated travels for individual agent.


#### 1.1.1. The number of trips occurring ‘k’ times for a single individual ‘i’ in a day
###### Generate a combined trip count distribution based on age groups, day type, and trip purpose.

```python

# set up the number of people by age
age_pop = {'Seniors': 3, 'Adult': 3, 'MidAdult': 3, 'Child': 3, 'Teen': 3}

# the number of trips occurring ‘k’ times for a single individual ‘i’ in a day
eachTrip_simul_total_sample = abts.generate_combined_trip_count(
    naive_prob=naive_result,
    trip_count=trip_count,
    age_n_dict=age_pop,
    method='multinomial',
    Home_cbg=550790005021, #home CBG (Census Block Group)
    W_k_weekday = 0.7,
    W_k_weekend = 1.0,
    W_t_weekday={'Work': 1.2, 'S_d_r': 1, 'Rec_lei': 1.1},
    W_t_weekend={'Work': 0.7, 'S_d_r': 1, 'Rec_lei': 1.1},
    print_progress=True
)

# Parameters:
# - naive_prob: DataFrame containing Naive Bayes probabilities.
# - trip_count: DataFrame with trip count data.
# - age_n_dict: Dictionary mapping age groups to the number of individuals.
# - method: String specifying the method to distribute trips ('multinomial' or 'cdf').
# - Home_cbg: The home census block group id.
# - W_k_weekday: Weight multiplier for trip counts on weekdays.
# - W_k_weekend: Weight multiplier for trip counts on weekends.
# - W_t_weekday: Dictionary with trip purpose as keys and weights as values for weekdays.
# - W_t_weekend: Dictionary with trip purpose as keys and weights as values for weekends.
# - print_progress: Boolean flag to print progress.


eachTrip_simul_total_sample.head()
```



|     |   uniqID | ageGroup   | Day_Type   | Week_Type   | TRPPURP   |   count |     Home_cbg |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK |
|----:|---------:|:-----------|:-----------|:------------|:----------|--------:|-------------:|--------:|--------:|--------:|--------:|
| 487 |        6 | MidAdult   | Friday     | Weekday     | S_d_r     |       0 | 550790005021 |     0.7 |       1 |     1   |     1   |
| 717 |       10 | Child      | Saturday   | Weekend     | Others    |       0 | 550790005021 |     0.7 |       1 |     1   |     1   |
| 687 |       10 | Child      | Friday     | Weekday     | Rec_lei   |       0 | 550790005021 |     0.7 |       1 |     1.1 |     1.1 |
| 878 |       13 | Teen       | Friday     | Weekday     | Serv_trip |       0 | 550790005021 |     0.7 |       1 |     1   |     1   |
| 639 |       11 | Child      | Wednesday  | Weekday     | Home      |       2 | 550790005021 |     0.7 |       1 |     1   |     1   |


- Results: simulated counts of each trip for each individual (agent)




### 1.2. Trip Chains Builder
###### The objective of this equation is to find, within the original NHTS data, a trip sequence <i>O</i> composed of trips most similar to those in the simulated trip list <i>S<sub>i</sub></i> (e.g., Home, Home, Work, Rec_lei, etc.) for each individual agent in the current simulation. This process is divided into three steps (equations 3-5).

###### The first step is to find the optimal origin sequence <i>O<sub>i</sub></i> most similar to <i>S<sub>i</sub></i>. Here, <i>S<sub>i</sub></i> is a list of the types and counts of an individual's daily trips by day type, where the sequence is not yet specified. We search for the <i>S<sub>i</sub></i> that is most similar to the set of all possible trip sequences from NHTS <i>Q</i>, and designate the trip sequence from <i>Q</i> with the highest similarity score as <i>O<sub>i</sub></i>. To determine similarity, the Kronecker delta function is used, scoring 1 for each matching element between the two lists and 0 for non-matching ones, with the similarity score being the sum of these values.

###### The second step is to randomly assign the trip sequence for <i>S<sub>i</sub></i>. The first and last trips are designated as Home. Then, excluding the initial and final Home trips, the sequence of the remaining trips <i>t</i> in the simulated trip list <i>S</i> for each individual agent is shuffled randomly. As a result, the sequence of the trips, apart from the first and last Home, is determined randomly.

###### The third step is to reassign the sequence of trips in <i>S<sub>i</sub></i> based on <i>O<sub>i</sub></i>. Here, <i>l</i> and <i>j</i> are indices traversing each trip purpose in <i>O</i> and <i>S</i>, respectively. First, it checks whether the trip following <i>O<sub>(i,a,l)</sub></i> in sequence exists in <i>S<sub>(i,a)</sub></i>. If not, it finds the index <i>j</i> in <i>S<sub>(i,a)</sub></i> that matches the trip purpose of <i>O<sub>(i,a,l+1)</sub></i>. Then, the corresponding trip purpose at <i>S<sub>(i,a,j)</sub></i> is assigned to <i>O<sub>(i,a,l)+1</sub></i>, rearranging <i>S<sub>(i,a)</sub></i> accordingly.

<div style="text-align: center;">
    <img src="/ABTS/image/EQ3_6_tripChain.png" alt="Equation 3-5. Simulated sequence 'S' reassignment based on the most similar origin sequence (O) for individual (i) within age group (a)" height="450"/>
    <p>Equation 3-5. Simulated sequence 'S' reassignment based on the most similar origin sequence 'O' for individual 'i' within age group 'a'</p>
</div>



#### 1.2.0. Data staging
###### Here, again, you run this once and use the result for simulating multiple travel schedules.

##### 1.2.0.1. Create trip seqeunce of individuals using origin NHTS data

```python
trip_sequence_origin = abts.create_trip_sequence(repaired_NHTS, print_progress = True)
trip_sequence_origin.head()
```

|        | age_class   | Day_Type   |    uniqID | Trip_sequence                                                   |   count |
|-------:|:------------|:-----------|----------:|:----------------------------------------------------------------|--------:|
| 125583 | Seniors     | Weekday    | 303362241 | Home-D_shop-Home                                                |       1 |
|  61063 | MidAdult    | Weekday    | 302647372 | Home-Work-Home                                                  |       1 |
|  51450 | MidAdult    | Weekday    | 300479321 | Home-S_d_r-Home-Serv_trip-D_shop-Home-Serv_trip-Home-S_d_r-Home |       1 |
| 195589 | Teen        | Weekend    | 303065214 | Home-S_d_r-Others-Home                                          |       1 |
| 152311 | Seniors     | Weekday    | 404375522 | Home-Serv_trip-D_shop-D_shop-Home                               |       1 |


#### 1.2.1. Finding optimal origin sequence O<i><sub>i</sub></i> most similar to S<i><sub>i</sub></i>
#### 1.2.2. Randomly assign the trip sequence for S<i><sub>i</sub></i>
#### 1.2.3. Reassign the sequence of trip in S<i><sub>i</sub></i> based on O<i><sub>i</sub></i>

```python
simul_trip_sequence_sample = abts.makeTripSequence(eachTrip_simul_total_sample, trip_sequence_origin, print_progress = True)
display(simul_trip_sequence_sample[(simul_trip_sequence_sample['uniqID'] == 7) & (simul_trip_sequence_sample['Day_Type'] == 'Monday')])

```

|     |   uniqID | ageGroup   |     Home_cbg | Day_Type   | Week_Type   | TRPPURP   |   sequence | seq_NHTS                                          |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK |
|----:|---------:|:-----------|-------------:|:-----------|:------------|:----------|-----------:|:--------------------------------------------------|--------:|--------:|--------:|--------:|
| 206 |        7 | MidAdult   | 550790005021 | Monday     | Weekday     | Home      |          1 | Home-Work-S_d_r-Work-S_d_r-Meals-Work-D_shop-Home |     0.7 |       1 |     1   |     1   |
| 207 |        7 | MidAdult   | 550790005021 | Monday     | Weekday     | Work      |          2 | Home-Work-S_d_r-Work-S_d_r-Meals-Work-D_shop-Home |     0.7 |       1 |     1.2 |     0.7 |
| 208 |        7 | MidAdult   | 550790005021 | Monday     | Weekday     | S_d_r     |          3 | Home-Work-S_d_r-Work-S_d_r-Meals-Work-D_shop-Home |     0.7 |       1 |     1   |     1   |
| 209 |        7 | MidAdult   | 550790005021 | Monday     | Weekday     | Home      |          4 | Home-Work-S_d_r-Work-S_d_r-Meals-Work-D_shop-Home |     0.7 |       1 |     1   |     1   |



### 1.3. Trip Timing Estimator


<div style="text-align: center;">
    <img src="/ABTS/image/EQ7_dwellTime.png" alt="Equation 6. Dwell time of all trips for each individual" height="180"/>
    <p>Equation 6. Dwell time of all trips for each individual</p>
</div>


<div style="text-align: center;">
    <img src="/ABTS/image/EQ8_tripStartTime.png" alt="Equation 7. Trip start time for each individual" height="90"/>
    <p>Equation 7. Trip start time for each individual</p>
</div>




#### 1.3.0. Data staging

##### 1.3.0.1. Extracting dwell time by trip purpose using NHTS

```python
dwellTime_dict = abts.dwellTime_listFromNHTS(repaired_NHTS)
```

##### 1.3.0.2. Extracting trip start time by trip purpose using NHTS

```python
startTime_dict = abts.startTime_listFromNHTS(repaired_NHTS) 
```



#### 1.3.1. Estimating dwell time
#### 1.3.2. Estimating trip start time

```python
# assign dwell time and start time to simul data
simul_trip_start_time_sample = abts.assignDwellStartT(simul_trip_sequence_sample, dwellTime_dict, startTime_dict, print_progress = True)
display(simul_trip_start_time_sample[(simul_trip_start_time_sample['uniqID'] == 7) & (simul_trip_start_time_sample['Day_Type'] == 'Monday')])
```


|     |   uniqID | ageGroup   |     Home_cbg | Day_Type   | TRPPURP   |   sequence |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK |   trip_count_class |   Dwell_Time |   sta_T_min | Trip_mode   |
|----:|---------:|:-----------|-------------:|:-----------|:----------|-----------:|--------:|--------:|--------:|--------:|-------------------:|-------------:|------------:|:------------|
| 206 |        7 | MidAdult   | 550790005021 | Monday     | Home      |          1 |     0.7 |       1 |     1   |     1   |                  2 |           42 |           0 | Car         |
| 207 |        7 | MidAdult   | 550790005021 | Monday     | Work      |          2 |     0.7 |       1 |     1.2 |     0.7 |                  2 |          510 |         510 | Car         |
| 208 |        7 | MidAdult   | 550790005021 | Monday     | S_d_r     |          3 |     0.7 |       1 |     1   |     1   |                  2 |            3 |         nan | Car         |
| 209 |        7 | MidAdult   | 550790005021 | Monday     | Home      |          4 |     0.7 |       1 |     1   |     1   |                  2 |          390 |         nan | Car         |




### 1.4. Trip Mode Assigner


<div style="text-align: center;">
    <img src="/ABTS/image/EQ9_tripmode.png" alt="Equation 8. Assign Trip mode applying round trip" height="225"/>
    <p>Equation 8. Assign Trip mode applying round trip</p>
</div>

```python
simul_trip_mode_sample = abts.put_tripMode(simul_trip_start_time_sample, trip_mode, print_progress = True)
display(simul_trip_mode_sample[(simul_trip_mode_sample['uniqID'] == 7) & (simul_trip_mode_sample['Day_Type'] == 'Monday')])
```


|     |   uniqID | ageGroup   |     Home_cbg | Day_Type   | TRPPURP   |   sequence |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK |   trip_count_class |   Dwell_Time |   sta_T_min | Trip_mode   |   Origin |         Dest |    Ws |     Wd | TRPPURP_det   |
|----:|---------:|:-----------|-------------:|:-----------|:----------|-----------:|--------:|--------:|--------:|--------:|-------------------:|-------------:|------------:|:------------|---------:|-------------:|------:|-------:|:--------------|
| 206 |        7 | MidAdult   | 550790005021 | Monday     | Home      |          1 |     0.7 |       1 |     1   |     1   |                  2 |           42 |           0 | nan         |      nan | 550790005021 | nan   | nan    | Home          |
| 207 |        7 | MidAdult   | 550790005021 | Monday     | Work      |          2 |     0.7 |       1 |     1.2 |     0.7 |                  2 |          510 |         510 | Car         |      nan | 550790026001 |   1.5 |   0.5  | Work          |
| 208 |        7 | MidAdult   | 550790005021 | Monday     | S_d_r     |          3 |     0.7 |       1 |     1   |     1   |                  2 |            3 |         nan | Car         |      nan | 550790001022 |   1.5 |   0.25 | Religion      |
| 209 |        7 | MidAdult   | 550790005021 | Monday     | Home      |          4 |     0.7 |       1 |     1   |     1   |                  2 |          390 |         nan | Car         |      nan | 550790005021 | nan   | nan    | Home          |




### 1.5. Spatial Trip Route Estimator


<div style="text-align: center;">
    <img src="/ABTS/image/EQ10_assignDest.png" alt="Equation 9. Probability of trips 't' from origin area 'A' to destination 'D'" height="300"/>
    <p>Equation 9. Probability of trips 't' from origin area 'A' to destination 'D'</p>
</div>



#### 1.5.0. Data staging

##### 1.5.0.1. Ratio between straight path and network path

```python
ratio_table, average_distance_ratio = abts.calculate_sampled_network_distance(cbg_gdf = cbg, network_road = network_road, num_samples = 200)
print('average_distance_ratio: ', average_distance_ratio)

# average_distance_ratio:  1.161508600010609
```



#### 1.5.1. Estimate probabilistic destinations for trip purpose t

```python
simul_trip_routes_sample, assigned_trip_table = abts.assignDest(prob_2020_09_combined, simul_trip_mode_sample, print_progress = True) # randomly select 'W_d' and 'W_s' in prob_2020_09_combined dataframe.
```



```python
# set W_d and W_s manually
W_d_W_s = {'Rec_lei': {'Ws': 1.5, 'Wd': 0.5}, 
           'Meals': {'Ws': 0.5, 'Wd': 0.75}, 
           'Work': {'Ws': 1.0, 'Wd': 0.5}, 
           'Serv_trip': {'Ws': 0.5, 'Wd': 0.75}, 
           'D_shop': {'Ws': 0.5, 'Wd': 0.25}, 
           'S_d_r': {'Ws': 1.0, 'Wd': 0.5}, 
           'Others': {'Ws': 0.5, 'Wd': 0.25}, 
           'V_fr_rel': {'Ws': 1.0, 'Wd': 0.5}}

simul_trip_routes_sample, assigned_trip_table = abts.assignDest(prob_2020_09_combined, simul_trip_mode_sample,
                                                                 W_d_W_s = W_d_W_s,
                                                                 print_progress = True)

display(simul_trip_routes_sample[(simul_trip_routes_sample['uniqID'] == 7) & (simul_trip_routes_sample['Day_Type'] == 'Monday')])
```


|     |   uniqID | ageGroup   |     Home_cbg | Day_Type   | TRPPURP   |   sequence |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK |   trip_count_class |   Dwell_Time |   sta_T_min | Trip_mode   |       Origin |         Dest |    Ws |     Wd | TRPPURP_det   |   TripDist |   TripTime |
|----:|---------:|:-----------|-------------:|:-----------|:----------|-----------:|--------:|--------:|--------:|--------:|-------------------:|-------------:|------------:|:------------|-------------:|-------------:|------:|-------:|:--------------|-----------:|-----------:|
| 206 |        7 | MidAdult   | 550790005021 | Monday     | Home      |          1 |     0.7 |       1 |     1   |     1   |                  2 |           42 |           0 | nan         | 550790005021 | 550790005021 | nan   | nan    | Home          |   0.552907 |        nan |
| 207 |        7 | MidAdult   | 550790005021 | Monday     | Work      |          2 |     0.7 |       1 |     1.2 |     0.7 |                  2 |          510 |         510 | Car         | 550790005021 | 550790026001 |   1.5 |   0.5  | Work          |   7.75753  |         12 |
| 208 |        7 | MidAdult   | 550790005021 | Monday     | S_d_r     |          3 |     0.7 |       1 |     1   |     1   |                  2 |            3 |         nan | Car         | 550790026001 | 550790001022 |   1.5 |   0.25 | Religion      |   9.48366  |         14 |
| 209 |        7 | MidAdult   | 550790005021 | Monday     | Home      |          4 |     0.7 |       1 |     1   |     1   |                  2 |          390 |         nan | Car         | 550790001022 | 550790005021 | nan   | nan    | Home          |   5.54624  |          8 |


#### 1.5.2. Compute trip distance and duration

```python
simul_trip_Dist_Time_sample = abts.estimate_tripDist_Time(simul_trip_routes_sample, cbg, distance_ratio = average_distance_ratio, print_progress = True)

display(simul_trip_Dist_Time_sample[(simul_trip_Dist_Time_sample['uniqID'] == 7) & (simul_trip_Dist_Time_sample['Day_Type'] == 'Monday')])
```


|     |   uniqID | ageGroup   |     Home_cbg | Day_Type   |   sequence |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK | TRPPURP   | TRPPURP_det   |    Ws |     Wd |       Origin |         Dest |   TripDist | Trip_mode   |   TripSTARTT |   TripTime |   TripENDT |   start_min |   Dwell_Time |   end_min |
|----:|---------:|:-----------|-------------:|:-----------|-----------:|--------:|--------:|--------:|--------:|:----------|:--------------|------:|-------:|-------------:|-------------:|-----------:|:------------|-------------:|-----------:|-----------:|------------:|-------------:|----------:|
| 206 |        7 | MidAdult   | 550790005021 | Monday     |          1 |     0.7 |       1 |     1   |     1   | Home      | Home          | nan   | nan    | 550790005021 | 550790005021 |   0.552907 | nan         |          nan |        nan |        nan |           0 |          510 |       510 |
| 207 |        7 | MidAdult   | 550790005021 | Monday     |          2 |     0.7 |       1 |     1.2 |     0.7 | Work      | Work          |   1.5 |   0.5  | 550790005021 | 550790026001 |   7.75753  | Car         |          510 |         12 |        522 |         522 |          510 |      1032 |
| 208 |        7 | MidAdult   | 550790005021 | Monday     |          3 |     0.7 |       1 |     1   |     1   | S_d_r     | Religion      |   1.5 |   0.25 | 550790026001 | 550790001022 |   9.48366  | Car         |         1032 |         14 |       1046 |        1046 |            3 |      1049 |
| 209 |        7 | MidAdult   | 550790005021 | Monday     |          4 |     0.7 |       1 |     1   |     1   | Home      | Home          | nan   | nan    | 550790001022 | 550790005021 |   5.54624  | Car         |         1049 |          8 |       1057 |        1057 |          390 |      1447 |


#### 1.5.3. Optimize Trips with Logical and space-time constraints


```python
simul_trip_result_sample = abts.optimizeTrips_constraint(simul_trip_Dist_Time_sample, dwellTime_dict, print_progress = True)

display(simul_trip_result_sample[(simul_trip_result_sample['uniqID'] == 7) & (simul_trip_result_sample['Day_Type'] == 'Monday')])

```


|     |   uniqID | ageGroup   |     Home_cbg | Day_Type   |   sequence |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK | TRPPURP   | TRPPURP_det   |    Ws |     Wd |       Origin |         Dest |   TripDist | Trip_mode   |   TripSTARTT |   TripTime |   TripENDT |   start_min |   Dwell_Time |   end_min |
|----:|---------:|:-----------|-------------:|:-----------|-----------:|--------:|--------:|--------:|--------:|:----------|:--------------|------:|-------:|-------------:|-------------:|-----------:|:------------|-------------:|-----------:|-----------:|------------:|-------------:|----------:|
| 205 |        7 | MidAdult   | 550790005021 | Monday     |          1 |     0.7 |       1 |     1   |     1   | Home      | Home          | nan   | nan    | 550790005021 | 550790005021 |   0.552907 | nan         |          nan |        nan |        nan |           0 |          435 |       435 |
| 206 |        7 | MidAdult   | 550790005021 | Monday     |          2 |     0.7 |       1 |     1.2 |     0.7 | Work      | Work          |   1.5 |   0.5  | 550790005021 | 550790026001 |   7.75753  | Car         |          435 |         12 |        447 |         447 |          156 |       603 |
| 207 |        7 | MidAdult   | 550790005021 | Monday     |          3 |     0.7 |       1 |     1   |     1   | S_d_r     | Religion      |   1.5 |   0.25 | 550790026001 | 550790001022 |   9.48366  | Car         |          603 |         14 |        617 |         617 |            5 |       622 |
| 208 |        7 | MidAdult   | 550790005021 | Monday     |          4 |     0.7 |       1 |     1   |     1   | Home      | Home          | nan   | nan    | 550790001022 | 550790005021 |   5.54624  | Car         |          622 |          8 |        630 |         630 |          810 |      1440 |



## 2. Results (example)


<div style="text-align: center;">
    <img src="/ABTS/image/Result_2020_09_11.png" alt="Figure 2. Spatial patterns of travel counts in destination CBG in Milwaukee (2020/09~11)" width="600"/>
    <p>Figure 2. Spatial patterns of travel counts in destination CBG in Milwaukee (2020/09)</p>
</div>



<div style="text-align: center;">
    <img src="/ABTS/image/Result_Adult_MidAdult_Work_2020_09_11.png" alt="Figure 3. Spatial patterns of travel counts for work trips in destination CBG by Adult and Mid-adult agegroup (2020/09~11)" width="600"/>
    <p>Figure 3. Spatial patterns of travel counts for work trips in destination CBG by Adult and Mid-adult agegroup (2020/09~11)</p>
</div>




---

# Related Document: 
- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process

# Author

- **Author:** Moongi Choi
- **Email:** u1316663@utah.edu
