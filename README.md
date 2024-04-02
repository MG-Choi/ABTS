# ABTS
Agent Based Travel Scheduler library to predict individual travel schedules


## License
ABTS / version 0.0.3
- install:

```python
!pip install ABTS
import ABTS as abts
```

###### 이 알고리즘은 개개인의 travel schedule을 예측하는 시뮬레이션으로, ABTS (Agent-Based Travel Scheduler)이다. 파라미터 4가지를 이용하여 예측을 하며, 기존의 OD데이터를 이용하여 과거의 travel을 trip purpose, age 별로 decompose할 수 있고, 미래 예측 역시 가능하다. 모델에 대한 더 자세한 description은 다음의 두 논문에서 찾을 수 있다.

- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process


###### Figure 1 shows the framework of ABTS. ABTS utilizes a three-step process to generate the initial outcome of individual travel schedules: 1) Comprehensive Travel Classifier, 2) Land-Use Estimator, and 3) Individual Travel Schedule. The output of ABTS includes individual daily trip chains that consists of each trip purpose, occurrence time, destination, duration, and mode, tailored according to age group.

<div style="text-align: center;">
    <img src="/ABTS/image/Figure1_Framework_of_ABTS.png" alt="Figure1. Framework_of_ABTS" width="800"/>
    <p>Table 1. Classification in ABTS</p>
</div>


###### Comprehensive travel classifier에서는 Table 1과 같이 trip purpose와 trip mode, Age group을 각각의 기준으로 classify한다. 예로 trip purpose는 9개의 Major trip purpose와 13개의 Sub trip purpose로 나눠진다.

<div style="text-align: center;">
    <img src="/ABTS/image/Classification_method_in_ABTS.png" alt="Table 1. Classification in ABTS" width="650"/>
    <p>Table 1. Classification in ABTS</p>
</div>


###### Land-Use estimator에서는 landuse 데이터를 생성한다. 우리는 landuse를 Table 1의 Sub trip purpose와 같이 13개의 category로 구분하여 shp파일로 구축하였다. 데이터의 출처는 밑의 0. Preprocessing 단계에 기술되어 있다.

---


# Run the model with example data

###### 여기서는 Individual travel schedule generator의 과정을 예제 데이터를 통해 돌려보도록 한다. case area는 Milwaukee County이고 2020년 9월 과거의 travel을 trip purpose별, age별로 decompose하고, 예측하는 과정을 보여주겠다.


## 0. Preprocessing (using sample simulation in library)

###### 여기서는 ABTS모델을 run하기 위해 NHTS data와 SafeGraph data를 preprocessing하는 과정을 보여준다. 원본 데이터의 경우 밑에 기술된 두 홈페이지에서 다운받을 수 있으며, 본 example에서 사용되는 데이터 역시 해당 사이트에서 다운 받은 데이터이다. 하지만 SafeGraph Neighborhood 데이터의 경우, 현재 타 소스에서 제공되고 있는 것으로 보이며 여전히 접근 가능하다.

- NHTS data: 출처는 Federal HighWay Administration (FHWA)이며, 2017년의 National Household Travel Survey 데이터를 사용하였다 (https://nhts.ornl.gov/downloads). 

- SafeGraph data: 출처는 SafeGraph이며 데이터 이름은 Neighborhood patten이다. (https://docs.safegraph.com/docs/neighborhood-patterns). 이 데이터는 한달동안의 한 주를 랜덤으로 하여 한 주 동안의 사람들의 CBG별 이동을 OD로 나타낸 데이터로, 어떤 CBG에서 어떤 CBG로 얼마만큼의 사람이 이동했는지 기록되어있다. 단, trip purpose나 age 등의 정보는 없으며 단지 count만 수집되었다. 우리는 이 데이터를 이용해 확률모델을 만들어 사람들의 trip purpose별 Destination CBG를 선택하는 알고리즘에 적용하였다.


### 0.0. Data import

###### 구글 드라이브에 원본 데이터를 업로드 해 놓았다 (밑 링크). 여기 데이터에는 다음과 같은 데이터가 있다. https://drive.google.com/drive/folders/1YlsNPeTIC-0hjgYNnatuhmIGWQVSvg0-?usp=sharing 

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

- PERSONID_new: 새롭게 부여된 unique ID
- HHSTFIPS: Milwaukee code
- R_AGE_IMP: Age
- R_AGE_new: new classification of age group
- 나머지 column들에 대한 정보는 https://nhts.ornl.gov/downloads 여기서 제공하는 Format library 에서 알 수 있다.


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

###### W_d와 W_s의 조합들을 미리 많이 만들어놓은 후, 하나로 concatenate하는 과정. 이는 사용자가 W_s와 W_d의 조합을 얼마나 많이 만드느냐에 따라 다르기 때문에 코드를 만들어 놓지 않았다. 여기 예시데이터에서의 조합은 Ws = 0.5, 1.5, 1 / Wd = 0.25, 0.5 , 0.75 로 설정 하여 하나로 합쳐 'prob_2020_09_combined'의 이름으로 만들었다.



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

- output dataframe은 weekday와 weekend의 trip purpose별 각 destination으로 갈 확률을 exhibit한다. dataframe의 각 칸의 key는 destination CBG, value는 확률을 의미한다. 이 정해진 확률값은 추후 destination을 결정하는데 사용된다.


---


## 1. Execution for ABTS
###### 이 장에서는 위에서 가공된 preprocessed data를 가지고 ABTS를 run하려고 한다. preprocessed data의 경우 본 라이브러리 내에서 제공하므로 import하여 사용하면 된다. 

### 1.0. Data import

```python
import ABTS as abts

prob_2020_09_combined = abts.prob_2020_09_combined # Combined Probability of travels from O to D in Sep, 2020
repaired_NHTS = abts.repaired_NHTS # preprocessed NHTS
trip_mode = abts.trip_mode # trip mode
cbg = abts.cbg # CBG shp file
network_road = abts.network_road # road network data in Milwaukee

```
###### 데이터 import


### 1.1. Trip Occurence Builder
###### trip occurence builder는 ~~ ~ probability 부터


<div style="text-align: center">
    <img src="/ABTS/image/EQ1_prob_t_trip.png" alt="Equation 1. The probability of a person within age group ‘a’ having ‘t’ trip occurs in a single day." height="250"/>
    <p>Equation 1. The probability of a person within age group ‘a’ having ‘t’ trip occurs in a single day.</p>
</div>

###### probability를 기반으로 count함.

<div style="text-align: center">
    <img src="/ABTS/image/EQ2_count_t_trips.png" alt="Equation 2. Counting ‘t’ trips occurring ‘k’ times for a single individual ‘i’ in a day." height="225"/>
    <p>Equation 2. Counting ‘t’ trips occurring ‘k’ times for a single individual ‘i’ in a day.</p>
</div>



#### 1.1.0. Data staging
###### 데이터 staging은 한번만 시행해도 되는 것.


##### 1.1.0.1. The probability of a person with age ‘a’ having ‘t’ trips occurs in a single day

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


##### 1.1.0.2. The number of trips based on unique IDs, age, day type, and trip purpose derived from NHTS data

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




#### 1.1.1. The number of trips occurring ‘k’ times for a single individual ‘i’ in a day

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

eachTrip_simul_total_sample.head()
```

|     |   uniqID | ageGroup   | Day_Type   | Week_Type   | TRPPURP   |   count |     Home_cbg |   Wk_wD |   Wk_wK |   Wt_wD |   Wt_wK |
|----:|---------:|:-----------|:-----------|:------------|:----------|--------:|-------------:|--------:|--------:|--------:|--------:|
| 487 |        6 | MidAdult   | Friday     | Weekday     | S_d_r     |       0 | 550790005021 |     0.7 |       1 |     1   |     1   |
| 717 |       10 | Child      | Saturday   | Weekend     | Others    |       0 | 550790005021 |     0.7 |       1 |     1   |     1   |
| 687 |       10 | Child      | Friday     | Weekday     | Rec_lei   |       0 | 550790005021 |     0.7 |       1 |     1.1 |     1.1 |
| 878 |       13 | Teen       | Friday     | Weekday     | Serv_trip |       0 | 550790005021 |     0.7 |       1 |     1   |     1   |
| 639 |       11 | Child      | Wednesday  | Weekday     | Home      |       2 | 550790005021 |     0.7 |       1 |     1   |     1   |




### 1.2. Trip Chains Builder
###### trip chains builder에서는~

<div style="text-align: center;">
    <img src="/ABTS/image/EQ3_6_tripChain.png" alt="Equation 3-6. Simulated sequence 'S' reassignment based on the most similar origin sequence (O) for individual (i) within age group (a)" height="450"/>
    <p>Equation 3-6. Simulated sequence 'S' reassignment based on the most similar origin sequence 'O' for individual 'i' within age group 'a'</p>
</div>



#### 1.2.0. Data staging

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
    <img src="/ABTS/image/EQ7_dwellTime.png" alt="Equation 7. Dwell time of all trips for each individual" height="180"/>
    <p>Equation 7. Dwell time of all trips for each individual</p>
</div>


<div style="text-align: center;">
    <img src="/ABTS/image/EQ8_tripStartTime.png" alt="Equation 8. Trip start time for each individual" height="130"/>
    <p>Equation 8. Trip start time for each individual</p>
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
    <img src="/ABTS/image/EQ9_tripmode.png" alt="Equation 9. Assign Trip mode applying round trip" height="225"/>
    <p>Equation 9. Assign Trip mode applying round trip</p>
</div>



### 1.5. Spatial Trip Route Estimator

#### 1.5.0. Data staging

##### 1.5.0.1. Ratio between straight path and network path


#### 1.5.1. Estimate probabilistic destinations for trip purpose t


#### 1.5.2. Compute trip distance and duration



#### 1.5.3. Optimize Trips with Logical and space-time constraints





## 2. Results (example)






---

# Related Document: 
- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process

# Author

- **Author:** Moongi Choi
- **Email:** u1316663@utah.edu
