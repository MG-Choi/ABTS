# ABTS
Agent Based Travel Scheduler library to predict individual travel schedules


## License
ABTS / version 0.0.3
- install:

```python
!pip install ABTS
```

###### 이 알고리즘은 개개인의 travel schedule을 예측하는 시뮬레이션으로, ABTS (Agent-Based Travel Scheduler)이다. 파라미터 4가지를 이용하여 예측을 하며, 기존의 OD데이터를 이용하여 과거의 travel을 trip purpose, age 별로 decompose할 수 있고, 미래 예측 역시 가능하다. 모델에 대한 더 자세한 description은 다음의 두 논문에서 찾을 수 있다.

- Choi, M., & Hohl. A. (2024). Derivation of Spatiotemporal Risk Areas and Travel Behaviors During Pandemic Through Reverse Estimation of Mobility Patterns by Agent-Based Modeling, Spatial and Spatio-temporal Epidemiology. Under review (1st round)

- Choi, M., Seo, J., & Hohl. A. (2024). ABTS: Agent-Based Travel Scheduler. In process


###### Figure 1 shows the framework of ABTS. ABTS utilizes a three-step process to generate the initial outcome of individual travel schedules: 1) Comprehensive Travel Classifier, 2) Land-Use Estimator, and 3) Individual Travel Schedule. The output of ABTS includes individual daily trip chains that consists of each trip purpose, occurrence time, destination, duration, and mode, tailored according to age group.

![Figure1. Framework_of_ABTS](/ABTS/image/Figure1_Framework_of_ABTS.png)

###### Comprehensive travel classifier에서는 Table 1과 같이 trip purpose와 trip mode, Age group을 각각의 기준으로 classify한다. 예로 trip purpose는 9개의 Major trip purpose와 13개의 Sub trip purpose로 나눠진다.

<div style="text-align: center;">
    <img src="/ABTS/image/Classification_method_in_ABTS.png" alt="Table 1. Classification in ABTS" width="650"/>
</div>


###### Land-Use estimator에서는 landuse 데이터를 생성한다. 우리는 landuse를 Table 1의 Sub trip purpose와 같이 13개의 category로 구분하여 shp파일로 구축하였다. 데이터의 출처는 밑의 0. Preprocessing 단계에 기술되어 있다.



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
trippub_organized = organize_columns(trippub, print_progress = True)
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
repaired_NHTS = preprocess_NHTS(trippub_organized, print_progress = True) # will take long time
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
trip_mode_prop_all = preprocess_NHTS_tripMode(trippub_organized) 
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
neighbor_2020_09_OD = DOtoOD(neighbor_2020_09, print_progress = True)
display(neighbor_2020_09_OD.heaD())
```

|    |         area | work_behavior_from_area                                                                                            | weekday_device_from_area_home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | weekend_device_from_area_home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|---:|-------------:|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | 550791854002 | {'550790042003': 4, '550790042002': 4}                                                                             | {'550791854002': 6, '550790201003': 4, '550790084002': 4, '550790032001': 4, '550791101002': 4, '550790144002': 4, '550790092001': 4, '550790099002': 4, '550790001012': 4, '550791861001': 4, '550790036001': 4, '550790113002': 4, '550790098001': 4, '550790044001': 4, '550790060002': 4, '550791856001': 4, '550791870002': 4, '550790046004': 4, '550790137001': 4, '550791205022': 4, '550790112002': 4, '550790068001': 4, '550790200001': 4, '550790099001': 4}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | {'550791854002': 4, '550791002003': 4, '550790091002': 4, '550790144002': 4, '550790005013': 4, '550790059003': 4, '550790077002': 4, '550790087001': 4, '550790086002': 4, '550791853001': 4, '550791101003': 4, '550790906001': 4, '550790113001': 4, '550790166001': 4}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|  1 | 550790033002 | {'550790040003': 4, '550790019004': 4, '550790005022': 4, '550790002011': 4, '550791853001': 4, '550790076003': 4} | {'550790033002': 5, '550790051002': 4, '550790039003': 4, '550790910002': 4, '550790065003': 4, '550790034001': 12, '550790001021': 4, '550790003021': 4, '550790052002': 4, '550791201022': 4, '550790031004': 4, '550790050003': 4, '550790501021': 5, '550791001002': 4, '550790063003': 4, '550790149002': 4, '550790912002': 4, '550790002021': 4, '550790602001': 5, '550790024003': 5, '550790010001': 4, '550790059001': 5, '550790601012': 4, '550791874001': 4, '550790501024': 4, '550790160002': 4, '550790051003': 5, '550790035004': 11, '550790011001': 4, '550790902002': 14, '550790006004': 5, '550791003001': 4, '550790003042': 4, '550790004001': 4, '550790034004': 4, '550790005013': 20, '550790059003': 5, '550790005022': 4, '550790033003': 6, '550790006005': 5, '550790002011': 5, '550791861001': 4, '550790038002': 4, '550790017003': 11, '550790035003': 4, '550790001022': 4, '550791002001': 4, '550790126002': 4, '550790058002': 4, '550790036002': 4, '550790036001': 8, '550791865001': 4, '550790903001': 13, '550790141001': 7, '550790017005': 4, '550790044001': 4, '550790016002': 4, '550790005023': 4, '550790912003': 4, '550790111002': 4, '550790021001': 4, '550790183001': 4, '550790063001': 4, '550790033004': 4, '550791008002': 4, '550790054002': 4, '550791854001': 4, '550790002023': 4, '550791853001': 12, '550790059002': 4, '550791014001': 4, '550790031003': 4, '550790066002': 4, '550799800001': 4, '550790007003': 4, '550790020001': 4, '550791001003': 4, '550790217005': 4, '550790048003': 4, '550791004003': 4, '550790910004': 4, '550790906001': 4, '550790005011': 4, '550791853002': 4, '550790058003': 4, '550790051001': 5, '550790017004': 4, '550791602033': 4, '550790041001': 4, '550791205013': 4, '550791011001': 4, '550791101001': 9, '550790901003': 6, '550790092002': 4, '550790010004': 5} | {'550791854002': 4, '550791009002': 5, '550790051002': 4, '550790034001': 13, '550790024001': 4, '550791501004': 4, '550791101002': 4, '550790055001': 4, '550790149002': 4, '550790912002': 4, '550790602001': 4, '550790024003': 8, '550790501011': 4, '550791865002': 4, '550790035004': 4, '550790902002': 12, '550790005013': 6, '550790059003': 4, '550790005022': 4, '550790001012': 4, '550790033003': 4, '550790049002': 4, '550790006005': 4, '550790038001': 4, '550790005021': 4, '550790017003': 10, '550790001022': 4, '550791002001': 4, '550791201021': 4, '550791855002': 4, '550790036001': 4, '550790903001': 7, '550790004002': 5, '550790044001': 4, '550790352001': 4, '550791007003': 4, '550790602003': 4, '550790035001': 4, '550790033004': 4, '550790055004': 4, '550791860001': 4, '550791868001': 4, '550791853001': 4, '550791101003': 4, '550790013001': 4, '550790029001': 4, '550790031003': 4, '550791001003': 5, '550790048003': 4, '550790167001': 4, '550790051001': 4, '550790089002': 4, '550790901003': 4, '550791010001': 4, '550790200001': 4} |

- area: Census Block Groups (CBG), origin home area
- work_behavior_from_area: # people trip to workplace from 'area' (Workplace: # people)
- weekday_device_from_area_home: # people trip from 'area' to certain CBG in weekday (certain CBG: # people)
- weekend_device_from_area_home: # people trip from 'area' to certain CBG in weekend


#### 0.2.2. Computing probability of trips from origin cbg to dest cbg


```python
prob_trips_2020_09_ws05_wd025 = compute_probabilityByk_Ws_Wd(neighbor_2020_09_OD, landUse, W_s = 0.5, W_d = 0.25)
display(prob_trips_2020_09_ws05_wd025.head())
```

여기 할 차례


#### 0.2.3. Combine all probability tables by their month

여기에는 concatenate으로 해라 이정도만 쓰기.


#### 0.2.4. fill empty probability


```python
prob_2020_09_combined = prob_2020_09_combined.apply(fill_values, axis = 1)
prob_2020_09_combined.to_csv('[yourPath]' + 'prob_2020_09_combined.xlsx') # save
display(prob_2020_09_combined.head())
```





## 1. Execution for ABTS

### 1.0. Data import

###### 데이터 import


### 1.1. Trip Occurence Builder


#### 1.1.0. Data staging
###### 데이터 staging은 한번만 시행해도 되는 것.

##### 1.1.0.1. The probability of a person with age ‘a’ having ‘t’ trips occurs in a single day


##### 1.1.0.2. The number of trips based on unique IDs, age, day type, and trip purpose derived from NHTS data



#### 1.1.1. The number of trips occurring ‘k’ times for a single individual ‘i’ in a day





### 1.2. Trip Chains Builder

#### 1.2.0. Data staging

##### 1.2.0.1. Create trip seqeunce of individuals using origin NHTS data

#### 1.2.1. Finding optimal origin sequence O<i><sub>i</sub></i> most similar to S<i><sub>i</sub></i>
#### 1.2.2. Randomly assign the trip sequence for S<i><sub>i</sub></i>
#### 1.2.3. Reassign the sequence of trip in S<i><sub>i</sub></i> based on O<i><sub>i</sub></i>

###### 여기에는 하나의 equation



### 1.3. Trip Timing Estimator

#### 1.3.0. Data staging

##### 1.3.0.1. Extracting dwell time by trip purpose using NHTS


##### 1.3.0.2. Extracting trip start time by trip purpose using NHTS


#### 1.3.1. Estimating dwell time
#### 1.3.2. Estimating trip start time¶

###### 여기에는 하나의 equation


### 1.4. Trip Mode Assigner



### 1.5. Spatial Trip Route Estimator

#### 1.5.0. Data staging

##### 1.5.0.1. Ratio between straight path and network path


#### 1.5.1. Estimate probabilistic destinations for trip purpose t


#### 1.5.2. Compute trip distance and duration



#### 1.5.3. Optimize Trips with Logical and space-time constraints





## 2. Results (example)