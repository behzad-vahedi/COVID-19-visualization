## Mapping Spatial and Temporal Patterns of COVID-19 Cases and Deaths in the Contiguous US 

County Level COVID-19  data are downloaded from usafacts.org on May 2nd, 2020. 

created by Behzad Vahedi (vahedi.behzad@gmail.com)


```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify as mc
import imageio
```


```python
import geoplot as gplt
import geoplot.crs as gcrs
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
%matplotlib inline
```


```python
# county boudaries downloaded from arcgis.com

counties = gpd.read_file('./contiguous_counties/contiguous_counties.shp') #(3108, 56)
counties.drop(counties.columns.difference(['NAME','STATE_NAME','STATE_FIPS','CNTY_FIPS','FIPS','POPULATION','Shape_Leng','Shape_Area','geometry']), 1, inplace=True)
counties.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>STATE_NAME</th>
      <th>STATE_FIPS</th>
      <th>CNTY_FIPS</th>
      <th>FIPS</th>
      <th>POPULATION</th>
      <th>Shape_Leng</th>
      <th>Shape_Area</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Autauga</td>
      <td>Alabama</td>
      <td>01</td>
      <td>001</td>
      <td>01001</td>
      <td>56903</td>
      <td>1.884137</td>
      <td>0.148903</td>
      <td>POLYGON ((-86.82067 32.34731, -86.81446 32.370...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Baldwin</td>
      <td>Alabama</td>
      <td>01</td>
      <td>003</td>
      <td>01003</td>
      <td>214651</td>
      <td>3.678276</td>
      <td>0.404489</td>
      <td>POLYGON ((-87.97309 31.16482, -87.93710 31.173...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Barbour</td>
      <td>Alabama</td>
      <td>01</td>
      <td>005</td>
      <td>01005</td>
      <td>26585</td>
      <td>2.218514</td>
      <td>0.222431</td>
      <td>POLYGON ((-85.74337 31.62624, -85.71720 31.679...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Bibb</td>
      <td>Alabama</td>
      <td>01</td>
      <td>007</td>
      <td>01007</td>
      <td>23003</td>
      <td>1.852453</td>
      <td>0.157736</td>
      <td>POLYGON ((-87.41986 33.01177, -87.31532 33.012...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Blount</td>
      <td>Alabama</td>
      <td>01</td>
      <td>009</td>
      <td>01009</td>
      <td>57971</td>
      <td>2.067456</td>
      <td>0.167530</td>
      <td>POLYGON ((-86.96799 33.86045, -86.92667 33.872...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# death and cases data downloaded from usafacts.org

cases = pd.read_csv('covid_confirmed.csv') # (3195, 106)
deaths = pd.read_csv('covid_deaths.csv') # (3195, 106)
cases.rename(columns = {'countyFIPS':'FIPS'}, inplace = True) 
deaths.rename(columns = {'countyFIPS':'FIPS'}, inplace = True)

```


```python
# convert FIPS values from object to int64
def parse(x):
    try:
        return int(x)
    except ValueError:
        return np.nan

counties['FIPS'] = counties['FIPS'].apply(parse)
```


```python
# GeoDataFrames of cummulative number of cases and deaths pre county (one column per day)  
case_per_county = counties.merge(cases, on='FIPS') # shape: (3108, 114)
death_per_county = counties.merge(deaths, on='FIPS') #shape: (3108, 114)


# GeoDataFrames of number of NEW cases and deaths per county (one column per day) 
case_per_day = case_per_county.copy()
case_per_day.iloc[:,-102:]  = case_per_day.iloc[:,-102:] .diff(axis=1) #shape (3108, 114)

death_per_day = death_per_county.copy()
death_per_day.iloc[:,-102:] = death_per_day.iloc[:,-102:].diff(axis=1) #shape (3108, 114)
```

### Mapping number of new cases per day per county


```python
'''
classify data based on a modified version of natural break method into 7 classes
used for choropleth map classifiacation
The NaturalBreak calssification is modified to hightilght the heavy head of the data (smaller values)
''' 
schme_NB = mc.NaturalBreaks(case_per_day.iloc[:,-101:],7) 
bins = schme_NB.bins
bins = np.insert(bins, 0, [0.9,3.0])
bins = np.delete(bins,5)
scheme_CPD = mc.UserDefined(case_per_day.iloc[:,-101:].to_numpy().flatten(),bins)
```


```python
# map each day, using number of cases per day per county as attribue value
counter = 0 
case_images = []

for col in case_per_day.iloc[:,-101:].columns:    

    gplt.choropleth(
        case_per_day, hue=col, 
        projection=gcrs.WebMercator(), 
        scheme=scheme_CPD,
        cmap='OrRd', 
        legend=True, 
        legend_kwargs={'frameon': False, 'loc': 'lower right','fontsize':'13'},
        figsize=(15, 15))
    plt.title('Number of Confirmed COVID-19 Cases Per Day \n' + col, fontsize=18)
    filename = './figs/case_per_day/snapshot_'+ str(counter).zfill(3) + '.png'
    plt.savefig(filename)
    case_images.append(imageio.imread(filename))
    counter += 1

imageio.mimsave('./figs/case_per_day/confirmed_cases.gif', case_images)
```


![png](output_11_0.png)



![png](output_11_1.png)



![png](output_11_2.png)



![png](output_11_3.png)



![png](output_11_4.png)



![png](output_11_5.png)



![png](output_11_6.png)



![png](output_11_7.png)



![png](output_11_8.png)



![png](output_11_9.png)



![png](output_11_10.png)



![png](output_11_11.png)



![png](output_11_12.png)



![png](output_11_13.png)



![png](output_11_14.png)



![png](output_11_15.png)



![png](output_11_16.png)



![png](output_11_17.png)



![png](output_11_18.png)



![png](output_11_19.png)



![png](output_11_20.png)



![png](output_11_21.png)



![png](output_11_22.png)



![png](output_11_23.png)



![png](output_11_24.png)



![png](output_11_25.png)



![png](output_11_26.png)



![png](output_11_27.png)



![png](output_11_28.png)



![png](output_11_29.png)



![png](output_11_30.png)



![png](output_11_31.png)



![png](output_11_32.png)



![png](output_11_33.png)



![png](output_11_34.png)



![png](output_11_35.png)



![png](output_11_36.png)



![png](output_11_37.png)



![png](output_11_38.png)



![png](output_11_39.png)



![png](output_11_40.png)



![png](output_11_41.png)



![png](output_11_42.png)



![png](output_11_43.png)



![png](output_11_44.png)



![png](output_11_45.png)



![png](output_11_46.png)



![png](output_11_47.png)



![png](output_11_48.png)



![png](output_11_49.png)



![png](output_11_50.png)



![png](output_11_51.png)



![png](output_11_52.png)



![png](output_11_53.png)



![png](output_11_54.png)



![png](output_11_55.png)



![png](output_11_56.png)



![png](output_11_57.png)



![png](output_11_58.png)



![png](output_11_59.png)



![png](output_11_60.png)



![png](output_11_61.png)



![png](output_11_62.png)



![png](output_11_63.png)



![png](output_11_64.png)



![png](output_11_65.png)



![png](output_11_66.png)



![png](output_11_67.png)



![png](output_11_68.png)



![png](output_11_69.png)



![png](output_11_70.png)



![png](output_11_71.png)



![png](output_11_72.png)



![png](output_11_73.png)



![png](output_11_74.png)



![png](output_11_75.png)



![png](output_11_76.png)



![png](output_11_77.png)



![png](output_11_78.png)



![png](output_11_79.png)



![png](output_11_80.png)



![png](output_11_81.png)



![png](output_11_82.png)



![png](output_11_83.png)



![png](output_11_84.png)



![png](output_11_85.png)



![png](output_11_86.png)



![png](output_11_87.png)



![png](output_11_88.png)



![png](output_11_89.png)



![png](output_11_90.png)



![png](output_11_91.png)



![png](output_11_92.png)



![png](output_11_93.png)



![png](output_11_94.png)



![png](output_11_95.png)



![png](output_11_96.png)



![png](output_11_97.png)



![png](output_11_98.png)



![png](output_11_99.png)



![png](output_11_100.png)


### Mapping cumulative number of total cases per county


```python
# normalizing number of cases (cumulative) by 100k population
case_per_county_normalized = case_per_county.copy()
case_per_county_normalized.iloc[:,-102:] = case_per_county_normalized.iloc[:,-102:].divide(
    case_per_county_normalized['POPULATION']/100000,axis=0)

# classify data based on a modified version of natural break method into 7 classes (used for choropleth map classifiacation)
# The NaturalBreak calssification is modified to hightilght the heavy head of the data (smaller values)
schme_NB = mc.NaturalBreaks(case_per_county_normalized.iloc[:,-102:],6) 
bins = schme_NB.bins
bins = np.insert(bins, 0, [0.03,28.0])
scheme_CPC = mc.UserDefined(case_per_county_normalized.iloc[:,-102:].to_numpy().flatten(),bins)
```


```python
counter = 0
case_images = []

for col in case_per_county_normalized.iloc[:,-102:].columns:    

    gplt.choropleth(
        case_per_county_normalized, hue=col, 
        projection=gcrs.WebMercator(), 
        scheme=scheme_CPC,
        cmap='OrRd', 
        legend=True, 
        legend_kwargs={'title': 'Incidence Rate','frameon': False, 'loc': 'lower right','fontsize':'13'},
        figsize=(15, 15))
    
    plt.title('Number of Confirmed COVID-19 Cases Per 100k Population \n by: Behzad Vahedi \n' + col, fontsize=18)
    filename = './figs/case_per_county/snapshot_'+ str(counter).zfill(3) + '.png'
    plt.savefig(filename)
    case_images.append(imageio.imread(filename))
    counter += 1

imageio.mimsave('./figs/case_per_county/cases_per_county.gif', case_images)

```


![png](output_14_0.png)



![png](output_14_1.png)



![png](output_14_2.png)



![png](output_14_3.png)



![png](output_14_4.png)



![png](output_14_5.png)



![png](output_14_6.png)



![png](output_14_7.png)



![png](output_14_8.png)



![png](output_14_9.png)



![png](output_14_10.png)



![png](output_14_11.png)



![png](output_14_12.png)



![png](output_14_13.png)



![png](output_14_14.png)



![png](output_14_15.png)



![png](output_14_16.png)



![png](output_14_17.png)



![png](output_14_18.png)



![png](output_14_19.png)



![png](output_14_20.png)



![png](output_14_21.png)



![png](output_14_22.png)



![png](output_14_23.png)



![png](output_14_24.png)



![png](output_14_25.png)



![png](output_14_26.png)



![png](output_14_27.png)



![png](output_14_28.png)



![png](output_14_29.png)



![png](output_14_30.png)



![png](output_14_31.png)



![png](output_14_32.png)



![png](output_14_33.png)



![png](output_14_34.png)



![png](output_14_35.png)



![png](output_14_36.png)



![png](output_14_37.png)



![png](output_14_38.png)



![png](output_14_39.png)



![png](output_14_40.png)



![png](output_14_41.png)



![png](output_14_42.png)



![png](output_14_43.png)



![png](output_14_44.png)



![png](output_14_45.png)



![png](output_14_46.png)



![png](output_14_47.png)



![png](output_14_48.png)



![png](output_14_49.png)



![png](output_14_50.png)



![png](output_14_51.png)



![png](output_14_52.png)



![png](output_14_53.png)



![png](output_14_54.png)



![png](output_14_55.png)



![png](output_14_56.png)



![png](output_14_57.png)



![png](output_14_58.png)



![png](output_14_59.png)



![png](output_14_60.png)



![png](output_14_61.png)



![png](output_14_62.png)



![png](output_14_63.png)



![png](output_14_64.png)



![png](output_14_65.png)



![png](output_14_66.png)



![png](output_14_67.png)



![png](output_14_68.png)



![png](output_14_69.png)



![png](output_14_70.png)



![png](output_14_71.png)



![png](output_14_72.png)



![png](output_14_73.png)



![png](output_14_74.png)



![png](output_14_75.png)



![png](output_14_76.png)



![png](output_14_77.png)



![png](output_14_78.png)



![png](output_14_79.png)



![png](output_14_80.png)



![png](output_14_81.png)



![png](output_14_82.png)



![png](output_14_83.png)



![png](output_14_84.png)



![png](output_14_85.png)



![png](output_14_86.png)



![png](output_14_87.png)



![png](output_14_88.png)



![png](output_14_89.png)



![png](output_14_90.png)



![png](output_14_91.png)



![png](output_14_92.png)



![png](output_14_93.png)



![png](output_14_94.png)



![png](output_14_95.png)



![png](output_14_96.png)



![png](output_14_97.png)



![png](output_14_98.png)



![png](output_14_99.png)



![png](output_14_100.png)



![png](output_14_101.png)


### Mapping cumulative number of deaths per county


```python
# classify data based on a modified version of natural break method into 7 classes (used for choropleth map classifiacation)
# The NaturalBreak calssification is modified to hightilght the heavy head of the data (smaller values)
schme_NB = mc.NaturalBreaks(death_per_county.iloc[:,-102:],6) 
bins = schme_NB.bins
bins = np.insert(bins, 0, [0.9, 3.3])
scheme_DPC = mc.UserDefined(death_per_county.iloc[:,-102:].to_numpy().flatten(),bins)
```


```python
counter = 0

for col in death_per_county.iloc[:,-102:].columns:    

    gplt.choropleth(
        death_per_county, hue=col, 
        projection=gcrs.WebMercator(), 
        scheme=scheme_DPC,
        cmap='OrRd', 
        legend=True, 
        legend_kwargs={'title': 'Number of Deaths','frameon': False, 'loc': 'lower right','fontsize':'13'},
        figsize=(15, 15))
    
    plt.title('Number of COVID-19 Death Per County \n by: Behzad Vahedi \n' + col, fontsize=18)
    filename = './figs/death_per_county/snapshot_'+ str(counter).zfill(3) + '.png'
    plt.savefig(filename)
#     case_images.append(imageio.imread(filename))
    counter += 1

# imageio.mimsave('./figs/death_per_county/cases_per_county.gif', case_images)
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)



![png](output_17_6.png)



![png](output_17_7.png)



![png](output_17_8.png)



![png](output_17_9.png)



![png](output_17_10.png)



![png](output_17_11.png)



![png](output_17_12.png)



![png](output_17_13.png)



![png](output_17_14.png)



![png](output_17_15.png)



![png](output_17_16.png)



![png](output_17_17.png)



![png](output_17_18.png)



![png](output_17_19.png)



![png](output_17_20.png)



![png](output_17_21.png)



![png](output_17_22.png)



![png](output_17_23.png)



![png](output_17_24.png)



![png](output_17_25.png)



![png](output_17_26.png)



![png](output_17_27.png)



![png](output_17_28.png)



![png](output_17_29.png)



![png](output_17_30.png)



![png](output_17_31.png)



![png](output_17_32.png)



![png](output_17_33.png)



![png](output_17_34.png)



![png](output_17_35.png)



![png](output_17_36.png)



![png](output_17_37.png)



![png](output_17_38.png)



![png](output_17_39.png)



![png](output_17_40.png)



![png](output_17_41.png)



![png](output_17_42.png)



![png](output_17_43.png)



![png](output_17_44.png)



![png](output_17_45.png)



![png](output_17_46.png)



![png](output_17_47.png)



![png](output_17_48.png)



![png](output_17_49.png)



![png](output_17_50.png)



![png](output_17_51.png)



![png](output_17_52.png)



![png](output_17_53.png)



![png](output_17_54.png)



![png](output_17_55.png)



![png](output_17_56.png)



![png](output_17_57.png)



![png](output_17_58.png)



![png](output_17_59.png)



![png](output_17_60.png)



![png](output_17_61.png)



![png](output_17_62.png)



![png](output_17_63.png)



![png](output_17_64.png)



![png](output_17_65.png)



![png](output_17_66.png)



![png](output_17_67.png)



![png](output_17_68.png)



![png](output_17_69.png)



![png](output_17_70.png)



![png](output_17_71.png)



![png](output_17_72.png)



![png](output_17_73.png)



![png](output_17_74.png)



![png](output_17_75.png)



![png](output_17_76.png)



![png](output_17_77.png)



![png](output_17_78.png)



![png](output_17_79.png)



![png](output_17_80.png)



![png](output_17_81.png)



![png](output_17_82.png)



![png](output_17_83.png)



![png](output_17_84.png)



![png](output_17_85.png)



![png](output_17_86.png)



![png](output_17_87.png)



![png](output_17_88.png)



![png](output_17_89.png)



![png](output_17_90.png)



![png](output_17_91.png)



![png](output_17_92.png)



![png](output_17_93.png)



![png](output_17_94.png)



![png](output_17_95.png)



![png](output_17_96.png)



![png](output_17_97.png)



![png](output_17_98.png)



![png](output_17_99.png)



![png](output_17_100.png)



![png](output_17_101.png)

