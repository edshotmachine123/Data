```python
import pandas as pd
import numpy as np
```


```python

df1 = pd.read_csv('nyc_weather_2018.csv')
```


```python
df2 = pd.read_csv('weather_stations.csv')
```


```python
df1.head()
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
      <th>date</th>
      <th>datatype</th>
      <th>station</th>
      <th>attributes</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01T00:00:00</td>
      <td>PRCP</td>
      <td>GHCND:US1CTFR0039</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01T00:00:00</td>
      <td>PRCP</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01T00:00:00</td>
      <td>PRCP</td>
      <td>GHCND:US1NJBG0017</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0017</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
weather = df1[df1.datatype=="SNOW"]
```


```python
weather
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
      <th>date</th>
      <th>datatype</th>
      <th>station</th>
      <th>attributes</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0017</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0018</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0023</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0039</td>
      <td>,,N,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78699</th>
      <td>2018-12-31T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:USC00308577</td>
      <td>,,7,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>78703</th>
      <td>2018-12-31T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:USW00014732</td>
      <td>,,W,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>78715</th>
      <td>2018-12-31T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:USW00014734</td>
      <td>,,W,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>78744</th>
      <td>2018-12-31T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:USW00094728</td>
      <td>,,W,</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>78769</th>
      <td>2018-12-31T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:USW00094789</td>
      <td>,,W,</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>12834 rows × 5 columns</p>
</div>




```python
df2.head()
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
      <th>id</th>
      <th>name</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GHCND:US1CTFR0022</td>
      <td>STAMFORD 2.6 SSW, CT US</td>
      <td>41.064100</td>
      <td>-73.577000</td>
      <td>36.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GHCND:US1CTFR0039</td>
      <td>STAMFORD 4.2 S, CT US</td>
      <td>41.037788</td>
      <td>-73.568176</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GHCND:US1NJBG0001</td>
      <td>BERGENFIELD 0.3 SW, NJ US</td>
      <td>40.921298</td>
      <td>-74.001983</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GHCND:US1NJBG0002</td>
      <td>SADDLE BROOK TWP 0.6 E, NJ US</td>
      <td>40.902694</td>
      <td>-74.083358</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GHCND:US1NJBG0003</td>
      <td>TENAFLY 1.3 W, NJ US</td>
      <td>40.914670</td>
      <td>-73.977500</td>
      <td>21.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
station = df2[df2.name.str.contains("NJ US")]
station
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
      <th>id</th>
      <th>name</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>GHCND:US1NJBG0001</td>
      <td>BERGENFIELD 0.3 SW, NJ US</td>
      <td>40.921298</td>
      <td>-74.001983</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GHCND:US1NJBG0002</td>
      <td>SADDLE BROOK TWP 0.6 E, NJ US</td>
      <td>40.902694</td>
      <td>-74.083358</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GHCND:US1NJBG0003</td>
      <td>TENAFLY 1.3 W, NJ US</td>
      <td>40.914670</td>
      <td>-73.977500</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GHCND:US1NJBG0005</td>
      <td>WESTWOOD 0.8 ESE, NJ US</td>
      <td>40.983041</td>
      <td>-74.015858</td>
      <td>15.8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GHCND:US1NJBG0006</td>
      <td>RAMSEY 0.6 E, NJ US</td>
      <td>41.058611</td>
      <td>-74.134068</td>
      <td>112.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>217</th>
      <td>GHCND:USC00289455</td>
      <td>WESTFIELD, NJ US</td>
      <td>40.650000</td>
      <td>-74.350000</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>218</th>
      <td>GHCND:USC00289832</td>
      <td>WOODCLIFF LAKE, NJ US</td>
      <td>41.013900</td>
      <td>-74.042500</td>
      <td>31.4</td>
    </tr>
    <tr>
      <th>271</th>
      <td>GHCND:USW00014734</td>
      <td>NEWARK LIBERTY INTERNATIONAL AIRPORT, NJ US</td>
      <td>40.682500</td>
      <td>-74.169400</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>273</th>
      <td>GHCND:USW00054743</td>
      <td>CALDWELL ESSEX CO AIRPORT, NJ US</td>
      <td>40.876390</td>
      <td>-74.283060</td>
      <td>52.7</td>
    </tr>
    <tr>
      <th>276</th>
      <td>GHCND:USW00094741</td>
      <td>TETERBORO AIRPORT, NJ US</td>
      <td>40.850000</td>
      <td>-74.061390</td>
      <td>2.7</td>
    </tr>
  </tbody>
</table>
<p>164 rows × 5 columns</p>
</div>




```python
df3=weather.merge(station.rename(dict(id='station'),axis=1),on='station')
```


```python
df3
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
      <th>date</th>
      <th>datatype</th>
      <th>station</th>
      <th>attributes</th>
      <th>value</th>
      <th>name</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
      <td>40.791492</td>
      <td>-74.13979</td>
      <td>17.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
      <td>40.791492</td>
      <td>-74.13979</td>
      <td>17.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
      <td>40.791492</td>
      <td>-74.13979</td>
      <td>17.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>229.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
      <td>40.791492</td>
      <td>-74.13979</td>
      <td>17.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-06T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
      <td>40.791492</td>
      <td>-74.13979</td>
      <td>17.7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8342</th>
      <td>2018-12-26T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJMD0088</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>METUCHEN 3.3 N, NJ US</td>
      <td>40.588840</td>
      <td>-74.37212</td>
      <td>27.4</td>
    </tr>
    <tr>
      <th>8343</th>
      <td>2018-12-27T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJMD0088</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>METUCHEN 3.3 N, NJ US</td>
      <td>40.588840</td>
      <td>-74.37212</td>
      <td>27.4</td>
    </tr>
    <tr>
      <th>8344</th>
      <td>2018-12-31T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJMD0088</td>
      <td>,,N,</td>
      <td>0.0</td>
      <td>METUCHEN 3.3 N, NJ US</td>
      <td>40.588840</td>
      <td>-74.37212</td>
      <td>27.4</td>
    </tr>
    <tr>
      <th>8345</th>
      <td>2018-12-24T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJES0040</td>
      <td>T,,N,</td>
      <td>0.0</td>
      <td>VERONA TWP 0.8 W, NJ US</td>
      <td>40.832600</td>
      <td>-74.25830</td>
      <td>161.2</td>
    </tr>
    <tr>
      <th>8346</th>
      <td>2018-12-30T00:00:00</td>
      <td>SNOW</td>
      <td>GHCND:US1NJES0040</td>
      <td>T,,N,</td>
      <td>0.0</td>
      <td>VERONA TWP 0.8 W, NJ US</td>
      <td>40.832600</td>
      <td>-74.25830</td>
      <td>161.2</td>
    </tr>
  </tbody>
</table>
<p>8347 rows × 9 columns</p>
</div>




```python
df3.shape
```




    (8347, 9)




```python
df3['station'].unique().shape
```




    (66,)




```python
df3.drop_duplicates('station').mean()
```




    value        16.333333
    latitude     40.728933
    longitude   -74.257065
    elevation    58.348485
    dtype: float64




```python
ex2
```


```python
df4 = pd.read_csv('fb_2018.csv')
```


```python
df4.head()
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-02</td>
      <td>177.68</td>
      <td>181.58</td>
      <td>177.5500</td>
      <td>181.42</td>
      <td>18151903</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-03</td>
      <td>181.88</td>
      <td>184.78</td>
      <td>181.3300</td>
      <td>184.67</td>
      <td>16886563</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-04</td>
      <td>184.90</td>
      <td>186.21</td>
      <td>184.0996</td>
      <td>184.33</td>
      <td>13880896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-05</td>
      <td>185.59</td>
      <td>186.90</td>
      <td>184.9300</td>
      <td>186.85</td>
      <td>13574535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-08</td>
      <td>187.20</td>
      <td>188.90</td>
      <td>186.3300</td>
      <td>188.28</td>
      <td>17994726</td>
    </tr>
  </tbody>
</table>
</div>




```python
df5 = df4.assign(pct_changemethod = lambda x : x.volume.pct_change())
```


```python
df6 = df5.sort_values(by='pct_changemethod',ascending = False)
```


```python
df6.head()
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
      <th>date</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>pct_changemethod</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>2018-01-12</td>
      <td>178.06</td>
      <td>181.48</td>
      <td>177.40</td>
      <td>179.37</td>
      <td>77551299</td>
      <td>7.087876</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2018-03-19</td>
      <td>177.01</td>
      <td>177.17</td>
      <td>170.06</td>
      <td>172.56</td>
      <td>88140060</td>
      <td>2.611789</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2018-07-26</td>
      <td>174.89</td>
      <td>180.13</td>
      <td>173.75</td>
      <td>176.26</td>
      <td>169803668</td>
      <td>1.628841</td>
    </tr>
    <tr>
      <th>182</th>
      <td>2018-09-21</td>
      <td>166.64</td>
      <td>167.25</td>
      <td>162.81</td>
      <td>162.93</td>
      <td>45994800</td>
      <td>1.428956</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2018-03-26</td>
      <td>160.82</td>
      <td>161.10</td>
      <td>149.02</td>
      <td>160.06</td>
      <td>126116634</td>
      <td>1.352496</td>
    </tr>
  </tbody>
</table>
</div>




```python
ex3
```


```python
df7 = pd.read_csv('earthquakes.csv')
```


```python
df7.head()
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
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>detail</th>
      <th>dmin</th>
      <th>felt</th>
      <th>gap</th>
      <th>ids</th>
      <th>mag</th>
      <th>magType</th>
      <th>...</th>
      <th>sources</th>
      <th>status</th>
      <th>time</th>
      <th>title</th>
      <th>tsunami</th>
      <th>type</th>
      <th>types</th>
      <th>tz</th>
      <th>updated</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>37389218</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.008693</td>
      <td>NaN</td>
      <td>85.0</td>
      <td>,ci37389218,</td>
      <td>1.35</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539475168010</td>
      <td>M 1.4 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,</td>
      <td>-480.0</td>
      <td>1539475395144</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>37389202</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.020030</td>
      <td>NaN</td>
      <td>79.0</td>
      <td>,ci37389202,</td>
      <td>1.29</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539475129610</td>
      <td>M 1.3 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,</td>
      <td>-480.0</td>
      <td>1539475253925</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.4</td>
      <td>37389194</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.021370</td>
      <td>28.0</td>
      <td>21.0</td>
      <td>,ci37389194,</td>
      <td>3.42</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539475062610</td>
      <td>M 3.4 - 8km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,dyfi,focal-mechanism,geoserve,nearby-cities,o...</td>
      <td>-480.0</td>
      <td>1539536756176</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>37389186</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.026180</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>,ci37389186,</td>
      <td>0.44</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539474978070</td>
      <td>M 0.4 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,</td>
      <td>-480.0</td>
      <td>1539475196167</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>73096941</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.077990</td>
      <td>NaN</td>
      <td>192.0</td>
      <td>,nc73096941,</td>
      <td>2.16</td>
      <td>md</td>
      <td>...</td>
      <td>,nc,</td>
      <td>automatic</td>
      <td>1539474716050</td>
      <td>M 2.2 - 10km NW of Avenal, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,scit...</td>
      <td>-480.0</td>
      <td>1539477547926</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
df7[df7.magType=="ml"]
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
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>detail</th>
      <th>dmin</th>
      <th>felt</th>
      <th>gap</th>
      <th>ids</th>
      <th>mag</th>
      <th>magType</th>
      <th>...</th>
      <th>sources</th>
      <th>status</th>
      <th>time</th>
      <th>title</th>
      <th>tsunami</th>
      <th>type</th>
      <th>types</th>
      <th>tz</th>
      <th>updated</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>37389218</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.008693</td>
      <td>NaN</td>
      <td>85.0</td>
      <td>,ci37389218,</td>
      <td>1.35</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539475168010</td>
      <td>M 1.4 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,</td>
      <td>-480.0</td>
      <td>1539475395144</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>37389202</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.020030</td>
      <td>NaN</td>
      <td>79.0</td>
      <td>,ci37389202,</td>
      <td>1.29</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539475129610</td>
      <td>M 1.3 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,</td>
      <td>-480.0</td>
      <td>1539475253925</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.4</td>
      <td>37389194</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.021370</td>
      <td>28.0</td>
      <td>21.0</td>
      <td>,ci37389194,</td>
      <td>3.42</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539475062610</td>
      <td>M 3.4 - 8km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,dyfi,focal-mechanism,geoserve,nearby-cities,o...</td>
      <td>-480.0</td>
      <td>1539536756176</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>37389186</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.026180</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>,ci37389186,</td>
      <td>0.44</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>automatic</td>
      <td>1539474978070</td>
      <td>M 0.4 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,</td>
      <td>-480.0</td>
      <td>1539475196167</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>20280432</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>,ak20280432,</td>
      <td>1.70</td>
      <td>ml</td>
      <td>...</td>
      <td>,ak,</td>
      <td>automatic</td>
      <td>1539473176017</td>
      <td>M 1.7 - 105km W of Talkeetna, Alaska</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,origin,</td>
      <td>-540.0</td>
      <td>1539473596465</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9325</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>38063983</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.042720</td>
      <td>NaN</td>
      <td>56.0</td>
      <td>,ci38063983,</td>
      <td>0.51</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>1537230344890</td>
      <td>M 0.5 - 4km WNW of Julian, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,scit...</td>
      <td>-480.0</td>
      <td>1537290279205</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>9326</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>38063975</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.036520</td>
      <td>NaN</td>
      <td>26.0</td>
      <td>,ci38063975,</td>
      <td>1.82</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>1537230230260</td>
      <td>M 1.8 - 4km W of Julian, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,focal-mechanism,geoserve,nearby-cities,origin...</td>
      <td>-480.0</td>
      <td>1537276829920</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>9328</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>38063967</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.030410</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>,ci38063967,</td>
      <td>1.00</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>1537230135130</td>
      <td>M 1.0 - 3km W of Julian, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,geoserve,nearby-cities,origin,phase-data,scit...</td>
      <td>-480.0</td>
      <td>1537276800970</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>9330</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>38063959</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.018650</td>
      <td>NaN</td>
      <td>61.0</td>
      <td>,ci38063959,</td>
      <td>1.10</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>1537229545350</td>
      <td>M 1.1 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,focal-mechanism,geoserve,nearby-cities,origin...</td>
      <td>-480.0</td>
      <td>1537230211640</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>9331</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>38063935</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.016980</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>,ci38063935,</td>
      <td>0.66</td>
      <td>ml</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>1537228864470</td>
      <td>M 0.7 - 9km NE of Aguanga, CA</td>
      <td>0</td>
      <td>earthquake</td>
      <td>,focal-mechanism,geoserve,nearby-cities,origin...</td>
      <td>-480.0</td>
      <td>1537305830770</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>6803 rows × 26 columns</p>
</div>




```python
pd.cut(df7.mag,bins=5,labels =['1','2','3','4','5'])
```




    0       2
    1       2
    2       3
    3       1
    4       2
           ..
    9327    2
    9328    2
    9329    3
    9330    2
    9331    2
    Name: mag, Length: 9332, dtype: category
    Categories (5, object): ['1' < '2' < '3' < '4' < '5']




```python
df7.mag.value_counts()
```




     1.60    303
     1.30    294
     1.40    272
     1.10    272
     1.20    243
            ... 
    -0.91      1
     2.91      1
    -0.73      1
    -1.23      1
     3.22      1
    Name: mag, Length: 477, dtype: int64




```python
ex4
```


```python
df8 = pd.read_csv('weather_by_station.csv')
```


```python
df8.head()
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
      <th>date</th>
      <th>datatype</th>
      <th>station</th>
      <th>value</th>
      <th>station_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>PRCP</td>
      <td>GHCND:US1CTFR0039</td>
      <td>0.0</td>
      <td>STAMFORD 4.2 S, CT US</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01</td>
      <td>PRCP</td>
      <td>GHCND:US1NJBG0015</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01</td>
      <td>PRCP</td>
      <td>GHCND:US1NJBG0017</td>
      <td>0.0</td>
      <td>GLEN ROCK 0.7 SSE, NJ US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0017</td>
      <td>0.0</td>
      <td>GLEN ROCK 0.7 SSE, NJ US</td>
    </tr>
  </tbody>
</table>
</div>




```python
df8.shape
```




    (80256, 5)




```python
df8[df8.datatype=="SNOW"]
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
      <th>date</th>
      <th>datatype</th>
      <th>station</th>
      <th>value</th>
      <th>station_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0015</td>
      <td>0.0</td>
      <td>NORTH ARLINGTON 0.7 WNW, NJ US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0017</td>
      <td>0.0</td>
      <td>GLEN ROCK 0.7 SSE, NJ US</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0018</td>
      <td>0.0</td>
      <td>PALISADES PARK 0.2 WNW, NJ US</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0023</td>
      <td>0.0</td>
      <td>OAKLAND 0.9 SSE, NJ US</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-01-01</td>
      <td>SNOW</td>
      <td>GHCND:US1NJBG0039</td>
      <td>0.0</td>
      <td>RIVER EDGE 0.4 NNE, NJ US</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>80175</th>
      <td>2018-12-31</td>
      <td>SNOW</td>
      <td>GHCND:USC00308577</td>
      <td>0.0</td>
      <td>SYOSSET, NY US</td>
    </tr>
    <tr>
      <th>80179</th>
      <td>2018-12-31</td>
      <td>SNOW</td>
      <td>GHCND:USW00014732</td>
      <td>0.0</td>
      <td>LA GUARDIA AIRPORT, NY US</td>
    </tr>
    <tr>
      <th>80191</th>
      <td>2018-12-31</td>
      <td>SNOW</td>
      <td>GHCND:USW00014734</td>
      <td>0.0</td>
      <td>NEWARK LIBERTY INTERNATIONAL AIRPORT, NJ US</td>
    </tr>
    <tr>
      <th>80220</th>
      <td>2018-12-31</td>
      <td>SNOW</td>
      <td>GHCND:USW00094728</td>
      <td>0.0</td>
      <td>NY CITY CENTRAL PARK, NY US</td>
    </tr>
    <tr>
      <th>80245</th>
      <td>2018-12-31</td>
      <td>SNOW</td>
      <td>GHCND:USW00094789</td>
      <td>0.0</td>
      <td>JFK INTERNATIONAL AIRPORT, NY US</td>
    </tr>
  </tbody>
</table>
<p>13395 rows × 5 columns</p>
</div>




```python
df
```
