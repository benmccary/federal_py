
# Exploratory analysis for federal contractors

This is just a simple exploratory analysis to get familiar with the Python language, docker containers, and Jupyter notebooks.  I've downloaded datasets from the [Federal Contractors Database](https://www.usaspending.gov/#/download_center/custom_award_data). For variable types, see the [data dictionary](http://fedspendingtransparency.github.io/dictionary-v1.1/).

While the goal is simply to get used to Python and some other technologies, I am interested in the specific question of ***what factors are the best predictors of minority owned companies***.  

[Track progress of the project on my trello board](https://trello.com/b/lZYSGp4M/federal-contractors-python)

#### Init
Read in the data and load packages


```python
import pandas as pd
import numpy as np
import pandas_profiling as pp

dat = pd.read_csv('data/2017.csv', low_memory=False)
```

## Profiling


```python
dat.shape
```




    (72367, 225)



Since the data has 225 columns and +70k rows, I'm only going to do a profile report on a small subset of the rows.  I'm also going to save the report as an HTML file outside of this analysis.  


```python
profile = pp.ProfileReport(dat.loc[0:10000])
profile.to_file(outputfile = "profiling/profile.html")
```


```python
import matplotlib.pyplot as plt
cmt = dat.corr()
```

I'm mostly interested in dollars the companies recieve and the size of the company. So I'm going to make a function that checks for a certain level of correlation for the selected variable. The following cells look at:  

* Dollars Obligated
* Number of Employees
* Minority Owned Flag


```python
def corMat(dd, corlv, var):
    ind = abs(dd[var]) > corlv
    return dd.loc[ind, ind];
```


```python
corMat(cmt,0.05, "dollarsobligated").dollarsobligated
```




    dollarsobligated                         1.000000
    baseandexercisedoptionsvalue             0.967203
    baseandalloptionsvalue                   0.861076
    progsourcesubacct                       -0.076980
    prime_awardee_executive1_compensation   -0.084980
    prime_awardee_executive2_compensation   -0.072067
    prime_awardee_executive3_compensation   -0.071098
    prime_awardee_executive4_compensation   -0.071617
    prime_awardee_executive5_compensation   -0.065659
    Name: dollarsobligated, dtype: float64




```python
corMat(cmt,0.05, "numberofemployees").numberofemployees
```




    progsourceagency                             -0.068632
    progsourcesubacct                            -0.112453
    ccrexception                                 -0.357217
    vendor_cd                                     0.093742
    congressionaldistrict                         0.093742
    placeofperformancezipcode                    -0.145658
    transactionnumber                             0.100897
    numberofemployees                             1.000000
    veteranownedflag                             -0.077613
    receivescontracts                             0.077046
    issubchapterscorporation                     -0.124280
    islimitedliabilitycorporation                 0.051856
    ispartnershiporlimitedliabilitypartnership    0.057622
    prime_awardee_executive1_compensation         0.975024
    prime_awardee_executive2_compensation         0.980690
    prime_awardee_executive3_compensation         0.972878
    prime_awardee_executive4_compensation         0.963208
    prime_awardee_executive5_compensation         0.973902
    Name: numberofemployees, dtype: float64




```python
corMat(cmt,0.25, "minorityownedbusinessflag").minorityownedbusinessflag
```




    progsourcesubacct                               -0.288274
    placeofperformancezipcode                        0.255004
    firm8aflag                                       0.316287
    minorityownedbusinessflag                        1.000000
    apaobflag                                        0.614026
    baobflag                                         0.329557
    naobflag                                         0.263247
    haobflag                                         0.269490
    isdotcertifieddisadvantagedbusinessenterprise    0.270052
    prime_awardee_executive1_compensation           -0.337572
    prime_awardee_executive2_compensation           -0.375064
    prime_awardee_executive3_compensation           -0.398767
    prime_awardee_executive4_compensation           -0.446047
    prime_awardee_executive5_compensation           -0.400471
    Name: minorityownedbusinessflag, dtype: float64



The minority owned business flag shows several interesting correlations:  

1. apaobflag, baobflag, naobflag, and haobflag are just subtypes of minority flags: Asian Pacific American, Black American, Native American, and Hispanic American, respectively. (Thus they're not particularly interesting.
2. firm8aflag is for 8(a) Program Participant Organizations, which is a program for small, underpriviledged companies.
3. DOT certified disadvantaged companies has a slightly smaller correlation.
4. All ofthe executive compensations have a negative correlation, meaning that as executive compensation goes up, the likelihood of being minority owned is smaller.


```python
# dat.prime_awardee_executive1_compensation.describe()
```

## Aggregation

I need to aggregate the awards based on company. Some companies have a lot of awards, so it could mess with things.  I'm also interested in adding a column that shows the count of awards for companies since that might be a good predictor of minority owned businesses.

My hunch is that minority owned businesses are smaller than average, and have lower executive compensation/annual revenue than average.

## Add Rural/Urban flag

This dataset includes zip codes and address, but that's not good as a categorical variable. I'd like to create one, but I'm going to have to figure out the best way to do it. On other projects I've had to look up population statistics from the US Census, then join those in based on FIPS codes. We'll see if that's necessary here.

## Logistic Regression

I'd like to do a logistic regression for a few of the variables above to see if they predict minority owned business
