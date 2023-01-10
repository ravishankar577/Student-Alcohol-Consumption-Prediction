# Alcoholic Detector
In this project we try to find alcohol consumption and health status of students from their social characteristics.

## Pre-processing The Dataset
The two dataset files those have been used in the project have been obtained from this [page][dataset] from [Kaggle][kaggle]. 

The dataset contains the following columns:

1.  `school`: Student's school
2.  `sex`: Student's sex 
3.  `age`: Student's age
4.  `address`: Student's home address type
5.  `famsize`: Family size
6.  `Pstatus`: Parent's cohabitation status
7.  `Medu`: Mother's education
8.  `Fedu`: Father's education
9.  `Mjob`: Mother's job
10. `Fjob`: Father's job
11. `reason`: Reason to choose this school
12. `guardian`: Student's guardian
13. `traveltime`: Home to school travel time
14. `studytime`: Weekly study time
15. `failures`: Number of past class failures
16. `schoolsup`: Extra educational support
17. `famsup`: family educational support
18. `paid`: Extra paid classes within the course subject
19. `activities`: Extra-curricular activities
20. `nursery`: Attended nursery school 
21. `higher`: Wants to take higher education
22. `internet`: Internet access at home
23. `romantic`: With a romantic relationship
24. `famrel`: Quality of family relationships
25. `freetime`: Free time after school
26. `goout`: Going out with friends
27. `Dalc`: Workday alcohol consumption
28. `Walc`: Weekend alcohol consumption
29. `health`: Current health status
30. `absences`: Number of school absences
31. `G1`: First period grade
31. `G2`: Second period grade
32. `G3`: Final grade

The school is not important because we will try to find the generalized alcohol consumption. Unnecessary columns like `school`, `guardian` and `paid` have been removed from dataset.

The grade columns which indicate school success have been simplified. School success calculated via the formula down below:

`success = (G1 + G2 + G3) / 3`

`age` and `absences` are the only columns that have numeric absolute value. So those columns remain as original.

`health`, `Dalc`, `Walc`, `goout`, `freetime`, `failures`, `traveltime`, `studytime` and `famrel` columns have the data that parsed as categorical despite they have numerical quantity values. So their data have been become between 0 and 1.

`sex`, `address`, `famsize`, `Pstatus`, `schoolsup`, `famsup`, `activities`, `nursery`, `higher`, `internet` and `romantic` columns encoded as binary because they have two values each.

The remaining columns are categorically processed with the method called *One Hot Encoding*

## One Hot Encoding Problem
Tensors are multi-dimensional vectors like arrays and matrixes etc. In machine learning, data must be provided as a tensor. Categorical columns like *dog or cat* encoded with this method.
    
*Pandas* is a python library that allows us to process the data has rows and columns. I made a mistake when categorical data were processed with this method. I encoded the data and write them all to their own columns as array. Naturally TensorFlow raised an exception when I feed the placeholders with my wrong data. After I noticed what I did, I encoded the columns in seperate columns for every categorical value in each column. 

For example we can encode `label` column in the following dataset:

| index | value | label |
| :---: | :---: | :---: |
| 1     | 45    | 'cat' |
| 2     | 13    | 'dog' |
| 3     | 42    | 'cat' |
| 4     | 98    | 'rat' |

Wrong encoding:


| index | value | label     |
| :---: | :---: | :-------: |
| 1     | 45    | [1, 0, 0] |
| 2     | 13    | [0, 1, 0] |
| 3     | 42    | [1, 0, 0] |
| 4     | 98    | [0, 0, 1] |


Correct encoding:


| index | value | labelcat | labeldog | labelrat |
| :---: | :---: | :------: | :------: | :------: |
| 1     | 45    | 1        | 0        | 0        |
| 2     | 13    | 0        | 1        | 0        |
| 3     | 42    | 1        | 0        | 0        |
| 4     | 98    | 0        | 0        | 1        |


First, I used `pandas.get_dummies()` the result was enough for train but I faced another problem at test phase. When I created a dataframe for the test input, I noticed the column order of dataframe was changed after the dataframe to matrix conversion. So I decided to change the conversion method and do it manually row by row.

## Learning Phase
Simple, feed-forward neural network has been used in the project. The architecture of the neural network is: 

`Input[45] > Hidden_1[100] > Hidden_2[100] > Output[2]`

Learning rate is set to 0.001 and for this learning rate, number of epochs is set to 10000. Because of to be exposed to overfitting, more than 10000 cycles is unnecessary for this learning rate.


### Training data errors:
![Training errors](/outputs/test_errors.png)

### Test data errors:
![Test errors](/outputs/test_errors.png)

## Usage of TensorFlow Sessions

In TensorFlow, the defined placeholders must have been fed with data via the `feed_dict` argument in `Session.run()` function. In addition, the data that requested from the running session must be specified in the `Session.run()` function.

For example:

``` python
# What we give for feeding placeholders
feed = {
    input_placeholder: training_inputs,
    output_placeholder: training_outputs
}

#                                *Feed with this data*
#                                          v
training_cost = sess.run(cost, feed_dict=feed)
#                         ^ 
#           *Evaluate and return this data*
```

## Resources
- [Kaggle][kaggle]
- [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) 
- [Multi Layer Perceptron MNIST](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/multi_layer_perceptron_mnist.html)  
- [Droput layers](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/dropout_layer.html)

[dataset]: https://www.kaggle.com/uciml/student-alcohol-consumption
[kaggle]: https://www.kaggle.com/datasets