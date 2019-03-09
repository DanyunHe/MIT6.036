import numpy as np
import code_for_hw5 as hw5

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw5.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw5.standard and hw5.one_hot.
# 'name' is not numeric and would need a different encoding.
features1 = [('cylinders', hw5.standard),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

features2 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = hw5.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = hw5.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = hw5.std_y(auto_values)
print("mu={},sigma={}".format(mu,sigma))

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
        
#Your code for cross-validation goes here
#Make sure to scale the RMSE values returned by xval_learning_alg by sigma,
#as mentioned in the lab, in order to get accurate RMSE values on the dataset

# transform data with polynomical orders 
f=[]
for i in [1,2,3]:
      f.append(hw5.make_polynomial_feature_fun(i))

# λ={0.0,0.01,0.02,⋯,0.1} for polynomial features of orders 1 and 2
#λ={0,20,40,⋯,200} for polynomial features of order 3

lamb1=np.arange(0,0.1,0.01)
lamb2=np.arange(0,200,20)

# store the RMSE values at [feature_set=1,2, polynomial_order=1,2, lambda=lamb1]
RMSE1=np.zeros([2,2,10]) 

# store the RMSE values at [feature_set=1,2, polynomial_order=3, lambda=lamb1]
RMSE2=np.zeros([2,10])

for feature in [0,1]:
      for order in [0,1]:
            for l,lam in enumerate(lamb1):
                  RMSE1[feature,order,l]=hw5.xval_learning_alg(f[order](auto_data[feature]), auto_values, lam, 10)

for feature in [0,1]:
      for l,lam in enumerate(lamb2):
            RMSE2[feature,l]=hw5.xval_learning_alg(f[2](auto_data[feature]), auto_values, lam, 10)

# find the minimum RMSE values
idx1=np.argmin(RMSE1)
i1=idx1//20
j1=(idx1//10)%2
k1=idx1%10

idx2=np.argmin(RMSE2)
i2=idx2//10
j2=idx2%10

idx3=np.argmin(RMSE2[0,:])

print("The minimum value of RMSE with order 1,2 is at feature={},order={} and lambda={}, \
      with value {}".format(i1,j1,lamb1[k1],sigma*RMSE1[i1,j1,k1]))

print("The minimum value of RMSE with order 3 is at feature={}, and lambda={},\
 with value {}".format(i2,lamb2[j2],sigma*RMSE2[i2,j2]))

print("The minimum value of RMSE with order 3 is at feature=0, and lambda={},\
 with value {}".format(lamb2[idx3],sigma*RMSE2[0,idx3]))

