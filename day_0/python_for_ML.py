#Machine Learning Foundations
#foundations (linear algebra,vectors,matrix,matrix multiplication,norm,eigenvalue,eigen vector,linear transformatioons(basics)-3blue1brown
#probability and statistics basics-steve brunton
#calcus-3blue1brown
#python
#numpy pandas matplotlib seaborn
#data preprocessing(scaling,normalization,missing value treatment,outlier treatment

#list comprehension

import random
data=[ random.randint(1,100) for i in range(10)]
print(data)
processed_data=[x**2 for x in data if x%2==0]
print(processed_data)

def build_model(model_name,**hyperparameters):
    print(f"Building :{model_name} model")
    for key,value in hyperparameters.items():
        print(f"{key}:{value}")
build_model("CNN",layers=5,activation='Relu',optimizer='adam')

 

import random
new_data=[ random.randint(1,10) for x in range(5)]
print(new_data)
new_processed_data=[x**3 for x in new_data if x%2!=0]
print(new_processed_data)

def ml_modelling(model,*parameters,**hyperparameters):
    print(f"Model :{model}")
    print("parameters:",parameters)
    print("Hyperparameters:")
    for key,value in hyperparameters.items():
        print(f"{key}:{value}")
ml_modelling("Alexnet",32,64,128,layers=10,activation='relu',optimizer='adam')

