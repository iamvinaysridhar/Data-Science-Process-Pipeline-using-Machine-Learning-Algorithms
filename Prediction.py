if __name__ == "__main__":

    #Importing some libraries
    import numpy as np
    import pandas as pd
    import os
    #Getting rid of pesky warnings
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    np.warnings.filterwarnings('ignore')

    column_names = [
         	"Age",
		"BusinessTravel",	
		"Department",
		"DistanceFromHome",
		"Education",
		"EnvironmentSatisfaction",
		"Gender",
		"JobInvolvement",
		"JobLevel",
		"JobRole",
		"JobSatisfaction",
		"MaritalStatus",
		"MonthlyIncome",
		"NumCompaniesWorked",
		"OverTime",
		"PercentSalaryHike",
		"PerformanceRating",
		"StockOptionLevel",
		"TotalWorkingYears",
		"TrainingTimesLastYear",
		"WorkLifeBalance",
		"YearsAtCompany",
		"YearsInCurrentRole",
		"YearsSinceLastPromotion",
		"YearsWithCurrManager"
                ]
    #Importing the dataset
    location = 'final.csv'
    dataset = pd.read_csv(location)
    dataset = dataset.drop(['Unnamed: 0'],axis=1)
    X=dataset.iloc[:,dataset.columns !='Attrition']
    Y=dataset.iloc[:,dataset.columns =='Attrition']
    
    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X)
    

    #Using Pipeline
    import sklearn.pipeline
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import KernelPCA
    from imblearn.pipeline import make_pipeline
    
    
    clf = RandomForestClassifier()
    kernel = KernelPCA()
    
    pipeline = make_pipeline(kernel, clf)
    pipeline.fit(X,Y)

    #User-input
    v = []
    for i in column_names[:]:
        v.append(input(i+": "))
    answer = np.array(v)
    answer = answer.reshape(1,-1)
    answer = sc_X.transform(answer)
    print ("Predicts:"+ str(pipeline.predict(answer)))
    
