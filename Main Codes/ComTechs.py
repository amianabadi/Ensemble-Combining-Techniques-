def SMA(Observations,IDV_Models):
    # Observations: Contains the Observations.
    # IDV_Models: Contains the competing (input) models
    import numpy as np
    
    Avg_multi_models=np.array ([np.average(IDV_Models,axis=1)]); # axis=1,0,None: compute the average of an array (column), a row, and whole data.
    Avg_multi_models=Avg_multi_models.T;
    Predictants = np.array([
            Observations
        ]);
    Avg_Obs=np.average(Predictants);
    
    SMA_Output=np.array([
        Avg_Obs+np.average( (IDV_Models-Avg_multi_models).T ,axis=1)
    ]);
    # SMA_Output=[];
    # for i in range(0,len(Observations)):
    #     SMA_Output.append(Avg_Obs+np.average(IDV_Models[0:6,i]-Avg_multi_models[0:6]));
    return(SMA_Output)

def WAM(Observations,IDV_Models):
    # Observations: Contains the Observations.
    # IDV_Models: Contains the competing (input) models
    import numpy as np
    from scipy.optimize import minimize

    
    
    def objective_func(w):
        # We here define the objective function.
        w = np.array(w)  
        Predictors = IDV_Models.T  
        Predictants = Observations  
        
        
        # Compute predictions  
        predictions = Predictors.dot(w) 

        # Return sum of squares  
        return np.sum((predictions - Predictants) ** 2) # To calculate the sum of squares.

    def equality_constraints(w):
        # We set the equlaity contraints (W1+W2+...+Wn=1)
        return np.sum(w) - 1;

    # Prepare bounds and constraints  
    n_inputmodel = IDV_Models.shape[0]  
    bounds = [(0, 1)] * n_inputmodel  
    constraints = {'type': 'eq', 'fun': equality_constraints}

    # Initialize randomly while ensuring they sum to 1  
    w0 = np.random.rand(n_inputmodel)  
    w0 /= w0.sum()  # Normalize to sum to 1  

    # To run optimizations
    results = minimize(objective_func, w0, method='SLSQP', bounds=bounds, constraints=constraints)  


    #print(results)

    w=results.x;
    w=np.array([w]);
    w=w.T;
    Predictors = IDV_Models;
    Predictors=Predictors.T;
    WAM_Output=(Predictors.dot(w)).T;
    return(WAM_Output,w);

def MMSE(Observations,MultiModels,scale):
    # Observations: Contains the Observations.
    # IDV_Models: Contains the competing (input) models
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    
    
    """
        Note:
        Input DataFrames are internally deep-copied to avoid side effects.
    """
    IDV_Models = MultiModels.copy(deep=True)
    
    #--------------------- To perform the outliers detection-----------------
    from f3_Out_Corr import WMOC2
    IDV_Models = WMOC2(Observations, IDV_Models, scale)
    # -----------------------------------------------------------------------

    def objective_func(w):
        # We here define the objective function.
        w = np.array(w)  # Make sure w is a numpy array  
        Predictors = IDV_Models.T  # Transpose IDV_Models to align with weights  

        # Calculate predictions  
        predictions = Predictors.dot(w)  
        difference = predictions - Observations  

        # Calculate the sum of squares of the differences  
        return np.sum(difference ** 2) 
    
    # Number of input models  
    n_inputmodel = IDV_Models.shape[0]  
    
    # Set the bounds for weights (may want to modify for non-negative constraints)  
    bounds = [(None, None)] * n_inputmodel  # Unconstrained weights  

    # Set the initial values for weights  
    w0 = [0.5] * n_inputmodel  # Initial guess; can be randomized for better optimization  

    # Run the optimization  
    results = minimize(objective_func, w0, method='SLSQP', bounds=bounds) 
    
    # Optimal weights  
    w = results.x  
    
    # Calculate the Average of IDV_Models and Observations  
    Predictors = IDV_Models.T  
    Avg_multi_models = np.average(Predictors, axis=0)
    
    Avg_Obs = pd.DataFrame(np.zeros((len(Observations), 1)), columns=['Avg_Observation'])
    # Wet_Days = Observations > 0
    Avg_Obs.loc[:,'Avg_Observation'] = np.average(Observations) 
    
    # Calculate MMSE Output
    MMSE_Output = pd.DataFrame(np.zeros((len(Observations), 1)))
     
    # MMSE estimation
    MMSE_Output = (
        Avg_Obs.iloc[:, 0]
        + (Predictors - Avg_multi_models).dot(w)
    ).round(2)
    
    
    # Set MMSE to zero where all predictors are zero (zero input= zero output)
    indices = Predictors.index[Predictors.eq(0).all(axis=1)]
    MMSE_Output.loc[indices] = 0
    
    # Set MMSE to zero where weak indivisoul model has outliers value.
    # MMSE_Output.loc[MMSE_Output<0]=0

    return MMSE_Output, w

def M3SE(Observations,MultiModels):
    # Observations: Contains the Observations.
    # IDV_Models: Contains the competing (input) models
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    
    """
        Note:
        Input DataFrames are internally deep-copied to avoid side effects.
    """
    IDV_Models = MultiModels.copy(deep=True)
    
    #--------------------- To perform the frequency mapping---------------------------------
    from f4_frq_mapp import Frequency_mapping
    IDV_Models_mapped=Frequency_mapping(Observations,IDV_Models);
    # --------------------------------The end of frequency mapping--------------------------.

    def objective_func(w):
        """ Objective function to minimize the squared differences. """  
        w = np.array(w)  # Ensure w is a numpy array  
        Predictors = IDV_Models_mapped.T  # Transpose to align with weights  

        # Calculate predictions  
        predictions = Predictors.dot(w)  
        difference = predictions - Observations  

        # Return sum of squares  
        return np.sum(difference ** 2)
    
    # Set the number of input models and bounds for weights  
    n_inputmodel = IDV_Models.shape[0]  
    bounds = [(None, None)] * n_inputmodel  # Unconstrained weights  

    # Initialize weights  
    w0 = [0.5] * n_inputmodel  # Initial guess  

    # Perform optimization  
    results = minimize(objective_func, w0, method='SLSQP', bounds=bounds)
    
    # Optimal weights  
    w = results.x  
    
    # Calculate the Average of IDV_Models and Observations 
    Predictors = IDV_Models_mapped.T  
    Avg_multi_models = np.average(Predictors, axis=0)
    
    Avg_Obs = pd.DataFrame(np.zeros((len(Observations), 1)), columns=['Avg_Observation'])
    # Wet_Days = Observations > 0
    Avg_Obs.loc[:,'Avg_Observation'] = np.average(Observations) 
    
    # Calculate M3SE Output
    M3SE_Output = pd.DataFrame(np.zeros((len(Observations), 1)))
      
    # M3SE estimation
    M3SE_Output = (
        Avg_Obs.iloc[:, 0]
        + (Predictors - Avg_multi_models).dot(w)
    ).round(2)
    
    # Set M3SE to zero where all predictors are zero (zero input= zero output)
    Predictors = IDV_Models.T  
    indices = Predictors.index[Predictors.eq(0).all(axis=1)]
    M3SE_Output.loc[indices] = 0
    
    # Set M3SE to zero where weak indivisoul model has outliers value.
    M3SE_Output.loc[M3SE_Output<0]=0
    
    
    return M3SE_Output, w
