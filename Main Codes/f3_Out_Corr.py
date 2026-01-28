
def WMOC2(Observations, IDV_Models, scale, mode='flexible'):
    """
    The MMSE is possible to generate -values when the wead input models has an 
    outliers align with low values of storng models, especially in dry events.
    Hence, this Fn is to correct the outliers of input models in dry days/months.
    
    Weak Models' Outlier Correction (WMOC).
    Assumption:
        Outlier is a value (input models) more than  observation + Alt_Val. 

    Parameters
    ----------
    Observations : pandas.Series (1 × n)
        Measured rainfall
    IDV_Models : pandas.DataFrame (m × n)
        Individual model simulations
    scale : str
        'Daily', 'Monthly', 'Annually'
    mode : str
        'strict'   -> only one outlier per day is corrected
        'flexible' -> all outliers are corrected
    Alt_Val : float
        difference from observed value.
    
    Returns
    -------
    IDV_Models_corrected : pandas.DataFrame
    """

    import numpy as np
    import pandas as pd

    IDV_Models = IDV_Models.copy()  # prevent side effects

    if scale == 'Annually':
        return IDV_Models

    # ------------------ Threshold parameters ------------------
    if scale == 'Daily':
        dry_th = 3
        Alt_Val = 1.5
        # Coef_rainfal = 1.2
    else:  # Monthly
        dry_th = 3
        Alt_Val = 6
        # Coef_rainfal = 1.2

    # ------------------ Select dry / low-rainfall days ------------------
    M1 = IDV_Models.loc[:, Observations.lt(dry_th)]
    obs_M1 = Observations.loc[M1.columns]

    # ------------------ Detect outliers ------------------
    mask_gt = M1.gt(obs_M1+Alt_Val, axis=1)
    # Number of models
    n_models = M1.shape[0]
    
    # Columns where ALL models are outliers
    all_outlier_cols = mask_gt.sum(axis=0) == n_models
        
    # Fix those columns
    for col in mask_gt.columns[all_outlier_cols]:
    
        # Threshold for this day
        th = obs_M1[col] + Alt_Val
    
        # Absolute distance of each model from threshold
        diff = (M1[col] - th).abs()
    
        # Index (row) of model with minimum deviation
        idx_min = diff.idxmin()
    
        # Force this model to be non-outlier
        mask_gt.loc[idx_min, col] = False
    
    # Mean of non-outlier products (per day)
    mean_others = M1.where(~mask_gt).mean(axis=0)

    # ------------------ SWITCHING LOGIC ------------------
    if mode == 'strict':
        # Only one outlier per day is corrected
        one_outlier_cols = mask_gt.sum(axis=0) == 1
        M1 = M1.mask(
            mask_gt & one_outlier_cols,
            mean_others,
            axis=1
        )

    elif mode == 'flexible':
        # Any value above threshold is corrected
        M1 = M1.mask(
            mask_gt,
            mean_others,
            axis=1
        )

    else:
        raise ValueError("mode must be 'strict' or 'flexible'")

    # ------------------ Write back ------------------
    IDV_Models.loc[:, Observations.lt(dry_th)] = M1

    return IDV_Models


def WMOC(Observations, IDV_Models, scale):
    '''
    Weak Models' Outlier Correction (WMOC)
    
    Set the predictors's outliers with mean of other models when observed 
    rainfall is zero.
    
    Parameters
    ----------
        - Observations: measured rainfall 1*n (n sample size)
        - IDV_Models: Indivisual models m*n (m is N.O of input models)
        - scale: time scale 'Annually', 'Monthly', and 'Daily'; 
    Returns
    -------
        Corrected Indivisual models
    
    '''
    
    import numpy as np
    import pandas as pd
    
    if scale != 'Annually': 
        
        # To get the indices with dry days/months
        # Determine the threshold
        if scale == 'Daily':
            dry_th=3 # for daily
            Alt_Val = 1.5 # alternative_value
            Coef_rainfal=1.2
        else :
            dry_th=7.5 # for monthly
            Alt_Val = 3 # alternative_value
            Coef_rainfal=2
        M1 = IDV_Models.loc[:, Observations.lt(dry_th)]
        
        # Observations which is mapped to M1
        obs_M1 = Observations.loc[M1.columns]
        
        threshold = np.where(obs_M1 == 0,
        Alt_Val,                      # zero value  
        Coef_rainfal * obs_M1        # Days with rainfall
        )
        threshold = pd.Series(threshold, index=M1.columns)
        
        # Get the days of M1 which is grater than threshold (Boolean Var True/False)
        mask_gt = M1.gt(threshold, axis=1)
        # Get the mask_gt's indices which has only one product is outliers
        one_outlier_cols = mask_gt.sum(axis=0) == 1
        # Calculate the average for one_outlier_cols
        mean_others = M1.where(~mask_gt).mean(axis=0)
        # Replacement of M1's outliers with mean_others
        M1 = M1.mask(
            mask_gt & one_outlier_cols,
            mean_others,
            axis=1
        )
        IDV_Models.loc[:, Observations.lt(1)]=M1
    return IDV_Models




