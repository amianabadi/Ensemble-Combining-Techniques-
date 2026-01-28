def Frequency_mapping(Observations,IDV_Models):
 
'''
This function is related to developed option in progress of MMSE (i.e., frequency mapping).

Input:
Observations: Contains the Observations.
IDV_Models: Contains the indivudual (input) models

Output:
IDV_Models_mapped: Mapped indivudual (input) models


References:
1- Ajami, N. K., Duan, Q., Gao, X., & Sorooshian, S. (2006). Multimodel Combination Techniques 
for Analysis of Hydrological Simulations: Application to Distributed Model Intercomparison 
Project Results. Journal of Hydrometeorology, 7(4), 755–768. https://doi.org/10.1175/JHM519.1.

2- Jafarzadeh, A., Khashei-Siuki, A., & Pourreza-Bilondi, M. (2022). Performance Assessment
 of Model Averaging Techniques to Reduce Structural Uncertainty of Groundwater Modeling. Water
 Resources Management, 36(1), 353–377. https://doi.org/10.1007/s11269-021-03031-x
'''

    import numpy as np
    
    N_model=IDV_Models.shape[0];
    IDV_Models_mapped=IDV_Models.copy()
    for iii in range(0,N_model):
        def ecdf(data):
            """ Compute ECDF """
            x = np.sort(data)
            n = x.size
            f = np.arange(1, n+1) / n # frequency
            return(x,f)
        [X1,f1]=ecdf(IDV_Models.iloc[iii,:]);
        [X2,f2]=ecdf(Observations);
        for ii in range(0,len(IDV_Models.iloc[iii,:])):
            from scipy import interpolate
            Intlpmodel1= interpolate.interp1d(X1, f1);
            favg=Intlpmodel1(IDV_Models.iloc[iii,ii]);

            Intlpmodel2= interpolate.interp1d(f2,X2);
            IDV_Models_mapped.iloc[iii,ii]=Intlpmodel2(favg);
        
    return(IDV_Models_mapped)