# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:18 2024

@author: Ahmad Jafarzadeh AJM
"""
import pandas as pd
import numpy as np
from hydroeval import evaluator, nse, rmse
from datetime import date
# %%
# Load data
file_path = 'Orginal data_Corrected.xlsx'
excel_data = pd.ExcelFile(file_path)

# List of station names
StationsName = excel_data.sheet_names
StationsName.sort()

# Determine the time scale
# scale ='Annually';
scale ='Monthly'; 
# scale ='Daily'; 

# New output file for data and weights 
output_file = f'Output_{scale}Total.xlsx'
weights_output_file = f'Weights_Output_{scale}.xlsx'  

# List the incorporated models
Models = ['MSWEP', 'CHIRPS', 'PERSIANN-CDR', 'PERSIANN-CCS-CDR']

# %%
# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer_outputs, \
    pd.ExcelWriter(weights_output_file,engine='xlsxwriter') as writer_weights :
    for NameStation in StationsName:
        # Get data for each station
        df = excel_data.parse(NameStation)
        # df = pd.read_excel('Orginal data_Corrected.xlsx', sheet_name=NameStation)
        print(f'Processing data for station: {NameStation}')

        # ----------------Time Convertion and Outliers detection
        if scale == 'Annually':
            df2 = df.groupby(['year']).sum().reset_index()
            
            # df2['detector_outliers'] = 0
            df2['Station_newdf'] = df2['Station']
            
            # Compute the long-term average of rainfall for each year
            # mean = df2['Station'].mean()
            # std = df2['Station'].std()

            # outliers = (df2['Station'] - mean).abs() > (1.4 * std)
            # df2.loc[outliers, 'detector_outliers'] = 1
            # df2.loc[outliers, 'Station_newdf'] = mean
            
            # A dummy variable to include year column
            col_output=1 
        elif scale == 'Monthly':
            df2 = df.groupby(['year', 'month']).sum().reset_index()
            
            # Outlier detection and replacement
            # df2['detector_outliers'] = 0
            df2['Station_newdf'] = df2['Station']
            
            # Calculate the Monthly Long-Term Average (MLTA) for 12 months 
            # MLTA=[]
            # for Jmonth in range(1, 13): 
            #     MLTA.append(df2[df2['month'] == Jmonth]['Station'].mean())
            # # Outlier detection
            # mean = df2['Station'].mean()
            # std = df2['Station'].std()
            # outliers = (df2['Station'] - mean).abs() > (1.4 * std)
            # print(outliers[outliers].index.tolist())
            # df2.loc[outliers, 'detector_outliers'] = 1
            
            # # generate the MLTA for each month of df2
            # MLTA_all = df2['month'].apply(lambda x: MLTA[int(x) - 1])
            
            # # Replacing the monthly outhliers wuth MLTA
            # df2.loc[outliers, 'Station_newdf'] = MLTA_all
            
            # A dummy variable to include year and month columns
            col_output=2
        elif scale == 'Daily':
            df2 = df
            # # A function to generate the julian day
            # def get_julian_day(year, month, day):
            #     current_date = date(year, month, day)
            #     ordinal_day = current_date.toordinal()
            #     return ordinal_day - date(current_date.year, 1, 1).toordinal() + 1
            
            # for iindex in range (0,len(df)):
            #     julian_day = get_julian_day(df['year'][iindex], df['month'][iindex], df['day'][iindex])
            #     df.at[iindex,'Julian_day']=julian_day
            
            # Outlier detection and replacement
            # df2['detector_outliers'] = 0
            df2['Station_newdf'] = df2['Station']
            
            # # Calculate the Daily Long-Term Average (DLTA) for 365 days 
            # DLTA=[]
            # for Jday in range(1, 367): 
            #     DLTA.append(df[df['Julian_day'] == Jday]['Station'].mean())
            
            # # Outlier detection
            # mean = df2['Station'].mean()
            # std = df2['Station'].std()
            # outliers = (df2['Station'] - mean).abs() > (10 * std)
            # print(outliers[outliers].index.tolist())
            # df2.loc[outliers, 'detector_outliers'] = 1
            
            # # generate the DLTA for each month of df2
            # DLTA_all = df['Julian_day'].apply(lambda x: DLTA[int(x) - 1])
            
            # # Replacing the monthly outhliers wuth MLTA
            # df2.loc[outliers, 'Station_newdf'] = DLTA_all
               
            # A dummy variable to include year, month, and column column
            col_output=3
        # ------------------------------------------------
        # df2.to_excel(f'Rearranged_Df{scale}.xlsx', index=False)
        
        # Assign multiple columns as input models
        MultiModels = df2.iloc[:, 4:9].round(1)
        # To remove some columns
        MultiModels=MultiModels.drop(columns='CHIRPS0.25');
        # To correct the name of models
        MultiModels.rename(columns={'CHIRPS0.05': 'CHIRPS'}, inplace=True)
        # A transpose
        MultiModels = MultiModels.T

        # Assign one column as measurement
        Measurement = df2.iloc[:, -1]

        # Import combination techniques
        from ComTechs import SMA, WAM, MMSE, M3SE

        # Get the output from combination techniques
        SMA_Output = SMA(Measurement, MultiModels)
        SMA_Output = pd.DataFrame(SMA_Output).T

        WAM_Output, WAM_Weights = WAM(Measurement, MultiModels)
        WAM_Output = WAM_Output.T

        MMSE_Output, MMSE_Weights = MMSE(Measurement, MultiModels,scale)
        MMSE_Output = MMSE_Output.T

        M3SE_Output, M3SE_Weights = M3SE(Measurement, MultiModels)
        M3SE_Output = M3SE_Output.T

        # Aggregate the simulation results
        Simulation_total = MultiModels.T
        Simulation_total = Simulation_total.assign(SMA=SMA_Output)
        Simulation_total = Simulation_total.assign(WAM=WAM_Output)
        Simulation_total = Simulation_total.assign(MMSE=MMSE_Output)
        Simulation_total = Simulation_total.assign(M3SE=M3SE_Output)

        # Define a function to calculate NSE, R2, RMSE
        def model_eval(y_sim, y_obs):
            y_mean_obs = np.mean(y_obs)
            y_mean_sim = np.mean(y_sim)
            ss_1 = (np.sum((y_obs - y_mean_obs) * (y_sim - y_mean_sim))) ** 2
            ss_total = np.sum((y_obs - y_mean_obs) ** 2)
            ss_residual = np.sum((y_sim - y_mean_sim) ** 2)
            r2 = ss_1 / ((ss_total) * (ss_residual))

            nse_value = evaluator(nse, y_sim, y_obs)[0]
            rmse_value = evaluator(rmse, y_sim, y_obs)[0]
            return (nse_value, r2, rmse_value)

        # Convert Measurement to a DataFrame
        Measurement = pd.DataFrame(Measurement)

        # Calculate performance criteria
        model_metrics = Simulation_total.apply(model_eval, args=(Measurement['Station_newdf'],))
        model_metrics = model_metrics.T

        # Rename columns
        new_columns = {0: 'NSE', 1: 'R2', 2: 'RMSE'}
        model_metrics.rename(columns=new_columns, inplace=True)

        # Aggregate the results
        result = pd.concat([Measurement, Simulation_total, model_metrics], axis=1)

        # Add year column into results
        result = pd.concat([df2.iloc[:, 0:col_output], result], axis=1)
        result = pd.DataFrame(result)
        
        # Save results to the Excel file, each station's results in a separate sheet
        result.to_excel(writer_outputs, sheet_name=NameStation, index=False)
        
        # Prepare weights data for saving  
        weights_data = {  
            'WAM':WAM_Weights,
            'M3SE': M3SE_Weights,  
            'MMSE': MMSE_Weights,  
        }  
        
        # Flatten the WAM array
        weights_data['WAM'] = weights_data['WAM'].flatten()
        
        # Create a DataFrame for weights  
        weights_df = pd.DataFrame(weights_data)  
        weights_df.insert(0,'Models',Models)
        
        # Save weights to the weights Excel file  
        weights_df.to_excel(writer_weights, sheet_name=NameStation, index=False)

        
        
        print(f'Finished processing for station: {NameStation}')
        print('*******************************************')
print('')
print('********* Finish ***********')
print('')