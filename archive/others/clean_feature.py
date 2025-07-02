import re
def clean_feature_name(feature):
    """
    Removes numeric prefixes and numeric suffixes from a TSFEL feature name.
    Example: '0_ECDF_0' -> 'ECDF'
    """
    # Step 1: Remove numeric prefix (e.g., "0_")
    feature = re.sub(r'^\d+_', '', feature)  
    
    # Step 2: Remove numeric suffix (e.g., "_0", "_1", etc.)
    feature = re.sub(r'_\d+$', '', feature)
    
    return feature
corr_features = ['0_Area under the curve', '0_Average power', 
                 '0_ECDF Percentile_0', '0_ECDF Percentile_1', '0_LPCC_11', 
                 '0_LPCC_2', '0_LPCC_7', '0_LPCC_8', '0_LPCC_9', '0_Max', '0_Mean', 
                 '0_Median', '0_Median absolute deviation', '0_Median absolute diff', 
                 '0_Positive turning points', '0_Root mean square', '0_Slope', 
                 '0_Spectral decrease', '0_Spectral distance', '0_Spectral roll-off', 
                 '0_Spectral skewness', '0_Spectral slope', '0_Spectral spread', 
                 '0_Standard deviation', '0_Sum absolute diff', '0_Variance', 
                 '0_Wavelet absolute mean_12.5Hz', '0_Wavelet absolute mean_2.78Hz', 
                 '0_Wavelet absolute mean_25.0Hz', '0_Wavelet absolute mean_3.12Hz', 
                 '0_Wavelet absolute mean_3.57Hz', '0_Wavelet absolute mean_4.17Hz', 
                 '0_Wavelet absolute mean_5.0Hz', '0_Wavelet absolute mean_6.25Hz', 
                 '0_Wavelet absolute mean_8.33Hz', '0_Wavelet energy_2.78Hz', 
                 '0_Wavelet energy_3.12Hz', '0_Wavelet energy_3.57Hz', 
                 '0_Wavelet energy_4.17Hz', '0_Wavelet energy_5.0Hz', 
                 '0_Wavelet energy_6.25Hz', '0_Wavelet energy_8.33Hz', 
                 '0_Wavelet standard deviation_12.5Hz', '0_Wavelet standard deviation_25.0Hz', 
                 '0_Wavelet standard deviation_3.12Hz', '0_Wavelet standard deviation_3.57Hz', 
                 '0_Wavelet standard deviation_4.17Hz', '0_Wavelet standard deviation_5.0Hz', 
                 '0_Wavelet standard deviation_6.25Hz', '0_Wavelet standard deviation_8.33Hz', 
                 '0_Wavelet variance_12.5Hz', '0_Wavelet variance_2.78Hz', '0_Wavelet variance_25.0Hz', 
                 '0_Wavelet variance_3.12Hz', '0_Wavelet variance_3.57Hz', '0_Wavelet variance_4.17Hz', 
                 '0_Wavelet variance_5.0Hz', '0_Wavelet variance_6.25Hz', '0_Wavelet variance_8.33Hz']
clean_selected_features = [clean_feature_name(feat) for feat in corr_features]
print(clean_selected_features)