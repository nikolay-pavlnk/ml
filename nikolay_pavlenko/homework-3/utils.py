name_features_heart = ['intercept', 'ca_0', 'ca_1', 'ca_2', 'ca_3', 
						'ca_4', 'cp_0', 'cp_1','cp_2', 'cp_3', 
						'exang_0', 'exang_1','fbs_0', 'fbs_1', 'restecg_0', 
						'restecg_1', 'restecg_2', 'sex_0', 'sex_1', 'slope_0', 
						'slope_1', 'slope_2', 'thal_0', 'thal_1', 'thal_2', 
						'thal_3', 'age', 'chol', 'oldpeak', 'thalach', 'trestbps']


mapper = {'smoker': {'no': 0, 'yes': 1}, 'sex': {'female': 0, 'male': 1}, 'region': 
		{'northwest': 0, 'southeast': 1, 'northeast': 2, 'southwest': 3}}

name_features_insurance = ['intercept', 'children_0', 'children_1', 
							'children_2', 'children_3', 'children_4', 'children_5',
							'region_0', 'region_1', 'region_2', 'region_3', 
							'sex_0', 'sex_1', 'smoker_0', 'smoker_1', 'age', 'bmi']

bin_feat_heart = [1, 3, 4, 5, 8, 7, 10, 9]
con_feat_heart = [0, 2, 6, 11, 12]

bin_feat_reg = [2, 3, 4, 5]
con_feat_reg = [0, 1]
