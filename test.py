


self.test_df['company_age'] = CURRENT_YEAR-self.test_df['yr_of_estab']

self.test_df.drop(self.drop_cols, axis=1, inplace =True)

self.test_df.drop_duplicates(inplace=True)


 test_transformed = preprocessor.transform(self.test_df)
 
one_hot_encoded_cols = list(preprocessor.named_transformers_['one_hot']['one_hot'].get_feature_names_out(self.one_hot_cols))
transformed_columns = self.num_features_cols + self.transform_cols + self.ordinal_cols + one_hot_encoded_cols + [col for col in self.train_df.columns if col not in self.num_features_cols + self.ordinal_cols + self.one_hot_cols + self.transform_cols]

train_transformed_df = pd.DataFrame(train_transformed, columns=transformed_columns)

one_hot_encoded_cols = list(preprocessor.named_transformers_['one_hot']['one_hot'].get_feature_names_out(self.one_hot_cols))
transformed_columns = self.num_features_cols + self.transform_cols + self.ordinal_cols + one_hot_encoded_cols + [col for col in self.train_df.columns if col not in self.num_features_cols + self.ordinal_cols + self.one_hot_cols + self.transform_cols]
train_transformed_df = pd.DataFrame(train_transformed_df, columns=transformed_columns)

test_transformed.drop('case_status_Certified', axis=1, inplace=True)

test_transformed = test_transformed.rename(columns={'case_status_Denied': 'case_status'})

train_transformed_df.to_csv(self.trans_traindata_path,index =False)