import pandas as pd

# Załóżmy, że analizujesz jeden z plików
# df_doh = pd.read_csv('CSVs/Total_CSVs/l1-doh.csv')
# df_non_doh = pd.read_csv('CSVs/Total_CSVs/l1-nondoh.csv')
# df_benign = pd.read_csv('CSVs/Total_CSVs/l2-benign-smote.csv')
# df_malicious = pd.read_csv('CSVs/Total_CSVs/l2-malicious.csv')



#Data Version 1 for AI 1
# ---------------------------------------------------------------------------
# doh_temp = df_doh.head(10000)
# non_doh_temp = df_non_doh.head(10000)

# merged_df = pd.concat([doh_temp, non_doh_temp], ignore_index=True)

# columns_to = ['SourceIP', 'DestinationIP', 'TimeStamp']

# merged_df.drop(columns=columns_to, inplace=True)
# print(merged_df.shape[0])
# print(merged_df.shape[1])
# print(merged_df.head())


# merged_df.to_csv('merged_sample.csv', index=False)

# ---------------------------------------------------------------------------

# #Data Version 2 for AI 2

# doh_temp = df_doh.head(10000).copy()
# non_doh_temp = df_non_doh.head(10000).copy()
# benign_temp = df_benign.head(10000).copy()
# malicious_temp = df_malicious.head(10000).copy()

# columns_to_drop = ['SourceIP', 'DestinationIP', 'TimeStamp']

# #zamieniamy Lable na 0 i 1

# non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
# doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
# benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
# malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 1)

# merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

# merged_df.drop(columns=columns_to_drop, inplace=True)
# print(merged_df.shape[0])
# print(merged_df.shape[1])
# merged_df.to_csv('merged_sample_for_1_agent.csv', index=False)


# # generating samples for second agent

# doh_temp = df_doh.head(10000).copy()
# benign_temp = df_benign.head(10000).copy()
# malicious_temp = df_malicious.head(10000).copy()


# doh_temp['Label'] = doh_temp['Label'].replace('DoH', 0)
# benign_temp['Label'] = benign_temp['Label'].replace('Benign', 0)
# malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 1)

# merged_df = pd.concat([doh_temp, benign_temp, malicious_temp], ignore_index=True)

# merged_df.drop(columns=columns_to_drop, inplace=True)
# print(merged_df.shape[0])
# print(merged_df.shape[1])
# merged_df.to_csv('merged_sample_for_2_agent.csv', index=False)

# ---------------------------------------------------------------------------

# #Data Version 3 for AI 3 (Training data)

# doh_temp = df_doh.head(20000).copy()
# non_doh_temp = df_non_doh.head(20000).copy()
# benign_temp = df_benign.head(20000).copy()
# malicious_temp = df_malicious.head(20000).copy()

# columns_to_drop = ['TimeStamp','SourceIP', 'DestinationIP' ,'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']

# # #zamieniamy Lable na 0 i 1 i 2

# non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
# doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
# benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
# malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 2)

# merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

# merged_df.drop(columns=columns_to_drop, inplace=True)
# print(merged_df.shape[0])
# print(merged_df.shape[1])
# merged_df.to_csv('merged_sample.csv', index=False)



#Making huge amnount of data for BIG testing


skip_count = 5000  # How many rows to skip
read_count = 10000  # How many rows to read


rows_to_skip = range(1, skip_count + 1)


df_doh = pd.read_csv('CSVs/Total_CSVs/l1-doh.csv', skiprows=rows_to_skip, nrows=read_count)
df_non_doh = pd.read_csv('CSVs/Total_CSVs/l1-nondoh.csv', skiprows=rows_to_skip, nrows=read_count)
df_benign = pd.read_csv('CSVs/Total_CSVs/l2-benign-smote.csv', skiprows=rows_to_skip, nrows=read_count)
df_malicious = pd.read_csv('CSVs/Total_CSVs/l2-malicious.csv', skiprows=rows_to_skip, nrows=read_count)


doh_temp = df_doh.head(10000).copy()
non_doh_temp = df_non_doh.head(10000).copy()
benign_temp = df_benign.head(10000).copy()
malicious_temp = df_malicious.head(10000).copy()

columns_to_drop = ['TimeStamp','SourceIP', 'DestinationIP' , 'SourcePort', 'DestinationPort' ,'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']

# #zamieniamy Lable na 0 i 1 i 2

non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 2)

merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

merged_df.drop(columns=columns_to_drop, inplace=True)
print(merged_df.shape[0])
print(merged_df.shape[1])
merged_df.to_csv('sample.csv', index=False)
