from py_compile import main
import pandas as pd


def opening_regular_data():
    df_doh = pd.read_csv('CSVs/Total_CSVs/l1-doh.csv')
    df_non_doh = pd.read_csv('CSVs/Total_CSVs/l1-nondoh.csv')
    df_benign = pd.read_csv('CSVs/Total_CSVs/l2-benign.csv')
    df_malicious = pd.read_csv('CSVs/Total_CSVs/l2-malicious.csv')

    return df_doh, df_non_doh, df_benign, df_malicious


def small_data_separate_params():
    df_doh, df_non_doh, df_benign, df_malicious = opening_regular_data()


    doh_temp = df_doh.head(20000).copy()
    non_doh_temp = df_non_doh.head(20000).copy()
    benign_temp = df_benign.head(20000).copy()
    malicious_temp = df_malicious.head(20000).copy()

    columns_to_drop = ['SourcePort', 'DestinationPort', 'TimeStamp', 'SourceIP', 'DestinationIP' ,'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']

# #zamieniamy Lable na 0 i 1 i 2

    non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
    doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
    benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
    malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 2)

    merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

    merged_df.drop(columns=columns_to_drop, inplace=True)
    print(merged_df.shape[0])
    print(merged_df.shape[1])
    merged_df.to_csv('data/small_data_separate.csv', index=False)


def small_data_all_params():
    df_doh, df_non_doh, df_benign, df_malicious = opening_regular_data()


    doh_temp = df_doh.head(20000).copy()
    non_doh_temp = df_non_doh.head(20000).copy()
    benign_temp = df_benign.head(20000).copy()
    malicious_temp = df_malicious.head(20000).copy()

    columns_to_drop = ['SourceIP', 'DestinationIP', 'TimeStamp', 'SourcePort', 'DestinationPort']

# #zamieniamy Lable na 0 i 1 i 2

    non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
    doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
    benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
    malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 2)

    merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

    merged_df.drop(columns=columns_to_drop, inplace=True)
    print(merged_df.shape[0])
    print(merged_df.shape[1])
    merged_df.to_csv('data/small_data_all.csv', index=False)
# #Data Version 3 for AI 3 (Training data)

def all_params():
    df_doh, df_non_doh, df_benign, df_malicious = opening_regular_data()


    doh_temp = df_doh.copy()
    non_doh_temp = df_non_doh.copy()
    benign_temp = df_benign.copy()
    malicious_temp = df_malicious.copy()

    columns_to_drop = ['SourceIP', 'DestinationIP', 'TimeStamp', 'SourcePort', 'DestinationPort']

# #zamieniamy Lable na 0 i 1 i 2

    non_doh_temp['Label'] = non_doh_temp['Label'].replace('NonDoH', 0)
    doh_temp['Label'] = doh_temp['Label'].replace('DoH', 1)
    benign_temp['Label'] = benign_temp['Label'].replace('Benign', 1)
    malicious_temp['Label'] = malicious_temp['Label'].replace('Malicious', 2)

    merged_df = pd.concat([non_doh_temp, doh_temp, benign_temp, malicious_temp], ignore_index=True)

    merged_df.drop(columns=columns_to_drop, inplace=True)
    print(merged_df.shape[0])
    print(merged_df.shape[1])
    merged_df.to_csv('data/all_params.csv', index=False)



def main():
    all_params()

if __name__ == "__main__":
    main()