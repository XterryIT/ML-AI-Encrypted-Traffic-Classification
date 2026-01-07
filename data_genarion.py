from py_compile import main
import pandas as pd


def opening_regular_data():
    df_non_doh = pd.read_csv('CSVs/Total_CSVs/l1-nondoh.csv')
    df_benign = pd.read_csv('CSVs/Total_CSVs/l2-benign.csv')
    df_malicious = pd.read_csv('CSVs/Total_CSVs/l2-malicious.csv')

    return df_non_doh, df_benign, df_malicious

def parsing(n, columns_to_drop):
    df_non_doh, df_benign, df_malicious = opening_regular_data()

    if n != 0:
        non_doh_temp = df_non_doh.head(n).copy()
        benign_temp = df_benign.head(n).copy()
        malicious_temp = df_malicious.head(n).copy()
    else:
        non_doh_temp = df_non_doh.copy()
        benign_temp = df_benign.copy()
        malicious_temp = df_malicious.copy()

    mapping = {'Benign': 1, 'Malicious': 2, 'NonDoH': 0}

    non_doh_temp['Label'] = non_doh_temp['Label'].map(mapping)
    benign_temp['Label'] = benign_temp['Label'].map(mapping)
    malicious_temp['Label'] = malicious_temp['Label'].map(mapping)

    merged_df = pd.concat([non_doh_temp, benign_temp, malicious_temp], ignore_index=True)
    merged_df.drop(columns=columns_to_drop, inplace=True)

    print("Amount of data:", merged_df.shape[0])
    print("Amount of attributes:", merged_df.shape[1])

    return merged_df

def mod():
    columns = ['SourceIP', 'DestinationIP', 'TimeStamp', 'SourcePort', 'DestinationPort'] # this should always be dropped
    c = input("Dropping more than basic columns? (y for yes)")
    if c == "y":
        columns.append['PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']
        print(columns)
    
    n = int(input("Dataset lines? (0 for all)")) # sets how many lines it takes from every file # set to 0 to get everything from the file
    merged_df = parsing(n, columns)
    if n == 0:
        if c == 'y':
            merged_df.to_csv('data/chosen_params.csv', index=False)
        else:
            merged_df.to_csv('data/all_params.csv', index=False)
    else:
        if c == 'y':
            merged_df.to_csv(f"data/{n}_chosen_params.csv", index=False)
        else:
            merged_df.to_csv(f"data/{n}_all_params.csv", index=False)


def main():
    mod()

if __name__ == "__main__":
    main()