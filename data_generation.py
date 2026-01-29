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

    mapping = {'NonDoH': 0, 'Benign': 1, 'Malicious': 2}

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
    # features: 'Duration', 'FlowBytesSent', 'FlowSentRate', 'FlowBytesReceived', 'FlowReceivedRate', 'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation'
    # columns to drop for each analyzed algorithm
    all = []
    rf = ['PacketLengthVariance', 'FlowBytesReceived', 'PacketTimeStandardDeviation', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'FlowSentRate', 'FlowReceivedRate', 'PacketLengthSkewFromMedian', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation'] # v2: ['FlowBytesReceived', 'PacketTimeStandardDeviation', 'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'FlowSentRate', 'FlowReceivedRate', 'PacketLengthSkewFromMedian', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation'] # v1: ['FlowSentRate', 'FlowReceivedRate', 'PacketLengthSkewFromMedian', 'PacketTimeMean', 'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']
    knn = ['FlowBytesSent', 'PacketLengthVariance', 'FlowSentRate', 'FlowReceivedRate', 'PacketLengthMedian', 'PacketLengthMode', 'PacketTimeMode', 'PacketTimeSkewFromMode', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMode', 'FlowBytesReceived'] # v1: ['FlowBytesSent', 'PacketLengthVariance', 'FlowSentRate', 'FlowReceivedRate', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMode', 'PacketTimeMode', 'PacketTimeSkewFromMode', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMode', 'FlowBytesReceived']
    mlp = ['FlowReceivedRate', 'PacketLengthMedian', 'PacketTimeMode', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']
    # modify to drop more columns
    columns.extend(rf)
    print("Dropping", columns)
    
    n = 0 # sets how many lines it takes from every file # set to 0 to get everything from the file
    merged_df = parsing(n, columns)
    merged_df.to_csv('data/rf_v3.csv', index=False) # naming: n(if less data is taken)_[algorithm_name](for which features are chosen).csv


def main():
    mod()

if __name__ == "__main__":
    main()