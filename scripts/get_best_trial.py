import pandas


if __name__ == '__main__':
    
    data = pandas.read_csv('trace_dump.csv')

    # make sure it's a search job
    data = data.loc[data['job'].str.contains('search')]

    # make sure it's validation data
    data = data.loc[data['split'].str.contains('valid')]

    # return folder of best trial
    data = data.sort_values('metric', ascending=False)
    print(str(data['child_folder'].iloc[0].item()).zfill(5))
        
