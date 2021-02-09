import pandas as pd
import numpy as np
import nltk
import os
import gender_guesser.detector as gender
from gender_detector import gender_detector as gd
import re
from popstudies_lda import reflection_tokenizer


def make_stopwords():
    stop_list = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'support',
                                         'custom_stopwords.txt'))
    custom_stop = stop_list['words'].to_list()
    stop = nltk.corpus.stopwords.words('english')
    for word in custom_stop:
        stop.append(word)
    return stop

def make_lemmas(df, d_path):
    df['lemmatize_token_abstract'] = df['clean_abstract'].apply(reflection_tokenizer)
    df['lemmatize_token_str_abstract'] = df['lemmatize_token_abstract'].agg(lambda x: ', '.join(map(str, x)))
    df['lemmatize_token_str_abstract'] = df['lemmatize_token_str_abstract'].str.replace(',', '')
    count = df['lemmatize_token_str_abstract'].apply(lambda x: pd.value_counts(x.split(" ")))
    count = count.sum(axis=0)
    count.sort_values(ascending=False).to_csv(os.path.join(d_path, 'support',
                                                           'wordcounts', 'abstract_lemmatized.csv'),
                                              header=True)

    df['lemmatize_token_title'] = df['Title'].apply(reflection_tokenizer)
    df['lemmatize_token_str_title'] = df['lemmatize_token_title'].agg(lambda x: ', '.join(map(str, x)))
    df['lemmatize_token_str_title'] = df['lemmatize_token_str_title'].str.replace(',', '')
    count = df['lemmatize_token_str_title'].apply(lambda x: pd.value_counts(x.split(" ")))
    count = count.sum(axis=0)
    count.sort_values(ascending=False).to_csv(os.path.join(d_path, 'support',
                                                           'wordcounts', 'title_lemmatized.csv'),
                                              header=True)
    return df


def clean_abstract(abstract_input):
    if ('©' in abstract_input) and not (abstract_input.startswith('©')):
        abstract_input = abstract_input.replace('©', '.©')
    abstract = re.sub(r'http\S+', '', abstract_input)
    abs_str=''
    abs_split = abstract.split('.')
    for splitter in abs_split:
        if ('©' not in splitter) and\
           ('Taylor' not in splitter) and\
           ('Francis' not in splitter) and \
           ('population investigation committee' not in splitter.lower()) and\
           ('material is available' not in splitter):
            abs_str = abs_str + splitter + '. '
    abs_str = abs_str.replace('. .', '.')
    abs_str = abs_str.replace('  ', ' ')
    return abs_str.strip()


def get_gender_guess(x, d):
    """Get gender guess"""
    if x.lower() == 'nan':
        return 'unknown'
    else:
        return d.get_gender(x)


def get_gender_detect(x, detector):
    if x.lower() == 'nan':
        return 'unknown'
    else:
        return detector.guess(x)


def build_datasets(d_path):
    """Build datasets for notebook analysis and RA."""
    main_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                       'search', 'parsed',
                                       'scopus_search_meta.tsv'),
                          sep='\t', encoding='ISO-8859-1')
    abs_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                      'abstract', 'parsed',
                                      'scopus_abstract.tsv'),
                         sep='\t', encoding='ISO-8859-1')
    ref_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                      'abstract', 'parsed',
                                      'scopus_references.tsv'),
                         sep='\t', encoding='ISO-8859-1')
    auth_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                       'abstract', 'parsed',
                                       'scopus_aff_auth.tsv'),
                          sep='\t', encoding='utf-8')
    plumx_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                        'plumx', 'parsed',
                                        'scopus_plumx.tsv'),
                           sep='\t', encoding='utf-8')
    abs_df = abs_df.drop_duplicates(subset=['doi'], keep='first')
    plumx_df = plumx_df.drop_duplicates(subset=['doi'], keep='first')
    main_df = pd.merge(main_df, abs_df, how='left',
                       left_on='DOI', right_on='doi')
    main_df = pd.merge(main_df, plumx_df, how='left',
                       left_on='DOI', right_on='doi')
    main_df.to_csv(os.path.join(d_path, 'scopus',
                                'merged_and_clean',
                                'main_df.tsv'), sep='\t',
                   encoding='utf-8')
    ref_df.to_csv(os.path.join(d_path, 'scopus',
                               'merged_and_clean',
                               'ref_df.tsv'), sep='\t',
                  encoding='utf-8', index=False)
    auth_df.to_csv(os.path.join(d_path, 'scopus',
                                'merged_and_clean',
                                'auth_df.tsv'), sep='\t',
                   encoding='utf-8')


    #auth_df = auth_df.drop_duplicates(subset=['doi', 'indexed_name'])
    #auth_df = auth_df[auth_df['forename'].notnull()]
    auth_df['forename'] = auth_df['forename'].astype(str)
    auth_df['forename'] = auth_df['forename'].str.replace('Ø', 'O')
    auth_df['forename'] = auth_df['forename'].str.replace('É', 'E')
    auth_df['forename'] = auth_df['forename'].str.split(' ', expand=False).str[0]
    d = gender.Detector()
    detector = gd.GenderDetector('uk')
    auth_df['gender_guesser'] = auth_df['forename'].apply(lambda x : get_gender_guess(x, d))
    auth_df['gender_detector'] = auth_df['forename'].apply(lambda x : get_gender_detect(x, detector))
    auth_df['gender_guesser'] = auth_df['gender_guesser'].str.replace('mostly_female', 'female')
    auth_df['gender_guesser'] = auth_df['gender_guesser'].str.replace('mostly_male', 'male')
    auth_df['gender_guesser'] = auth_df['gender_guesser'].str.replace('andy', 'unknown')
    auth_df['clean_gender'] = np.nan
    for index, row in auth_df.iterrows():
        if row['gender_guesser'] == row['gender_detector']:
            auth_df.loc[index, 'clean_gender'] = row['gender_detector']
        elif (row['gender_guesser'] == 'female') and (row['gender_detector'] == 'unknown'):
            auth_df.loc[index, 'clean_gender'] = 'female'
        elif (row['gender_guesser'] == 'unknown') and (row['gender_detector'] == 'female'):
            auth_df.loc[index, 'clean_gender'] = 'female'
        elif (row['gender_guesser'] == 'male') and (row['gender_detector'] == 'unknown'):
            auth_df.loc[index, 'clean_gender'] = 'male'
        elif (row['gender_guesser'] == 'unknown') and (row['gender_detector'] == 'male'):
            auth_df.loc[index, 'clean_gender'] = 'male'
    org_cleaner = pd.read_csv(os.path.join(d_path, 'support', 'org_cleaner.csv'))
    for index, row in org_cleaner.iterrows():
        auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace(row['original'], row['clean'], regex=False)
    for index, row in main_df.iterrows():
        abstract = re.sub(r'http\S+', '', str(row['abstract']))
        abs_str=''
        abs_split = abstract.split('.')
        for splitter in abs_split:
            if ('©' not in splitter) and\
               ('Taylor' not in splitter) and\
               ('Francis' not in splitter) and\
               ('material is available' not in splitter):
                abs_str = abs_str + splitter + '. '
            abs_str = abs_str.replace('. .', '.')
            abs_str = abs_str.replace('  ', ' ')
        main_df.loc[index, 'abstract'] = abs_str

    main_df = main_df.drop('doi_x', 1)
    main_df = main_df.drop('doi_y', 1)
    main_df = main_df.drop_duplicates(subset=['DOI'])
    main_df = main_df.drop_duplicates(subset=['Title', 'prismpagerange'])

    auth_df['forename'] = auth_df['forename'].str.upper()
    auth_df['surname'] = auth_df['surname'].str.upper()
    auth_df['indexed_name'] = auth_df['indexed_name'].str.upper()
    auth_df = auth_df[auth_df['doi'].isin(main_df['DOI'])]
    ref_df = ref_df[ref_df['doi'].isin(main_df['DOI'])]
    main_df['pagestart'] = pd.to_numeric(main_df['pagestart'], errors='coerce')
    main_df['pageend'] = pd.to_numeric(main_df['pageend'], errors='coerce')
    main_df['refcount'] = pd.to_numeric(main_df['refcount'], errors='coerce')
    curated = load_curated(d_path, 'popstudies_manal_review_final.csv')
    main_df = pd.merge(main_df, curated[['Topic', 'Subnation_popstudied', 'Regions',
                                         'Nation', 'Population', 'Dataset',
                                         'Time', 'DOI']],
                       how='left', left_on='DOI', right_on='DOI')

    auth_df = pd.merge(auth_df, main_df[['DOI', 'prismcoverdate']],
                       how='left', left_on='doi', right_on='DOI')

    main_df['Topic'] = main_df['Topic'].str.replace(':', ';')
    main_df['Topic'] = main_df['Topic'].str.replace(',', ';')
    main_df['Topic'] = main_df['Topic'].str.strip()
    main_df['Topic'] = main_df['Topic'].astype(str)
    main_df['abstract'] = main_df['abstract'].astype(str)


    def continent_merger(main_df, d_path):
        continent_merger = pd.read_csv(os.path.join(d_path,
                                                    'support',
                                                    'continent_merger.csv'))
        main_df = pd.merge(main_df, continent_merger,
                           how='left', left_on=['Regions', 'Nation'],
                           right_on=['Regions', 'Nation'])
        main_df['Continent'] = main_df['Continent'].str.replace('asia', 'Asia')
        return main_df

    main_df = continent_merger(main_df, d_path)
    main_df['clean_abstract'] = main_df['abstract'].apply(clean_abstract)
    main_df['abstract_length'] = main_df    ['clean_abstract'].str.len()
    main_df = make_lemmas(main_df, d_path)
    return main_df, ref_df, auth_df


def load_curated(d_path, filename):
    manual_curated = pd.read_csv(os.path.join(d_path, 'manual_review', 'final',
                                              filename), encoding='ISO-8859-1')
    return manual_curated.drop_duplicates(subset=['DOI'])
