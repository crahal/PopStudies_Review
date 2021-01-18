import pandas as pd
import numpy as np
import os
import re
import gender_guesser.detector as gender
from gender_detector import gender_detector as gd


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
#    auth_df['gender_guesser'] = np.where(auth_df['forename']=='nan', 'unknown', auth_df['gender_guesser'])
#    auth_df['gender_detector'] = np.where(auth_df['forename']=='nan', 'unknown', auth_df['gender_detector'])
#    auth_df['clean_gender'] = np.where(auth_df['forename']=='nan', 'unknown', auth_df['clean_gender'])
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Bangladesh Institute for Development Studies', 'Bangladesh Institute of Development Studies')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Biomedical Res./Training Institute', 'Biomedical Research and Training Institute')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('California Institute for Technology', 'California Institute of Technology')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('China Population and Development Research Centre', 'China Population and Development Research Center')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Social Policy and Intervention and Green Templeton College, Oxford University', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Social Policy and Intervention and St John’s College, Oxford University', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Social Policy and Intervention, University of Oxford and Green Templeton College', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Social Policy and Intervention, University of Oxford', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Sociology and Anthropology', 'Brown University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Sociology and Nuffield College, Oxford University', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Sociology and Nuffield College, University of Oxford', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Sociology and Population Studies Center, University of Pennsylvania', 'University of Pennsylvania')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Department of Sociology, University of Pennsylvania', 'University of Pennsylvania')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Cambridge Group for the History of Population and Social Structure', 'University of Cambridge')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Danish Center for Demographic Res.', 'Danish Center for Demographic Research')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Duke University Medical Center', 'Duke University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('ESRC Cambridge Group for the History of Population and Social Structure', 'University of Cambridge')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Erasmus University Medical Center', 'Erasmus University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Erasmus University Rotterdam', 'Erasmus University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('European Bank for Reconstr./Devmt.', 'European Bank for Reconstruction and Development')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Harvard Center for Population Studies', 'Harvard University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Harvard Center for Population and Development Studies', 'Harvard University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Labor and Worklife Program, Harvard Law School', 'Harvard University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Harvard School of Public Health', 'Harvard University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Harvard Center for Population Studies', 'Harvard University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Hebrew University of Jerusalem', 'Hebrew University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('IZA Institute of Labor Economics', 'IZA')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institute for the Study of Labor \(IZA\)', 'IZA')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Johns Hopkins Bloomberg School of Public Health', 'Johns Hopkins University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of Economics', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of Economics & Political Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of Economics and Political Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of EconomicsPolitical Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of Economies and Political Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of Hygiene & Tropical Medicine', 'LSHTM')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('London School of Hygiene and Tropical Medicine', 'LSHTM')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Innocenzo Gasparini Inst. Econ. Res.', 'Innocenzo Gasparini Institute for Economic Research')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut National d\'Etudes Demographiques (INED)', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut National d\'Études Démographiques', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut National d\'Études Démographiques (INED)', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut National d-Etudes Demographiques', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut National d’Etudes Démographiques', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut National d’Études Démographiques', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut national d’études démographiques', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institut national d’études démographiques (INED)', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Institute for Population and Development Studies, School of Public Administration and Policy, Xi’an Jiaotong University', 'Xi\'an Jiaotong University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Max Planck Inst. Demographic Res.', 'Max Planck')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Max Planck Inst. for Demogr. Res.', 'Max Planck')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Max Planck Institute Demogr. Res.', 'Max Planck')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Max Planck Institute for Demographic Research', 'Max Planck')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('National Bureau of Economic Research (NBER)', 'NBER')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('National Bureau of Economic Research', 'NBER')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Netherlands Interdisciplinary Demographic Institute/KNAW/ University of Groningen', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Netherlands Interdisciplinary Demographic Institute, University of Groningen', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Netherlands Interuniversity Demographic Institute (N.I.D.I)', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Netherlands Interdisciplinary Demographic Institute', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Netherlands Interdisciplinary Demogr', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Population Councilacty', 'Population Council')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Nuffield College', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Radboud University Nijmegen', 'Radboud University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Radboud University of Nijmegen', 'Radboud University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Somerville College and the Institute of Human Sciences, Oxford University', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('State University of New York', 'SUNY')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('SUNY-Stony Brook', 'SUNY')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Transactional Family Research Institute', 'Transnational Family Research Institute')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Transnational Family Research Inst.', 'Transnational Family Research Institute')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('UNICEF Innocenti Research Centre', 'UNICEF')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('United Nations Statistics Division', 'UN')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('United Nations Population Fund (UNFPA)', 'UN')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('United Nations Economic Commission for Europe', 'UN')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('United Nations Population Division', 'UN')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Universidad Complutense-Madrid', 'Universidad Complutense de Madrid')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Universidad Complutense de Madrid, Campus de Somosoguas', 'Universidad Complutense de Madrid')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University College London', 'UCL')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of California Los Angeles', 'University of California')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University College', 'UCL')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Florence, Department of Statistics, Informatica, Applicazioni ‘Giuseppe Parenti’', 'University of Florence')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Illinois at Chicago', 'University of Illinois')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Texas School of Public Health', 'University of Texas')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Texas at Austin', 'University of Texas')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Texas at El Paso', 'University of Texas')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Texas-Austin', 'University of Texas')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Wisconsin-Green Bay', 'University of Wisconsin')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Wisconsin–Madison', 'University of Wisconsin')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Wisconsin-Stevens Point', 'University of Wisconsin')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Wisconsin-Madison', 'University of Wisconsin')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Vienna Institute of Demography of the Austrian Academy of Sciences', 'Vienna Institute of Demography')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('VU University Amsterdam', 'VU University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Vrije Universiteit Amsterdam', 'Vrije Universiteit')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Vrije Universiteit Brussel', 'Vrije Universiteit')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Yale University and the Hoover Institution', 'Yale University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('African Population/Health Res. Ctr.', 'African Population and Health Research Center')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('All Souls College', 'University of Oxford')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Brown University, Brown University', 'Brown University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Center for Population and Development Studies, Renmin University of China', 'Renmin University')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('ESRC University of Cambridge', 'University of Cambridge')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('ICDDR,B', 'I.C.D.D.R.B')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('ICDDR,B', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Inst. Natl. d\'Etudes Demographiques', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Inst. nat. d\'etudes dem', 'INED')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('International Development Research Centre', 'IDRC')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('International Development Research Centre (IDRC)', 'IDRC')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('International Statistical Institute Research Centre', 'International Statistical Institute')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('International Statistical Research Centre', 'International Statistical Institute')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('LSE & Political Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('LSE and Political Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('LSEPolitical Science', 'LSE')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Medical Research Council’s Institute of Medical Sociology', 'Medical Research Council')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('NBER \(NBER\)', 'NBER')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('NIDI \(NIDI\)', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('The RAND', 'RAND')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('NORC at the University of Chicago', 'University of Chicago')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('RAND Europe', 'RAND')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('RAND Corporation', 'RAND')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Rand Corporation', 'RAND')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Rome \'La Sapienza', 'University of Rome La Sapienza')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of North Carolina at Chapel Hill', 'University of North Carolina')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Rand Graduate Institute', 'RAND')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Statistics Netherlands \(CBS\)', 'Statistics Netherlands')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('The Max Planck', 'Max Planck')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('The NIDI', 'NIDI')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Transnational Family Research Institutetute', 'Transnational Family Research Institute')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('United Nations Population Fund \(UNFPA\)', 'United Nations')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University Medical Center Groningen', 'University of Groningen')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Hawai\'i at Manoa', 'University of Hawaii')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University of Leuven \(KU Leuven\)', 'KU Leuven')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('University Washington', 'University of Washington')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('School of Aquatic and Fishery Sciences & Center for the Study of Demography and Ecology, University of Washington', 'University of Washington')
    auth_df['aff_orgs'] = auth_df['aff_orgs'].str.replace('Independent Consultant Based in Dorking', 'Independent Consultant')
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

    return main_df, ref_df, auth_df


def load_curated(d_path, filename):
    manual_curated = pd.read_csv(os.path.join(d_path, 'manual_review', 'final',
                                              filename), encoding='ISO-8859-1')
    return manual_curated.drop_duplicates(subset=['DOI'])
