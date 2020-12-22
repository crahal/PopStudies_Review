import os
import math
import json
import re
import requests
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm


def build_datasets(d_path):
    """Build datasets for notebook analysis and RA."""
    main_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                       'search', 'parsed',
                                       'scopus_search_meta.tsv'),
                          sep='\t', encoding='utf-8')
    abs_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                      'abstract', 'parsed',
                                      'scopus_abstract.tsv'),
                         sep='\t', encoding='utf-8')
    ref_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                      'abstract', 'parsed',
                                      'scopus_references.tsv'),
                         sep='\t', encoding='utf-8')
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
    main_df['refcount']  = pd.to_numeric(main_df['refcount'], errors='coerce')
    return main_df, ref_df, auth_df


def load_curated(d_path, filename):
    manual_curated = pd.read_csv(os.path.join(d_path, 'manual_review', 'final',
                                              filename))
    return manual_curated.drop_duplicates(subset=['DOI'])

def load_token(token_name):
    """Read supplementary text file."""
    try:
        with open(token_name, 'r') as file:
            return str(file.readline()).strip()
    except EnvironmentError:
        print('Error loading access token from file')


def call_scopus_search_api(url, apikey):
    api_return = requests.get(url+'&apiKey='+apikey)
    return api_return


def scopus_plumx(doi_list, d_path, apikey):
    base_url = 'https://api.elsevier.com/analytics/plumx/doi/'
    count = 0
    plumx_path = os.path.join(d_path, 'scopus', 'plumx')
    with open(os.path.join(plumx_path, 'parsed', 'scopus_plumx.tsv'),
              'w', encoding='utf-8') as tsvfile:
        plumx = csv.writer(tsvfile, delimiter='\t',
                               lineterminator='\n')
        plumx.writerow(['doi','plumx_capture_total',
                        'plumx_citation_total',
                        'plumx_socialmedia_total',
                        'plumx_abstr_views',
                        'plumx_full_views',
                        'plumx_linkouts'])
        for doi in tqdm(doi_list):
            doi = str(doi)
            url = base_url + doi
            api_return = requests.get(url, headers = {"X-ELS-APIKey": apikey})
            api_json = json.loads(api_return.content)
            with open(os.path.join(plumx_path, 'raw', 'file_' + doi.replace('\\', '').replace('/', '') + '.json'),
                      'w', encoding='utf-8') as outfile:
                json.dump(api_json, outfile)
            plumx_capture_total = np.nan
            plumx_citation_total = np.nan
            plumx_socialmedia_total = np.nan
            plumx_abstr_views = np.nan
            plumx_full_views = np.nan
            plumx_linkouts = np.nan
            if 'count_categories' in api_json:
                for cat in api_json['count_categories']:
                    try:
                        if cat['name'] == 'capture':
                            try:
                                plumx_capture_total = cat['total']
                            except (KeyError, TypeError):
                                plumx_capture_total = np.nan
                        elif cat['name'] == 'citation':
                            try:
                                plumx_citation_total = cat['total']
                            except (KeyError, TypeError):
                                plumx_citation_total = np.nan
                        elif cat['name'] == 'socialMedia':
                            try:
                                plumx_socialmedia_total = cat['total']
                            except (KeyError, TypeError):
                                plumx_socialmedia_total = np.nan
                        elif cat['name'] == 'usage':
                            try:
                                for use in cat['count_types']:
                                    if use['name'] == 'ABSTRACT_VIEWS':
                                        try:
                                            plumx_abstr_views = use['total']
                                        except (KeyError, TypeError):
                                            plumx_abstr_views = np.nan
                                    elif use['name'] == 'FULL_TEXT_VIEWS':
                                        try:
                                            plumx_full_views = use['total']
                                        except (KeyError, TypeError):
                                            plumx_full_views = np.nan
                                    elif use['name'] == 'LINK_OUTS':
                                        try:
                                            plumx_linkouts = use['total']
                                        except (KeyError, TypeError):
                                            plumx_linkouts = np.nan
                            except (KeyError, TypeError):
                                plumx_abstr_views = np.nan
                                plumx_full_views = np.nan
                                plumx_linkouts = np.nan
                    except (KeyError, TypeError):
                        pass
                plumx.writerow([doi, plumx_capture_total,
                                plumx_citation_total,
                                plumx_socialmedia_total,
                                plumx_abstr_views,
                                plumx_full_views,
                                plumx_linkouts])


def scopus_abstract(apikey, d_path, doi_list):
    print('****** Building Population Studies Abstracts ******')
    base_url = 'https://api.elsevier.com/content/abstract/doi/'
    abs_path = os.path.join(d_path, 'scopus', 'abstract')
    count = 0
    with open(os.path.join(abs_path, 'parsed', 'scopus_abstract.tsv'),
              'w', encoding='utf-8') as tsvfile:
        abstract = csv.writer(tsvfile, delimiter='\t',
                               lineterminator='\n')
        abstract.writerow(['doi', 'date-delivered', 'abstract',
                           'keywords', 'volume', 'issue', 'pagestart',
                           'pageend', 'refcount', 'subjects'])

        with open(os.path.join(abs_path, 'parsed', 'scopus_aff_auth.tsv'),
                  'w', encoding='utf-8') as tsvfile:
            aff_auth = csv.writer(tsvfile, delimiter='\t',
                                   lineterminator='\n')
            aff_auth.writerow(['doi', 'aff_sourcetext', 'aff_orgs',
                               'aff_country', 'aff_id', 'forename',
                               'surname', 'indexed_name', 'authorid',
                               'author_order'])
            with open(os.path.join(abs_path, 'parsed', 'scopus_references.tsv'),
                      'w', encoding='utf-8') as tsvfile:
                references = csv.writer(tsvfile, delimiter='\t',
                                       lineterminator='\n')
                references.writerow(['doi', 'ref_id', 'ref_fulltitle',
                                     'ref_title', 'ref_doi',
                                     'ref_pui', 'ref_car',
                                     'ref_geo', 'ref_scopus',
                                     'ref_scp', 'ref_fragmentid',
                                     'ref_medl',
                                     'ref_sgr', 'ref_arctfs'])
                for doi in tqdm(doi_list):
                    doi = str(doi)
                    url = base_url + doi
                    api_return = requests.get(url+'?apiKey=' + apikey,
                                              headers={"Accept":
                                                       "application/json"})
                    api_json = json.loads(api_return.content)

                    with open(os.path.join(abs_path, 'raw',
                                          'file_' + doi.replace('\\', '').replace('/', '') + '.json'),
                              'w', encoding='utf-8') as outfile:
                        json.dump(api_json, outfile)
                    if count == 0:
                        remaining = api_return.headers['X-RateLimit-Remaining']
                        print(str(remaining) + ' calls remaining')
                        count = count + 1
                    try:
                        head = api_json['abstracts-retrieval-response']['item']\
                                       ['bibrecord']['head']
                        tail = api_json['abstracts-retrieval-response']['item']\
                                       ['bibrecord']['tail']
                        try:
                            abs_date = api_json['abstracts-retrieval-response']\
                                               ['item']['ait:process-info']\
                                               ['ait:date-delivered']\
                                               ['@timestamp']
                        except (KeyError, TypeError):
                            abs_date = np.nan
                        try:
                            abst = head['abstracts']
                        except (KeyError, TypeError):
                            abst = np.nan
                        try:
                            keywords = ''
                            for keyword in head['citation-info']\
                                               ['author-keywords']\
                                               ['author-keyword']:
                                keywords = keywords + keyword['$'] + '; '
                            keywords = keywords[:-2]
                        except (KeyError, TypeError):
                            keywords = np.nan
                        try:
                            volume = head['source']['volisspag']['voliss']['@volume']
                        except (KeyError, TypeError):
                            volume = np.nan
                        try:
                            issue = head['source']['volisspag']['voliss']['@issue']
                        except (KeyError, TypeError):
                            issue = np.nan
                        try:
                            pagestart = head['source']['volisspag']['pagerange']['@first']
                        except (KeyError, TypeError):
                            pagestart = np.nan
                        try:
                            pageend = head['source']['volisspag']['pagerange']['@last']
                        except (KeyError, TypeError):
                            pageend = np.nan
                        try:
                            refcount = tail['bibliography']['@refcount']
                        except (KeyError, TypeError):
                            refcount = np.nan
                        try:
                            subjects = ''
                            for subject in api_json['abstracts-retrieval-response']\
                                                   ['subject-areas']['subject-area']:
                                subjects = subjects + subject['@abbrev'] + '; '
                            subjects = subjects[:-2]
                        except (KeyError, TypeError):
                            subjects = np.nan
                        abstract.writerow([doi, abs_date, abst, keywords,
                                           volume, issue, pagestart, pageend,
                                           refcount, subjects])
                        try:
                            auth_group = head['author-group']
                            if type(auth_group) is dict:
                                temp_list = []
                                temp_list.append(auth_group)
                                auth_group = temp_list
                            for affil in auth_group:
                                try:
                                    aff_country = affil['affiliation']['country']
                                except (KeyError, TypeError):
                                    aff_country = np.nan
                                try:
                                    aff_sourcetext = affil['affiliation']['ce:source-text']
                                except (KeyError, TypeError):
                                    aff_sourcetext = np.nan
                                try:
                                    aff_id = affil['affiliation']["affiliation-id"]["@afid"]
                                except (KeyError, TypeError):
                                    aff_id = np.nan
                                try:
                                    if type(affil['affiliation']['organization']) is dict:
                                        org_list = []
                                        org_list.append(affil['affiliation']['organization'])
                                        org_list = org_list
                                        aff_orgs = ''
                                        for org in org_list:
                                            aff_orgs = aff_orgs + org['$'] + '; '
                                        aff_orgs = aff_orgs[:-2]
                                except (KeyError, TypeError):
                                    aff_orgs = np.nan
                                for auth in affil['author']:
                                    try:
                                        forename = auth['ce:given-name']
                                    except (KeyError, TypeError):
                                        forename = np.nan
                                    try:
                                        surname = auth['ce:surname']
                                    except (KeyError, TypeError):
                                        surname = np.nan
                                    try:
                                        indexed_name = auth['ce:indexed-name']
                                    except (KeyError, TypeError):
                                        indexed_name = np.nan
                                    try:
                                        authorid = auth['@auid']
                                    except (KeyError, TypeError):
                                        authorid = np.nan
                                    try:
                                        author_order = auth['@seq']
                                    except (KeyError, TypeError):
                                        author_order = np.nan
                                    aff_auth.writerow([doi, aff_sourcetext,
                                                       aff_orgs,
                                                       aff_country, aff_id, forename,
                                                       surname, indexed_name,
                                                       authorid,
                                                       author_order])
                        except (KeyError, TypeError):
                            aff_auth.writerow([doi, np.nan, np.nan, np.nan,
                                               np.nan, np.nan, np.nan])
                        try:
                            for ref in tail['bibliography']['reference']:
                                try:
                                    ref_title = ref['ref-info']['ref-title']['ref-titletext']
                                except (KeyError, TypeError):
                                    ref_title = np.nan
                                try:
                                    ref_fulltitle = ref['ref-fulltext']
                                except (KeyError, TypeError):
                                    ref_fulltitle = np.nan
                                try:
                                    ref_id = ref['@id']
                                except (KeyError, TypeError):
                                    ref_id = np.nan
                                try:
                                    ref_pui = np.nan
                                    ref_car = np.nan
                                    ref_geo = np.nan
                                    ref_scopus = np.nan
                                    ref_scp = np.nan
                                    ref_fragmentid = np.nan
                                    ref_doi = np.nan
                                    ref_medl = np.nan
                                    ref_sgr = np.nan
                                    ref_arctfs = np.nan
                                    if type(ref['ref-info']['refd-itemidlist']['itemid']) is dict:
                                        ref_list = []
                                        ref_list.append(ref['ref-info']['refd-itemidlist']['itemid'])
                                        ref_list = ref_list
                                    else:
                                        ref_list = ref['ref-info']['refd-itemidlist']['itemid']
                                    for id_type in ref_list:
                                        if id_type['@idtype'] == 'DOI':
                                            ref_doi = id_type['$']
                                        elif id_type['@idtype'] == 'PUI':
                                            ref_pui = id_type['$']
                                        elif id_type['@idtype'] == 'CAR-ID':
                                            ref_car = id_type['$']
                                        elif id_type['@idtype'] == 'GEO':
                                            ref_geo = id_type['$']
                                        elif id_type['@idtype'] == 'SCOPUS':
                                            ref_scopus = id_type['$']
                                        elif id_type['@idtype'] == 'SCP':
                                            ref_scp = id_type['$']
                                        elif id_type['@idtype'] == 'FRAGMENTID':
                                            ref_fragmentid = id_type['$']
                                        elif id_type['@idtype'] == 'MEDL':
                                            ref_medl = id_type['$']
                                        elif id_type['@idtype'] == 'ARCTFS':
                                            ref_arctf= id_type['$']
                                        elif id_type['@idtype'] == 'SGR':
                                            ref_sgr= id_type['$']
                                except (KeyError, TypeError):
                                    pass
                                references.writerow([doi, ref_id, ref_fulltitle,
                                                     ref_title, ref_doi,
                                                     ref_pui, ref_car,
                                                     ref_geo, ref_scopus,
                                                     ref_scp, ref_fragmentid,
                                                     ref_medl, ref_sgr,
                                                     ref_arctfs])
                        except (KeyError, TypeError):
                            ref_title = np.nan
                            ref_fulltitle = np.nan
                            ref_id = np.nan
                            ref_doi = np.nan
                            references.writerow([doi, ref_id, ref_fulltitle,
                                                 ref_title, ref_doi])
                    except KeyError:
                        pass

def search_scopus_into_csv(apikey, d_path, count=100):
    '''
    A function to scrape the scopus api for all mentions of a keyword set

    query list: a colon delimited list in the form of dataset:searchquery
    aqikey:     a free apikey granted on application: note you need to be
                connected to the domain of a registered institution or
                your requests will not be successful...
    count:      the number of returns per page request, default 100 (max)
    '''

    print('****** Search for all Population Studies Papers ******')

    base_url = 'https://api.elsevier.com/content/search/scopus?'
    search_path = os.path.abspath(os.path.join(d_path, 'scopus', 'search'))
    with open(os.path.join(search_path,'parsed', 'scopus_search_meta.tsv'),
              'w', encoding='utf-8') as tsvfile:
        scopus_search = csv.writer(tsvfile, delimiter='\t',
                                   lineterminator='\n')
        scopus_search.writerow(['Title', 'prismurl', 'dcidentifier',
                                'eid', 'dccreator', 'Journal', 'prismissn',
                                'prismeissn', 'prismvolume',
                                'prismissueidentifier', 'prismpagerange',
                                'prismcoverdate', 'Date', 'DOI', 'citedbycount',
                                'affilnames', 'affilcities', 'affilcountries',
                                'prismaggregationtype', 'subtype',
                                'subtypedescription', 'sourceid', 'openaccess',
                                'openaccessFlag'])
        pagenum = 0
        numres = 99999999999999999999999999999999999999999999999999999999
        while pagenum < (math.ceil(int(numres)/count)):
            url = base_url + 'start=' + str((pagenum*count)) + '&count=' +\
                  str(count) +'&query=ISSN(00324728)'
            api_return = call_scopus_search_api(url, apikey)
            api_json = json.loads(api_return.content)
            with open(os.path.join(search_path, 'raw',
                                   'file_' + str(pagenum) + '.json'),
                      'w', encoding='utf-8') as outfile:
                json.dump(api_json, outfile)
            if pagenum == 0:
                numres = api_json['search-results']['opensearch:totalResults']
                print('There are ' + numres + ' Population Studies returned')
                remaining = api_return.headers['X-RateLimit-Remaining']
                print('We have: ' + str(remaining) + ' calls remaining')
            pagenum += 1
            try:
                for entry in api_json['search-results']['entry']:
                    try:
                        Title = entry['dc:title']
                    except KeyError:
                        Title = 'N/A'
                    try:
                        prismurl = entry['prism:url']
                    except KeyError:
                        prismurl = 'N/A'
                    try:
                        dcidentifier = entry['dc:identifier']
                    except KeyError:
                        dcidentifier = 'N/A'
                    try:
                        eid = entry['eid']
                    except KeyError:
                        eid = 'N/A'
                    try:
                        dccreator = entry['dc:creator']
                    except KeyError:
                        dccreator = 'N/A'
                    try:
                        prismpublicationname = entry['prism:publicationName']
                    except KeyError:
                        prismpublicationname = 'N/A'
                    try:
                        prismissn = entry['prism:issn']
                    except KeyError:
                        prismissn = 'N/A'
                    try:
                        prismeissn = entry['prism:eIssn']
                    except KeyError:
                        prismeissn = 'N/A'
                    try:
                        prismvolume = entry['prism:volume']
                    except KeyError:
                        prismvolume = 'N/A'
                    try:
                        prismissueidentifier = entry['prism:issueIdentifier']
                    except KeyError:
                        prismissueidentifier = 'N/A'
                    try:
                        prismpagerange = entry['prism:pageRange']
                    except KeyError:
                        prismpagerange = 'N/A'
                    try:
                        prismcoverdate = entry['prism:coverDate']
                    except KeyError:
                        prismcoverdate = 'N/A'
                    try:
                        prismcoverdisplaydate = entry['prism:coverDisplayDate']
                    except KeyError:
                        prismcoverdisplaydate = 'N/A'
                    try:
                        DOI = entry['prism:doi']
                    except KeyError:
                        DOI = 'N/A'
                    try:
                        citedbycount = entry['citedby-count']
                    except KeyError:
                        citedbycount = 'N/A'
                    try:
                        affilnames = ''
                        affilcities = ''
                        affilcountries = ''
                        for affiliation in entry['affiliation']:
                            affilnames = affilnames + str(affiliation['affilname']) + ':'
                            affilcities = affilcities + str(affiliation['affiliation-city']) + ':'
                            affilcountries = affilcountries + str(affiliation['affiliation-country']) + ':'
                        affilnames = affilnames.replace('None:', '')[:-1]
                        affilcities = affilcities.replace('None:', '')[:-1]
                        affilcountries = affilcountries.replace('None:', '')[:-1]
                    except KeyError:
                        affilnames = 'N/A'
                        affilcities = 'N/A'
                        affilcountries = 'N/A'
                    try:
                        prismaggregationtype = entry['prism:aggregationType']
                    except KeyError:
                        prismaggregationtype = 'N/A'
                    try:
                        subtype = entry['subtype']
                    except KeyError:
                        subtype = 'N/A'
                    try:
                        subtypedescription = entry['subtypeDescription']
                    except KeyError:
                        subtypedescription = 'N/A'
                    try:
                        sourceid = entry['source-id']
                    except KeyError:
                        sourceid = 'N/A'
                    try:
                        openaccess = entry['openaccess']
                    except KeyError:
                        openaccess = 'N/A'
                    try:
                        openaccessflag = entry['openaccessFlag']
                    except KeyError:
                        openaccessflag = 'N/A'
                    scopus_search.writerow([Title, prismurl, dcidentifier,
                                            eid, dccreator, prismpublicationname,
                                            prismissn, prismeissn, prismvolume,
                                            prismissueidentifier, prismpagerange,
                                            prismcoverdate, prismcoverdisplaydate,
                                            DOI, citedbycount,
                                            affilnames, affilcities,
                                            affilcountries, prismaggregationtype,
                                            subtype, subtypedescription,
                                            sourceid, openaccess, openaccessflag])
            except Exception as e:
                print('Something has gone badly wrong?! \n\n' + str(e))
    return pd.read_csv(os.path.join(search_path,
                                    'parsed',
                                    'scopus_search_meta.tsv'),
                       sep='\t')['DOI'].to_list()


def scopus_authors(author_list, d_path, apikey):
    base_url = 'https://api.elsevier.com/content/author/author_id/'
    count = 0
    author_path = os.path.join(d_path, 'scopus', 'author')
    with open(os.path.join(author_path, 'parsed', 'scopus_authors.tsv'),
              'w', encoding='utf-8') as tsvfile:
        authors = csv.writer(tsvfile, delimiter='\t',
                             lineterminator='\n')
        authors.writerow(['author', 'orcid', 'documentcount', 'citedbycount',
                          'citationcount', 'citationcount', 'pubrangestart',
                          'pubrangeend', 'indexedname', 'surname', 'givenname'])
        for author in tqdm(author_list):
            url = base_url + author
            api_return = requests.get(url,
                                      headers = {"X-ELS-APIKey": apikey,
                                                 "Accept":
                                                 "application/json"})
            api_json = json.loads(api_return.content)
            with open(os.path.join(author_path, 'raw', 'file_' + author + '.json'),
                      'w', encoding='utf-8') as outfile:
                json.dump(api_json, outfile)
            core = api_json['author-retrieval-response'][0]["coredata"]
            try:
                orcid  = core['orcid']
            except (KeyError, TypeError):
                orcid = np.nan
            try:
                documentcount = core['document-count']
            except (KeyError, TypeError):
                documentcount = np.nan
            try:
                citedbycount = core['cited-by-count']
            except (KeyError, TypeError):
                citedbycount = np.nan
            try:
                citationcount = core['citation-count']
            except (KeyError, TypeError):
                citationcount = np.nan
            profile = api_json['author-retrieval-response'][0]["author-profile"]
            try:
                indexedname = profile["preferred-name"]["indexed-name"]
            except (KeyError, TypeError):
                indexedname = np.nan
            try:
                surname = profile["preferred-name"]["surname"]
            except (KeyError, TypeError):
                surname = np.nan
            try:
                givenname = profile["preferred-name"]["given-name"]
            except (KeyError, TypeError):
                givenname = np.nan
            pubrange = api_json['author-retrieval-response'][0]['author-profile']['publication-range']
            try:
                pubrangestart = pubrange['@start']
            except (KeyError, TypeError):
                pubrangestart = np.nan
            try:
                pubrangeend = pubrange['@end']
            except (KeyError, TypeError):
                pubrangeend = np.nan
            authors.writerow([author, orcid, documentcount, citedbycount,
                              citationcount, citationcount, pubrangestart,
                              pubrangeend,indexedname,surname,givenname])


if __name__ == '__main__':
    apikey = load_token(os.path.join(os.getcwd(), '..',
                                     'keys', 'elsevier_apikey'))
    d_path = os.path.abspath(os.path.join('..', 'data'))
    doi_list = search_scopus_into_csv(apikey, d_path)
    scopus_abstract(apikey, d_path, doi_list)
    scopus_plumx(doi_list, d_path, apikey)
    auth_df = pd.read_csv(os.path.join(d_path, 'scopus',
                                       'abstract', 'parsed',
                                       'scopus_aff_auth.tsv'),
                          sep='\t', encoding='utf-8')
    authid = auth_df['authorid'].drop_duplicates()
    authid = authid.dropna().astype(str).str[0:-2].tolist()
    scopus_authors(authid, d_path, apikey)
