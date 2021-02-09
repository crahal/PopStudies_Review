import os
import math
import json
import re
import requests
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm


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
