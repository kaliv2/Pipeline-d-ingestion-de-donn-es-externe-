import pandas as pd
import requests
from bs4 import BeautifulSoup
from difflib import get_close_matches

BASE_URL = "https://maroc.welipro.com/recherche"

# -----------------------
# 🧱 Fonctions utilitaires
# -----------------------
def scrape_companies_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    companies = []
    for card in soup.select('div.card.border-bottom-1.border-bottom-success.rounded-bottom-0'):
        company = {}
        title_tag = card.select_one('h3.card-title a')
        company['name'] = title_tag.get_text(strip=True) if title_tag else None
        legal_tag = card.select_one('span.d-block.font-size-base')
        company['legal_form_and_capital'] = legal_tag.get_text(strip=True) if legal_tag else None
        activity_tag = card.select_one('div.card-body')
        company['activity'] = activity_tag.get_text(strip=True) if activity_tag else None

        # Informations à gauche
        left_list = card.select('div.row > div.col-md-6')[0].select('ul.list-group li.list-group-item')
        for item in left_list:
            label = item.select_one('span.font-weight-semibold')
            value = item.select_one('div.ml-auto')
            if label and value:
                key = label.get_text(strip=True).replace(':', '').lower()
                company[key] = value.get_text(strip=True)

        # Informations à droite
        right_list = card.select('div.row > div.col-md-6')[1].select('ul.list-group li.list-group-item')
        for item in right_list:
            label = item.select_one('span.font-weight-semibold')
            value = item.select_one('div.ml-auto')
            if label and value:
                key = label.get_text(strip=True).replace(':', '').lower()
                company[key] = value.get_text(strip=True)
            if not label and item.get_text(strip=True):
                company['address'] = item.get_text(strip=True)

        companies.append(company)
    return companies


def search_welipro(query, search_type="name"):
    type_mapping = {
        "ice": "ice",
        "rc": "rc",
        "name": ""
    }
    params = {
        'q': query,
        'type': type_mapping.get(search_type, ""),
        'rs': '',
        'cp': '1',
        'cp_max': '2035272260000',
        'et': '',
        'v': ''
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Erreur lors de la recherche pour {query}: {e}")
        return None

    companies = scrape_companies_from_html(response.text)
    if not companies:
        return None

    if search_type == "name":
        names = [c['name'] for c in companies if c.get('name')]
        closest_name = get_close_matches(query, names, n=1, cutoff=0.5)
        if closest_name:
            for company in companies:
                if company.get('name') == closest_name[0]:
                    return company
    return companies[0]


def enrich_clients_dataframe(df):
    enriched_rows = []
    for idx, row in df.iterrows():
        query = None
        search_type = "name"
        if pd.notna(row.get('ice')) and row.get('ice') != '':
            query = str(row['ice'])
            search_type = "ice"
        elif pd.notna(row.get('rc_and_ct')) and row.get('rc_and_ct') != '':
            query = str(row['rc_and_ct'])
            search_type = "rc"
        elif pd.notna(row.get('raison_sociale')) and row.get('raison_sociale') != '':
            query = row['raison_sociale']
            search_type = "name"

        enriched_data = {
            'numero_personne_host': row.get('numero_personne_host'),
            'raison_sociale': row.get('raison_sociale'),
            'sigle': row.get('sigle'),
            'ice': row.get('ice'),
            'rc_and_ct': row.get('rc_and_ct')
        }

        if query:
            company_info = search_welipro(query, search_type=search_type)
            if company_info:
                enriched_data.update({
                    'legal_form_and_capital': company_info.get('legal_form_and_capital'),
                    'activity': company_info.get('activity'),
                    'identifiant_fiscal': company_info.get('identifiant fiscal'),
                    'creation_date': company_info.get('date de création'),
                    'status': company_info.get('état'),
                    'address': company_info.get('address'),
                    'ice': company_info.get('ice', enriched_data['ice']),
                    'rc_and_ct': company_info.get('registre du commerce', enriched_data['rc_and_ct'])
                })
        enriched_rows.append(enriched_data)
    return pd.DataFrame(enriched_rows)

