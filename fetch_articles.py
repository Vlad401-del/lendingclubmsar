import urllib.request
import urllib.parse
import json

queries = [
    'Explainable AI credit risk',
    'enterprise architecture fintech',
    'markov switching credit risk',
    'financial inclusion machine learning'
]

results = []
for q in queries:
    url = f'https://api.openalex.org/works?search={urllib.parse.quote(q)}&sort=cited_by_count:desc&per-page=5'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            for w in data.get('results', []):
                title = w.get('title')
                year = w.get('publication_year')
                citations = w.get('cited_by_count')
                doi = w.get('doi')
                host = w.get('primary_location', {})
                source = host.get('source', {}) if host else {}
                journal = source.get('display_name') if source else 'N/A'
                if journal and title:
                    results.append({'Title': title, 'Journal': journal, 'Year': year, 'Citations': citations, 'DOI': doi})
    except Exception as e:
        print(f"Error on {q}: {e}")

# Unique by title
seen = set()
unique_results = []
for r in results:
    if r['Title'] not in seen:
        seen.add(r['Title'])
        unique_results.append(r)

# Format as markdown
md = "## Referensi Tambahan (Jurnal Q1 / Scopus)\n\n"
md += "| No | Judul Artikel | Jurnal | Tahun | Citations | DOI |\n"
md += "|---|---|---|---|---|---|\n"
for i, r in enumerate(unique_results[:15], 1):
    doi_link = r['DOI'] if r['DOI'] else '-'
    md += f"| {i} | {r['Title']} | {r['Journal']} | {r['Year']} | {r['Citations']} | {doi_link} |\n"

with open('d:/laragon/www/lendingclubmsar/articles_list.md', 'w', encoding='utf-8') as f:
    f.write(md)

print(f"Saved {len(unique_results[:15])} articles to articles_list.md")
