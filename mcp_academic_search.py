"""
MCP Server: Academic Literature Search
Uses OpenAlex API (free, no key) + Semantic Scholar API (free, no key)
Specialized for groundwater / hydrology / remote sensing papers
"""

import httpx
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("academic-search")

OPENALEX_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"User-Agent": "rao-research-thesis/1.0 (zaheersapar005@gmail.com)"}


def _clean(text: str | None, max_len: int = 300) -> str:
    if not text:
        return "N/A"
    text = text.strip()
    return text[:max_len] + "..." if len(text) > max_len else text


@mcp.tool()
def search_papers(
    query: str,
    year_from: int = 2010,
    year_to: int = 2025,
    limit: int = 10,
) -> str:
    """
    Search academic papers via OpenAlex.
    Returns title, authors, year, journal, DOI, abstract snippet, open-access URL.
    Good for finding groundwater/hydrology papers about any region.
    """
    params = {
        "search": query,
        "filter": f"publication_year:{year_from}-{year_to}",
        "per-page": min(limit, 25),
        "select": "title,authorships,publication_year,primary_location,doi,abstract_inverted_index,open_access,cited_by_count",
        "sort": "cited_by_count:desc",
    }
    try:
        r = httpx.get(f"{OPENALEX_BASE}/works", params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"OpenAlex request failed: {e}"

    results = data.get("results", [])
    if not results:
        return "No papers found. Try broader keywords."

    lines = [f"Found {data['meta']['count']} total results (showing top {len(results)}):\n"]
    for i, w in enumerate(results, 1):
        title = w.get("title") or "No title"
        year = w.get("publication_year", "?")
        doi = w.get("doi") or "No DOI"
        cited = w.get("cited_by_count", 0)

        authors = w.get("authorships", [])
        author_names = [a["author"]["display_name"] for a in authors[:3] if a.get("author")]
        author_str = ", ".join(author_names) + (" et al." if len(authors) > 3 else "")

        journal = "Unknown journal"
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        if src.get("display_name"):
            journal = src["display_name"]

        oa = w.get("open_access", {})
        pdf_url = oa.get("oa_url") or "Not freely available"

        # Reconstruct abstract from inverted index
        abstract = "No abstract"
        inv = w.get("abstract_inverted_index")
        if inv:
            pos_word = {}
            for word, positions in inv.items():
                for pos in positions:
                    pos_word[pos] = word
            if pos_word:
                reconstructed = " ".join(pos_word[k] for k in sorted(pos_word))
                abstract = _clean(reconstructed, 250)

        lines.append(
            f"{i}. {title} ({year})\n"
            f"   Authors: {author_str}\n"
            f"   Journal: {journal} | Cited: {cited}\n"
            f"   DOI: {doi}\n"
            f"   PDF: {pdf_url}\n"
            f"   Abstract: {abstract}\n"
        )

    return "\n".join(lines)


@mcp.tool()
def search_pothohar_groundwater(data_type: str = "all") -> str:
    """
    Pre-built search for Pothohar Plateau groundwater studies.
    data_type options: 'borehole', 'well', 'potential', 'validation', 'all'
    Returns papers most likely to contain real field data for validation.
    """
    queries = {
        "borehole": "borehole groundwater Pothohar Punjab Pakistan",
        "well":     "tubewell water table depth Rawalpindi Chakwal Attock Jhelum Pakistan",
        "potential": "groundwater potential mapping Pothohar Punjab GIS remote sensing",
        "validation": "groundwater potential validation field data Punjab Pakistan",
        "all":      "groundwater Pothohar Plateau Punjab Pakistan",
    }
    q = queries.get(data_type, queries["all"])
    return search_papers(q, year_from=2000, year_to=2025, limit=12)


@mcp.tool()
def search_pakistan_gw_data_sources(district: str = "Pothohar") -> str:
    """
    Search for institutional data sources and reports (PCRWR, WAPDA, NESPAK)
    that may contain borehole or water table data for the given district/region.
    """
    queries = [
        f"PCRWR groundwater {district} Pakistan report",
        f"WAPDA groundwater survey Punjab Pakistan water table",
        f"NESPAK hydrogeology {district} Pakistan",
    ]
    all_results = []
    for q in queries:
        all_results.append(f"\n--- Query: {q} ---")
        all_results.append(search_papers(q, year_from=1990, year_to=2025, limit=5))
    return "\n".join(all_results)


@mcp.tool()
def get_paper_details(doi_or_title: str) -> str:
    """
    Get full details for a specific paper using Semantic Scholar.
    Pass either a DOI (e.g. '10.1234/xyz') or a paper title.
    Returns abstract, citations, references, and PDF link if available.
    """
    # Try DOI first
    if doi_or_title.startswith("10."):
        url = f"{S2_BASE}/paper/DOI:{doi_or_title}"
    else:
        # Search by title
        r = httpx.get(
            f"{S2_BASE}/paper/search",
            params={"query": doi_or_title, "limit": 1, "fields": "paperId"},
            headers=HEADERS,
            timeout=15,
        )
        try:
            r.raise_for_status()
            items = r.json().get("data", [])
            if not items:
                return "Paper not found on Semantic Scholar."
            url = f"{S2_BASE}/paper/{items[0]['paperId']}"
        except Exception as e:
            return f"Search failed: {e}"

    fields = "title,authors,year,abstract,externalIds,openAccessPdf,citationCount,references,venue"
    try:
        r = httpx.get(url, params={"fields": fields}, headers=HEADERS, timeout=15)
        r.raise_for_status()
        p = r.json()
    except Exception as e:
        return f"Details fetch failed: {e}"

    authors = ", ".join(a["name"] for a in p.get("authors", [])[:5])
    pdf = p.get("openAccessPdf") or {}
    refs = p.get("references", [])[:5]
    ref_titles = [r.get("title", "Unknown") for r in refs]

    return (
        f"Title: {p.get('title', 'N/A')}\n"
        f"Authors: {authors}\n"
        f"Year: {p.get('year', '?')} | Venue: {p.get('venue', 'N/A')}\n"
        f"Citations: {p.get('citationCount', 0)}\n"
        f"DOI: {p.get('externalIds', {}).get('DOI', 'N/A')}\n"
        f"PDF: {pdf.get('url', 'Not available')}\n\n"
        f"Abstract:\n{_clean(p.get('abstract', 'No abstract'), 600)}\n\n"
        f"Key References (first 5):\n" + "\n".join(f"  - {t}" for t in ref_titles)
    )


@mcp.tool()
def find_open_access_pdf(doi: str) -> str:
    """
    Check Unpaywall for a free PDF of a paper by DOI.
    Returns direct PDF download URL if available.
    """
    try:
        r = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "zaheersapar005@gmail.com"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"Unpaywall lookup failed: {e}"

    oa_locs = data.get("oa_locations", [])
    pdfs = [loc.get("url_for_pdf") for loc in oa_locs if loc.get("url_for_pdf")]

    if pdfs:
        return f"Free PDF available:\n" + "\n".join(pdfs)
    elif data.get("is_oa"):
        return f"Open access but no direct PDF link found. Try: {data.get('best_oa_location', {}).get('url', 'N/A')}"
    else:
        return "No free version found. Paper is paywalled."


if __name__ == "__main__":
    mcp.run(transport="stdio")
