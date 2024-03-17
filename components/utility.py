import requests
from bs4 import BeautifulSoup


def scrape_links_for_text(urls, encoding="utf-8"):
    texts = []

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator="\n", strip=True)
            texts.append(text)
            print(f" Done scrape {url}")
        except Exception as e:
            print(f"Failed to scrape {url}: {str(e)}")

    return texts
