import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_web_subpages(starting_url):
    """
    Crawls the web starting from the given URL and returns a list of subpages.
    Will ignore "#" markdown links.

    Args:
        starting_url (str): The URL to start crawling from.

    Returns:
        list: A list of subpages found during the crawling process.
    """
    subpages = []
    visited = set()

    if not starting_url.endswith('/'):
        starting_url += '/'

    def crawl(url):
        visited.add(url)

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')

            for link in links:
                href = link.get('href')
                absolute_url = urljoin(url, href)
                if absolute_url in visited or starting_url not in absolute_url or '#' in absolute_url:
                    continue
                subpages.append(absolute_url)
                crawl(absolute_url)
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    crawl(starting_url)
    return subpages


if __name__ == '__main__':
    url = "https://azure.github.io/shrike/"
    subpages = get_web_subpages(url)
    print(subpages)