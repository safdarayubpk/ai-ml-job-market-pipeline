import asyncio
import logging
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

REMOTEOK_API = "https://remoteok.com/api"
TAGS = ["ai", "machine-learning", "llm", "nlp", "computer-vision"]
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


async def _fetch_tag(session: aiohttp.ClientSession, tag: str) -> list[dict]:
    url = f"{REMOTEOK_API}?tag={tag}"
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning(f"HTTP {resp.status} for tag={tag}")
                return []
            data = await resp.json(content_type=None)
    except Exception as e:
        logger.error(f"Failed to fetch tag={tag}: {e}")
        return []

    jobs = []
    for item in data:
        if not item.get("id"):
            continue
        # Strip HTML from description
        raw_desc = item.get("description", "") or ""
        soup = BeautifulSoup(raw_desc, "html.parser")
        description = soup.get_text(separator=" ", strip=True)[:2000]

        salary = None
        sal_min = item.get("salary_min") or 0
        sal_max = item.get("salary_max") or 0
        if sal_min or sal_max:
            salary = f"${sal_min:,}–${sal_max:,}"

        jobs.append({
            "title": (item.get("position") or "").strip(),
            "company": (item.get("company") or "").strip(),
            "description": description,
            "location": (item.get("location") or "Remote").strip(),
            "salary": salary,
            "source_url": item.get("url") or item.get("apply_url") or "",
        })

    logger.info(f"Fetched {len(jobs)} jobs for tag={tag}")
    return jobs


async def scrape_all_sources(max_per_source: int = 100) -> list[dict]:
    """
    Fetch AI/ML job listings from RemoteOK JSON API across multiple tags.
    Returns deduplicated list of raw job dicts.
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_fetch_tag(session, tag) for tag in TAGS],
            return_exceptions=True,
        )

    all_jobs: list[dict] = []
    seen_urls: set[str] = set()
    for result in results:
        if not isinstance(result, list):
            continue
        for job in result[:max_per_source]:
            url = job.get("source_url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_jobs.append(job)

    logger.info(f"Total unique jobs scraped: {len(all_jobs)}")
    return all_jobs
