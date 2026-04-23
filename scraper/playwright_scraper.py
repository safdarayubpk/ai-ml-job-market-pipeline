import asyncio
import logging
from playwright.async_api import async_playwright, Page

logger = logging.getLogger(__name__)

SOURCES = [
    "https://remoteok.com/remote-ai-jobs",
    "https://remoteok.com/remote-machine-learning-jobs",
]


async def _scrape_page_with_retry(page: Page, url: str, max_retries: int = 3) -> list[dict]:
    """Load a RemoteOK listing page and extract job rows. Retries on failure."""
    for attempt in range(max_retries):
        try:
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await page.wait_for_timeout(2000)  # allow JS to render
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to load {url} after {max_retries} attempts: {e}")
                return []
            wait = 2 ** attempt
            logger.warning(f"Retrying {url} in {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)

    results = []
    job_rows = await page.query_selector_all("tr.job")
    for row in job_rows:
        try:
            title_el = await row.query_selector("h2.title")
            company_el = await row.query_selector("h3.company")
            desc_el = await row.query_selector("div.description")
            location_el = await row.query_selector("div.location")
            salary_el = await row.query_selector("div.salary")
            url_attr = await row.get_attribute("data-url")

            title = await title_el.inner_text() if title_el else ""
            if not title.strip():
                continue

            results.append({
                "title": title.strip(),
                "company": (await company_el.inner_text()).strip() if company_el else "",
                "description": (await desc_el.inner_text()).strip() if desc_el else "",
                "location": (await location_el.inner_text()).strip() if location_el else "",
                "salary": (await salary_el.inner_text()).strip() if salary_el else None,
                "source_url": f"https://remoteok.com{url_attr}" if url_attr else url,
            })
        except Exception as e:
            logger.debug(f"Skipping row due to parse error: {e}")
            continue

    logger.info(f"Scraped {len(results)} jobs from {url}")
    return results


async def scrape_all_sources(max_per_source: int = 100) -> list[dict]:
    """
    Scrape all configured sources concurrently using asyncio.gather.
    Returns deduplicated list of raw job dicts.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        )

        pages = [await context.new_page() for _ in SOURCES]
        results = await asyncio.gather(
            *[_scrape_page_with_retry(page, url) for page, url in zip(pages, SOURCES)],
            return_exceptions=True,
        )
        await browser.close()

    all_jobs: list[dict] = []
    seen_urls: set[str] = set()
    for result in results:
        if isinstance(result, list):
            for job in result[:max_per_source]:
                url = job.get("source_url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_jobs.append(job)

    logger.info(f"Total unique jobs scraped: {len(all_jobs)}")
    return all_jobs
