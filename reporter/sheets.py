import os
import logging
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"], scopes=_SCOPES
    )
    return gspread.authorize(creds)


def push_report(
    llm_report: str,
    skill_frequencies: dict[str, float],
    cluster_breakdown: dict[int, dict],
    cluster_names: dict[int, str],
    seniority_counts: dict[str, int],
) -> str:
    """
    Push pipeline results to Google Sheets.
    Creates a new tab named by today's date (e.g., "2026-04-23").
    If the tab already exists, it is cleared and overwritten.
    Returns the spreadsheet URL, or empty string on failure.
    """
    try:
        client = _get_client()
        spreadsheet = client.open_by_key(os.environ["GOOGLE_SHEETS_ID"])
        tab_name = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=200, cols=10)
        except gspread.exceptions.APIError:
            worksheet = spreadsheet.worksheet(tab_name)
            worksheet.clear()

        rows: list[list] = []

        rows.append(["AI/ML Job Market Intelligence Report"])
        rows.append([llm_report])
        rows.append([])

        rows.append(["Top Skills", "% of Jobs"])
        for skill, pct in list(skill_frequencies.items())[:20]:
            rows.append([skill, pct])
        rows.append([])

        rows.append(["Cluster", "Job Count", "% of Total"])
        for cluster_id, stats in cluster_breakdown.items():
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            rows.append([name, stats["count"], stats["pct"]])
        rows.append([])

        rows.append(["Seniority", "Count"])
        for label, count in seniority_counts.items():
            rows.append([label, count])

        worksheet.update(rows, "A1")
        logger.info(f"Pushed report to Google Sheets tab: {tab_name}")
        return spreadsheet.url

    except Exception as e:
        logger.error(f"Google Sheets push failed: {e}")
        return ""
