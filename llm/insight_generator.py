import os
import time
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

_PROMPT_TEMPLATE = """\
You are a market intelligence analyst specializing in AI/ML talent trends. \
Based on the structured job market data below, write a concise 2-3 paragraph briefing \
for a technical audience. Focus on the most notable trends, dominant skills, and what \
the cluster distribution reveals about where AI hiring is headed. \
Write in an analytical, confident tone. Flowing paragraphs only - no bullet points.

Market Data
-----------
Analysis date: {date_range}
Total jobs analyzed: {total_jobs}
Top 10 skills by mention rate: {top_skills}
Job type clusters: {cluster_breakdown}
Seniority distribution: {seniority_breakdown}
"""


def build_prompt(
    total_jobs: int,
    top_skills: dict[str, float],
    cluster_breakdown: dict[int, dict],
    cluster_names: dict[int, str],
    seniority_counts: dict[str, int],
    date_range: str,
) -> str:
    """Build the structured prompt string from pipeline output data."""
    top_skills_str = ", ".join(
        f"{skill} ({pct}%)" for skill, pct in list(top_skills.items())[:10]
    )
    cluster_str = "; ".join(
        f"{cluster_names.get(k, f'Cluster {k}')} ({v['pct']}%)"
        for k, v in cluster_breakdown.items()
    )
    total_seniority = sum(seniority_counts.values()) or 1
    seniority_str = ", ".join(
        f"{label} {round(count / total_seniority * 100, 1)}%"
        for label, count in seniority_counts.items()
    )
    return _PROMPT_TEMPLATE.format(
        date_range=date_range,
        total_jobs=total_jobs,
        top_skills=top_skills_str,
        cluster_breakdown=cluster_str,
        seniority_breakdown=seniority_str,
    )


def generate_insight(prompt: str) -> str:
    """
    Call Groq API with the given prompt and return the generated insight text.
    Retries once on rate limit errors with a 10-second delay.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == 0:
                time.sleep(10)
            else:
                raise
