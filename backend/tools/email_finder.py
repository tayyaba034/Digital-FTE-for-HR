"""
tools/email_finder.py
HR/recruiter email lookup.
Strategy: Hunter.io → pattern guessing → Claude inference → manual fallback.
"""
import os
import re
from typing import Optional
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

# Common HR email patterns
HR_PATTERNS = [
    "hr@{domain}",
    "careers@{domain}",
    "recruiting@{domain}",
    "talent@{domain}",
    "jobs@{domain}",
    "recruitment@{domain}",
    "apply@{domain}",
    "people@{domain}",
    "hiring@{domain}",
]

# Known company email domains (expand as needed)
KNOWN_DOMAINS: dict[str, str] = {
    "google": "google.com",
    "alphabet": "google.com",
    "meta": "meta.com",
    "facebook": "meta.com",
    "amazon": "amazon.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "netflix": "netflix.com",
    "stripe": "stripe.com",
    "openai": "openai.com",
    "anthropic": "anthropic.com",
    "deepmind": "deepmind.com",
    "spotify": "spotify.com",
    "airbnb": "airbnb.com",
    "uber": "uber.com",
    "lyft": "lyft.com",
    "github": "github.com",
    "gitlab": "gitlab.com",
    "atlassian": "atlassian.com",
    "slack": "slack.com",
    "salesforce": "salesforce.com",
    "adobe": "adobe.com",
}


def _company_to_domain(company: str) -> Optional[str]:
    """Try to infer email domain from company name."""
    clean = company.lower().strip()
    
    # Check known domains first
    for key, domain in KNOWN_DOMAINS.items():
        if key in clean:
            return domain
    
    # Remove common suffixes
    for suffix in [" inc", " ltd", " llc", " corp", " limited", " plc", " ag", " gmbh", ".com"]:
        clean = clean.replace(suffix, "")
    
    # Remove special chars, collapse spaces
    clean = re.sub(r"[^a-z0-9]", "", clean.strip())
    
    if len(clean) >= 3:
        return f"{clean}.com"
    return None


class EmailFinder:
    def __init__(self):
        self.hunter_key = os.getenv("HUNTER_IO_API_KEY", "")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _hunter_domain_search(self, domain: str) -> Optional[str]:
        """Search Hunter.io for HR emails at a domain."""
        if not self.hunter_key:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.hunter.io/v2/domain-search",
                    params={
                        "domain": domain,
                        "api_key": self.hunter_key,
                        "department": "human_resources",
                        "limit": 3,
                    }
                )
                data = resp.json()
                emails = data.get("data", {}).get("emails", [])
                if emails:
                    # Prefer HR/recruiting emails
                    for email in emails:
                        addr = email.get("value", "")
                        if any(p in addr for p in ["hr", "recruit", "talent", "career", "hiring", "people"]):
                            log.info("email_finder.hunter_found", email=addr, domain=domain)
                            return addr
                    return emails[0].get("value")
        except Exception as e:
            log.warning("email_finder.hunter_error", error=str(e))
        return None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _hunter_email_finder(self, domain: str, first_name: str = "hr", last_name: str = "team") -> Optional[str]:
        """Use Hunter.io email finder to verify a guessed pattern."""
        if not self.hunter_key:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.hunter.io/v2/email-finder",
                    params={
                        "domain": domain,
                        "first_name": first_name,
                        "last_name": last_name,
                        "api_key": self.hunter_key,
                    }
                )
                data = resp.json()
                email = data.get("data", {}).get("email")
                score = data.get("data", {}).get("score", 0)
                if email and score > 50:
                    return email
        except Exception as e:
            log.warning("email_finder.hunter_finder_error", error=str(e))
        return None

    def _guess_emails(self, domain: str) -> list[str]:
        """Generate a list of likely HR email guesses for a domain."""
        return [pattern.format(domain=domain) for pattern in HR_PATTERNS]

    async def find_hr_email(self, company: str, job_url: str = "") -> dict:
        """
        Find HR/recruiting email for a company.
        Returns: {email, confidence, source, alternatives}
        """
        domain = _company_to_domain(company)

        if not domain:
            log.warning("email_finder.no_domain", company=company)
            return {
                "email": None,
                "confidence": 0.0,
                "source": "none",
                "alternatives": [],
                "note": f"Could not determine domain for '{company}'. Please enter HR email manually.",
            }

        # 1. Hunter.io domain search (highest confidence)
        if self.hunter_key:
            email = await self._hunter_domain_search(domain)
            if email:
                return {
                    "email": email,
                    "confidence": 0.90,
                    "source": "hunter_domain_search",
                    "alternatives": self._guess_emails(domain)[:3],
                }

        # 2. Pattern guesses (medium confidence)
        guesses = self._guess_emails(domain)
        best_guess = guesses[0]  # "hr@domain.com" is most common

        log.info("email_finder.pattern_guess", company=company, email=best_guess, domain=domain)
        return {
            "email": best_guess,
            "confidence": 0.45,
            "source": "pattern_guess",
            "alternatives": guesses[1:4],
            "note": f"Email guessed based on common patterns for {domain}. Please verify before sending.",
        }

    async def find_emails_batch(self, companies: list[dict]) -> list[dict]:
        """Find emails for multiple companies. Each dict: {id, company, job_url}"""
        results = []
        for company_info in companies:
            result = await self.find_hr_email(
                company=company_info.get("company", ""),
                job_url=company_info.get("job_url", ""),
            )
            results.append({"job_id": company_info.get("id"), **result})
        return results


# Singleton
email_finder = EmailFinder()
