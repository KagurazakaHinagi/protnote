"""Utility functions for network operations."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def fetch_with_retries(url, method="GET", **kwargs):
    """Fetch URL content with automatic retries and error handling."""

    retry_strategy = Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )

    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))

        # Always set a timeout!
        kwargs.setdefault("timeout", (5, 30))  # connect, read

        response = session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
