"""Public publisher requests with a bounded, TLS-verified DNS fallback."""
from __future__ import annotations

import ipaddress
import logging
import time
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)
# The local resolver can return unrelated addresses for FT. Keep this scoped
# to the publisher we verified, without changing the operating system's DNS.
DNS_FALLBACK_HOSTS = frozenset({'www.ft.com'})
DNS_RESOLVER = 'https://cloudflare-dns.com/dns-query'
_dns_cache: dict[str, tuple[float, tuple[str, ...]]] = {}


async def _public_addresses(client: httpx.AsyncClient, hostname: str) -> tuple[str, ...]:
    cached = _dns_cache.get(hostname)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    response = await client.get(
        DNS_RESOLVER, params={'name': hostname, 'type': 'A'},
        headers={'Accept': 'application/dns-json'}, follow_redirects=False,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('Status') != 0:
        raise RuntimeError(f'Public DNS lookup failed for {hostname}')
    answers = [row for row in payload.get('Answer', []) if row.get('type') == 1]
    addresses = []
    for row in answers:
        address = ipaddress.ip_address(row['data'])
        if address.version != 4 or not address.is_global:
            raise RuntimeError(f'Public DNS returned a non-public address for {hostname}')
        addresses.append(str(address))
    if not addresses:
        raise RuntimeError(f'Public DNS returned no addresses for {hostname}')
    ttl = max(0, min(300, *(int(row.get('TTL', 0)) for row in answers)))
    result = tuple(dict.fromkeys(addresses))[:2]
    _dns_cache[hostname] = (time.monotonic() + ttl, result)
    return result


async def _resolved_get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    logical_url = httpx.URL(url)
    last_error: httpx.HTTPError | None = None
    for address in await _public_addresses(client, logical_url.host):
        try:
            # HTTP Host, TLS SNI and certificate verification retain the
            # publisher's hostname. Only the TCP destination uses the address.
            return await client.get(
                logical_url.copy_with(host=address),
                headers={'Host': logical_url.host},
                extensions={'sni_hostname': logical_url.host},
                follow_redirects=False,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


async def get_public_page(client: httpx.AsyncClient, url: str) -> httpx.Response:
    hostname = httpx.URL(url).host
    if hostname not in DNS_FALLBACK_HOSTS or httpx.URL(url).scheme != 'https':
        return await client.get(url, follow_redirects=True)
    cached = _dns_cache.get(hostname)
    if not cached or cached[0] <= time.monotonic():
        try:
            return await client.get(url, follow_redirects=True)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            logger.info('Publisher connection failed; resolving %s through HTTPS DNS', hostname)
    current = url
    for _ in range(6):
        current_host = httpx.URL(current).host
        if current_host not in DNS_FALLBACK_HOSTS or httpx.URL(current).scheme != 'https':
            return await client.get(current, follow_redirects=True)
        response = await _resolved_get(client, current)
        if not response.is_redirect or 'location' not in response.headers:
            return response
        current = urljoin(current, response.headers['location'])
    raise httpx.TooManyRedirects('Publisher exceeded DNS-fallback redirect limit')
