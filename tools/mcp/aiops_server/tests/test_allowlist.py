"""Probe-target allowlist (AIOPS_ALLOWED_TARGETS) behavior."""

import pytest
from aiops.http_client import ALLOWED_TARGETS_ENV, build_client, check_target_allowed

ALLOWLIST = "10.0.0.0/8, fd00::/8, model.ns.svc, .svc.cluster.local"


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.2.34:8000/v1",
        "http://[fd00::1]:8000/v1",
        "http://model.ns.svc:8000/v1",
        "https://MODEL.NS.SVC/v1",
        "http://llm.models.svc.cluster.local/v1",
    ],
)
def test_allowed_targets(url):
    check_target_allowed(url, ALLOWLIST)


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.1/v1",
        "http://evil.example.com/v1",
        "http://svc.cluster.local.evil.example/v1",
        "http://other.ns.svc/v1",
    ],
)
def test_disallowed_targets(url):
    with pytest.raises(ValueError, match="not allowed"):
        check_target_allowed(url, ALLOWLIST)


def test_unset_allowlist_allows_any_http_target(monkeypatch):
    monkeypatch.delenv(ALLOWED_TARGETS_ENV, raising=False)
    check_target_allowed("http://169.254.169.254/v1")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/v1", "not a url"])
def test_non_http_urls_are_rejected(url):
    with pytest.raises(ValueError, match="http"):
        check_target_allowed(url, "")


def test_build_client_enforces_env_allowlist(monkeypatch):
    monkeypatch.setenv(ALLOWED_TARGETS_ENV, "10.0.0.0/8")
    with pytest.raises(ValueError, match="not allowed"):
        build_client("http://192.168.1.1:8000/v1")
    build_client("http://10.1.2.3:8000/v1").close()
