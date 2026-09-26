# Security Policy

The **Mechanistic LLM Router** project takes security vulnerabilities seriously. We appreciate the efforts of security researchers and practitioners to help keep our open-source software and deployments secure.

---

## Supported Versions

Only the latest active minor release receives active security patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability in this project, **please do NOT open a public GitHub issue.**

Instead, please report the vulnerability privately via one of the following methods:

1. **GitHub Security Advisory**:
   Navigate to the [Security Advisories tab](https://github.com/vfcarida/mechanistic-llm-router/security/advisories) on GitHub and click "Report a vulnerability".

2. **Direct Email**:
   Email the project maintainer at `vfcarida@gmail.com` with the subject prefix `[SECURITY VULNERABILITY] mechanistic-llm-router`.

### What to Include in Your Report

To help us investigate and triage your report quickly, please include:
- A clear description of the vulnerability.
- Steps to reproduce or proof-of-concept code.
- Affected versions, configurations, or endpoints.
- Any potential remediation or patch suggestions if available.

### Response SLA

- **Initial Acknowledgement**: Within 48 hours of receipt.
- **Triage & Assessment**: Within 7 business days.
- **Fix & Disclosure**: We aim to release a patch and coordinated security advisory within 30 days of confirmed vulnerability validation.

---

## Security Best Practices for Deployments

When running the gateway server in production environments:
1. **Never use default or empty API keys**: Always set `ROUTER_API_KEY` to a cryptographically strong secret.
2. **TLS Hardening**: Place the gateway behind a reverse proxy (e.g., Traefik, NGINX, or Kong) enforcing TLS 1.3.
3. **Observability Scrapes**: Keep the Prometheus metrics port (default `9090`) restricted to internal monitoring networks and not exposed publicly.
4. **Input Length Limits**: Keep `ROUTER_MAX_PROMPT_CHARS` configured (default `10,000` chars) to protect against memory exhaustion or denial-of-service payloads.
