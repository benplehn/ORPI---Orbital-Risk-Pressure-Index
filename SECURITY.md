# Security Policy

## Supported versions
Current main branch only.

## Reporting a vulnerability
If you find a security issue, do not open a public issue with exploit details.
Report privately to the maintainer.

## Secret handling requirements
- Never commit `.env` files.
- Never store real credentials in examples or tests.
- Rotate credentials immediately if exposure is suspected.

## Automated scanning
This repository uses GitHub Actions with gitleaks for secret detection on push and pull request.

## Local hardening checklist
1. Keep `.env` local only.
2. Rotate Space-Track credentials before making repository public.
3. Review git history for accidental credential leakage before first public push.
