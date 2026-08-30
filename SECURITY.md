# Security Policy

## Reporting a Vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/openbtk/openbtk-core/security/advisories/new)
on this repository. **Do not open a public issue.**

Target response: acknowledgement within 48 hours, coordinated disclosure within
90 days.

## What We Treat as Critical

OpenBTK processes Protected Health Information. Any path by which PHI can reach
somewhere it should not is handled as critical, specifically:

- PHI written to logs, a run manifest, a `DeidReport`, a cache, or an exception
  message.
- PHI transmitted to a third-party service under the default configuration.
- A de-identification failure that causes an identifier category to be missed
  systematically.
- Credentials written to a configuration file, a log, or a run manifest.

## Supported Versions

Pre-1.0. Only the latest release receives fixes.

## Scope

OpenBTK is a software toolkit, not a medical device and not a clinical
decision-support system. It provides technical controls that support a
compliance programme; it does not by itself make any system HIPAA- or
EU AI Act-compliant. Network, access and physical security of the deployment
environment are the operator's responsibility.
