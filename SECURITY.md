# Security

Reelsift runs on your own machine. Its review app listens only on 127.0.0.1, refuses requests whose Host or Origin header isn't the app itself, and serves media only for files the trip knows about. Your footage is never uploaded. The network is used to download the models when you ask for them; when they load, the model libraries may also check online for newer model files (set `HF_HUB_OFFLINE=1` to prevent that). Everything works offline.

## Reporting a vulnerability

Please report security problems privately through GitHub: on the repository's **Security** tab, choose **Report a vulnerability**. Don't open a public issue for them.

Include what an attacker could do, the steps to reproduce it, and the version (`reelsift --version`). The aim is to reply within a week, and to fix confirmed problems or explain the plan within a month.

Only the latest release is supported.
