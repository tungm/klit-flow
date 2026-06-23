# Corporate CA certificates

Drop your company's **root CA** (and any intermediate CAs) here as PEM-encoded
files ending in `.crt`, one certificate per file. They are copied into the Docker
image's system trust store at build time (`update-ca-certificates`), so `pip`,
the PyTorch index download, the tree-sitter parser download, and the HuggingFace
model download all succeed behind a TLS-inspecting proxy.

- Files **must** be PEM (`-----BEGIN CERTIFICATE-----`) and end in `.crt`.
  A DER `.cer` must be converted: `openssl x509 -inform der -in corp.cer -out corp-ca.crt`.
- Use one file per certificate; name them anything (e.g. `corp-root-ca.crt`).
- This folder is safe to commit — CA **public** certificates are not secrets.
  Do **not** put private keys here.

With only this `README.md` and `.gitkeep` present, the image builds normally
(the certificate step is a no-op), so nothing here is required outside a
corporate network.
