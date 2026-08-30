"""De-identification -- the flagship subsystem.

Cross-modal by design: free-text PHI removal, DICOM tag scrubbing and FHIR
field redaction are the same problem on different surfaces. See ADR-0006.
"""
