# OpenBTK

OpenBTK is a comprehensive library for biomedical data processing and ingestion. It supports text, image, and audio modalities, preparing them for vector database insertion.

## Features

- **Multi-modal Support**: Text (Clinical notes), Image (X-Rays, CT), Audio (Heart sounds).
- **Automated Chunking**: Intelligent splitting of data.
- **Vector Encoding**: Integration with BioBERT, CLIP, etc.
- **Extensible**: Easy to add new processors and encoders.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from openbtk.core.ingester import OpenBTKIngester
# ... usage example
```
