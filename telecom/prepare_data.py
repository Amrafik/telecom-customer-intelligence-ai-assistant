"""Download a small CC BY 4.0 public dataset; no synthetic fallback is silent."""
import hashlib
import io
import json
import zipfile
from urllib.request import urlopen

from telecom.config import ROOT

URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"


def main():
    destination = ROOT / "data/raw/customer_churn.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urlopen(URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        destination.write_bytes(archive.read(name))
    metadata = {
        "name": "UCI Iranian Churn", "synthetic": False,
        "citation": "Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z",
        "source": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        "download": URL, "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "notes": "Public anonymised training data; fictional app customers are authored separately. No Vodafone data or affiliation."
    }
    (ROOT / "data/provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Dataset ready: {destination}; SHA256 {metadata['sha256']}")


if __name__ == "__main__":
    main()
