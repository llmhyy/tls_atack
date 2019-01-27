# Extraction of TCP/TLS features

The module extracts TCP and TLS/SSL features from network traffic in PCAP files and stores them in a file.

More information about the features extracted can be found in `feature_extraction.docx`. Logging output can be found in `output.log`

## Getting Started

With a Python 3 environment, install the dependencies from `requirements.txt`

```
pip install -r requirements.txt
```

Execute the following command 

```
python main.py
```

Note:
* The directory where the pcap files are stored is fixed
* The directory to save the csv file containing features is fixed
