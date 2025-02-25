# Python Open-Ephys

This repository provides a set of tools using the Open-Ephys data acquisition system and GUI for electromyography (EMG) data.

<!-- ![intan_logo.png](/assets/intan_logo.png) -->


Code was written and tested using Windows 11, Python 3.10.
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Installation

1. Create a virtual environment using [Anaconda](https://www.anaconda.com/products/distribution) or Python's virtualenv
   - Using Anaconda:
      ~~~
      conda create -n ephys
      conda activate ephys
      ~~~
   - Using Python's virtualenv:
     ~~~
     python3 -m venv .ephys
     source .ephys/bin/activate # Linux
     call .ephys/Scripts/activate # Windows
     ~~~
2. Clone the repository and navigate to the project directory
   ~~~
   git clone https://github.com/Neuro-Mechatronics-Interfaces/Python_Open-Ephys.git
   cd Python_Open-Ephys
   ~~~
3. Install dependencies
    ~~~
    pip install -r requirements.txt
    ~~~
4. Setup Open-Ephys GUI 
    - Install from the [Open-Ephys website](https://open-ephys.org/gui) and select your system 
    - Install the [ZQM Plugin](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/ZMQ-Interface.html) for streaming data

## Usage

The OpenEphysClient class can be easily imported into your current project. The class provides a simple interface to connect to the Open-Ephys GUI and stream data.

```python
from open_ephys import OpenEphysClient
client = OpenEphysClient()
samples = client.get_samples(channel=8)
```
Check the directory for other demo example scripts
