## FIBI Post Process Manual Colorization Software (FPPMCS)

Requirements can be found in the requirements.txt file. Create a python virtual environment from that file,
```bash
# 1. Create a clean virtual environment named 'venv'
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Upgrade pip and install the required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Execute
python3 src/main.py
```

#### TODO:
- Minimum and Max RGB Channels
- 2D FFT (Fast Fourier Transform)

#### Completed:
- High resolution preview *(High Priority)* (in progress)
- Histogram log scale
- Hue slider
- Faster saves through threading *(Medium Priority)*
- ~~Histogram log scale *(High Priority)*~~(completed)