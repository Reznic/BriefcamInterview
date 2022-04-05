# BriefcamInterview
Implementation and evaluation of RANSAC Algorithm for detection of shapes from a noisy data


# Getting Started:

Make sure to install all of the project's dependencies
```
pip3 install -r requirements.txt
```

# Usage:
## Generating test data samples of random shapes
create a configuration file at <config_path> and
run:
```
generator.py <config_path> <output_path> [--debug]
```
Use the optional --debug flag to plot the generated shapes and data.

# Running Tests:
```
python3 -m pytest
```
