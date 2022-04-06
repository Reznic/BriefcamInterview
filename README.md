# BriefcamInterview
Implementation and evaluation of RANSAC Algorithm for detection of shapes from a noisy data


# Getting Started:

Make sure to install all of the project's dependencies
```
pip3 install -r requirements.txt
```

# Usage:
## Generating test data samples of random shapes
create a configuration file at <config_path> 
(See configurations_example.json file for example)

and run:
```
generator.py <config_path> <output_path> [--debug]
```
Use the optional --debug flag to plot the generated shapes and data.

## Estimating shapes from test data, with RANSAC algorithm
pass the output_path from the generator, as input_path for the estimator:
```
estimator.py <input_path> <output_path> [--debug]
```
Use the optional --debug flag to plot the estimated shapes and data.

# Running Tests:
```
python3 -m pytest
```
