# systemdynamics
 Converting causal loop diagrams into computational system dynamics models

## Setup Instructions

### Install Dependencies

Make sure you have `pip` installed. Then, you can install the package using

```sh
pip install systemdynamics
```

## Custom Equations

You can define custom equations for variables by adding an **"Equation"** column to the Elements sheet in your Kumu Excel file.

### Syntax

- **Variables**: Use variable names as they appear in the model
- **Parameters**: Use `#` symbol for parameters to be sampled (always positive in [0, parameter_value_aux])
  - `#` - bare parameter (sequential numbering)
  - `#1`, `#2`, etc. - numbered parameters
- **Functions**: Use `np.FUNCTION` syntax for numpy functions
  - Supported: `exp`, `log`, `sin`, `cos`, `tan`, `tanh`, `sqrt`, `abs`, `sigmoid`, etc.

### Examples

```
# Simple linear combination with sampled coefficients
#1 * A + #2 * B

# Exponential decay
np.exp(-#1 * A) * B

# Bounded S-curve (sigmoid)
#1 * np.tanh(#2 * (A - B))

# Saturation effect
#1 * (1 - np.exp(-#2 * A))
```

### Validation

The system automatically validates equations against incoming links:
- **Warning** if variables in equation are not incoming links
- **Warning** if incoming links are not used in equation

This helps catch typos and ensure the model structure is consistent.

## Additional Information

- If you encounter any issues, please ensure that you have all the necessary dependencies installed.
- For more information, refer to the [documentation]() or [contact us](j.f.uleman@gmail.com).