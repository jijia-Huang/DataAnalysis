## MODIFIED Requirements

### Requirement: Training Parameters Configuration
The system SHALL allow users to configure training parameters when applicable.

#### Scenario: Gradient descent parameters
- **WHEN** a user selects Gradient Descent algorithm
- **THEN** the system displays parameter inputs:
  - Learning rate (default: 0.01)
  - Maximum iterations (default: 1000)
  - Convergence tolerance (default: 1e-6)
- **AND** each parameter has a clear description and reasonable default value
- **AND** the parameters are validated before training

#### Scenario: Standard linear regression
- **WHEN** a user selects standard Linear Regression
- **THEN** the system displays regularization options:
  - Regularization type selection: None (default), L1 (Lasso), L2 (Ridge)
  - Regularization strength (alpha) input: range 0.001 to 100, default 1.0
  - Expandable/collapsible explanation section for L1 and L2 regularization
- **AND** each option has a clear description explaining its purpose
- **AND** when "None" is selected, the model uses standard LinearRegression without regularization
- **AND** when L1 or L2 is selected, the corresponding regularized model is used
- **AND** the regularization strength parameter is validated before training
- **AND** the parameters are displayed only when Linear Regression is selected

#### Scenario: Regularization explanation display
- **WHEN** a user views the regularization options
- **THEN** an expandable/collapsible section is displayed explaining regularization
- **AND** the explanation includes:
  - What L1 (Lasso) regularization does and its characteristics
  - What L2 (Ridge) regularization does and its characteristics
  - The differences between L1 and L2 regularization
  - When to use each type of regularization
- **AND** the explanation section is collapsed by default
- **AND** users can expand or collapse the explanation section as needed
- **AND** the explanation uses simple, non-technical language suitable for non-programmers

