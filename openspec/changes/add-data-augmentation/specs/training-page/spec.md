## ADDED Requirements
### Requirement: Data Augmentation Configuration
The system SHALL allow users to configure data augmentation settings before training.

#### Scenario: Enable data augmentation
- **WHEN** a user is ready to train a model
- **THEN** the system displays a checkbox to enable/disable data augmentation
- **AND** when enabled, the system displays augmentation parameter controls
- **AND** the default state is disabled

#### Scenario: Noise type selection
- **WHEN** data augmentation is enabled
- **THEN** the system displays radio buttons or dropdown for noise type selection:
  - Gaussian noise (default)
  - Uniform noise
- **AND** each noise type has a brief description explaining its characteristics

#### Scenario: Noise strength configuration
- **WHEN** data augmentation is enabled
- **THEN** the system displays a slider or number input for noise strength
- **AND** the noise strength range is 0.01 to 0.5 (default: 0.1)
- **AND** the noise strength represents a multiplier of the feature's standard deviation
- **AND** a description explains that higher values add more noise

#### Scenario: Augmentation multiplier configuration
- **WHEN** data augmentation is enabled
- **THEN** the system displays a number input for augmentation multiplier
- **AND** the multiplier range is 1 to 5 (default: 2)
- **AND** the multiplier indicates how many times the original data will be augmented
- **AND** a description explains that 2x means doubling the training data size

#### Scenario: Augmentation information display
- **WHEN** data augmentation is enabled
- **THEN** the system displays information explaining:
  - What data augmentation does
  - Which columns will be augmented (numeric columns only)
  - Which columns will not be augmented (categorical columns)
  - That only the training set will be augmented, not the test set

### Requirement: Data Augmentation Execution
The system SHALL augment training data according to user configuration before model training.

#### Scenario: Augment numeric features only
- **WHEN** data augmentation is enabled and training starts
- **THEN** the system identifies numeric columns in the training data
- **AND** noise is added only to numeric columns
- **AND** categorical columns remain unchanged

#### Scenario: Apply noise to training data
- **WHEN** data augmentation is enabled with Gaussian noise
- **THEN** the system adds Gaussian noise to each numeric column
- **AND** the noise is sampled from a normal distribution with mean 0 and standard deviation equal to (column_std * noise_strength)
- **AND** the noise is added to the original values

#### Scenario: Apply uniform noise to training data
- **WHEN** data augmentation is enabled with uniform noise
- **THEN** the system adds uniform noise to each numeric column
- **AND** the noise is sampled from a uniform distribution in the range [-column_std * noise_strength, column_std * noise_strength]
- **AND** the noise is added to the original values

#### Scenario: Generate augmented samples
- **WHEN** data augmentation multiplier is set to N
- **THEN** the system generates N-1 additional augmented copies of the training data
- **AND** each copy has independent noise added
- **AND** all augmented copies are combined with the original training data

#### Scenario: Preserve test set
- **WHEN** data augmentation is enabled
- **THEN** the system augments only the training set (X_train, y_train)
- **AND** the test set (X_test, y_test) remains unchanged
- **AND** this ensures accurate model evaluation

#### Scenario: Augmentation statistics display
- **WHEN** data augmentation completes
- **THEN** the system displays before/after statistics:
  - Number of samples (before and after)
  - Mean and standard deviation for each numeric column (before and after)
- **AND** the statistics are displayed in a clear, easy-to-read format
- **AND** users can see the effect of augmentation on their data

#### Scenario: Augmentation failure handling
- **WHEN** data augmentation fails (e.g., invalid parameters, memory error)
- **THEN** a clear error message is displayed
- **AND** the error message suggests how to fix the issue
- **AND** training can proceed without augmentation if the user chooses

