# training-page Specification

## Purpose
TBD - created by archiving change add-training-page. Update Purpose after archive.
## Requirements
### Requirement: Data Source Selection
The system SHALL allow users to select training data from multiple sources.

#### Scenario: Data from analysis page
- **WHEN** a user navigates to the training page after analyzing data
- **THEN** the system automatically detects data from the data analysis page
- **AND** the user can use this data for training without re-uploading

#### Scenario: New data upload
- **WHEN** no data is available from the analysis page
- **THEN** the system provides a file uploader for CSV files
- **AND** the user can upload new data for training
- **AND** the uploaded data is validated before training

### Requirement: Target Variable Selection
The system SHALL allow users to select one or multiple target variables for training.

#### Scenario: Single target variable selection
- **WHEN** a user wants to predict a single variable
- **THEN** the system displays a dropdown or multi-select component showing all available columns
- **AND** the user can select one column as the target variable
- **AND** the selected column is clearly indicated

#### Scenario: Multiple target variables selection
- **WHEN** a user wants to predict multiple variables simultaneously
- **THEN** the system allows selecting multiple columns as target variables
- **AND** the selected columns are clearly indicated
- **AND** the system trains a model that can predict all selected targets

#### Scenario: Feature variable identification
- **WHEN** target variables are selected
- **THEN** all other columns are automatically identified as feature variables
- **AND** the system excludes target variables from features

### Requirement: Training Algorithm Selection
The system SHALL allow users to choose between different training algorithms.

#### Scenario: Algorithm selection
- **WHEN** a user is ready to train a model
- **THEN** the system displays available algorithms (Linear Regression, Gradient Descent)
- **AND** the user can select one algorithm from a dropdown or radio buttons
- **AND** each algorithm has a brief description of its characteristics

### Requirement: Loss Function Selection
The system SHALL provide a simple interface for selecting loss functions.

#### Scenario: Loss function selection
- **WHEN** a user selects a training algorithm
- **THEN** the system displays available loss functions (MSE, MAE) in a simple dropdown
- **AND** each loss function has a clear, non-technical description
- **AND** the default loss function is MSE
- **AND** for standard Linear Regression, loss function is used for evaluation only

#### Scenario: Gradient descent loss function
- **WHEN** a user selects Gradient Descent algorithm
- **THEN** the selected loss function affects the training process
- **AND** the loss function is used to compute gradients during training

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
- **THEN** no additional parameters are required
- **AND** the training starts immediately after data and algorithm selection

### Requirement: Training Execution
The system SHALL execute training and provide feedback during the process.

#### Scenario: Training start
- **WHEN** a user clicks the "Train" button
- **THEN** the system validates all inputs (data, target variables, parameters)
- **AND** if validation passes, training begins
- **AND** if validation fails, clear error messages are displayed

#### Scenario: Training progress
- **WHEN** training is in progress
- **THEN** a loading indicator or progress message is displayed
- **AND** for Gradient Descent, the loss values are recorded during training
- **AND** the user understands that training is in progress

#### Scenario: Training completion
- **WHEN** training completes successfully
- **THEN** a success message is displayed
- **AND** training results are immediately available for viewing
- **AND** the trained model is stored in session state for use in prediction

#### Scenario: Training failure
- **WHEN** training fails
- **THEN** a clear, non-technical error message is displayed
- **AND** the error message suggests how to fix the issue
- **AND** the application remains functional for other operations

### Requirement: Training Results Display
The system SHALL display comprehensive training results.

#### Scenario: Loss curve display
- **WHEN** Gradient Descent training completes
- **THEN** a loss curve chart is displayed showing loss values over iterations
- **AND** the chart is interactive (using Plotly)
- **AND** users can see how the loss decreased during training

#### Scenario: Model parameters display
- **WHEN** training completes
- **THEN** model parameters (coefficients, intercept) are displayed
- **AND** the parameters are clearly labeled and easy to read
- **AND** for multiple target variables, parameters for each target are shown separately

#### Scenario: Evaluation metrics display
- **WHEN** training completes
- **THEN** evaluation metrics are displayed in a table:
  - R² score (coefficient of determination)
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
- **AND** metrics are calculated for each target variable (if multiple)
- **AND** metrics are clearly labeled and easy to interpret

### Requirement: Model Saving
The system SHALL allow users to save trained models for later use.

#### Scenario: Model save
- **WHEN** a user wants to save a trained model
- **THEN** the system provides an input field for model name
- **AND** the user can enter a descriptive name
- **AND** clicking "Save Model" saves the model to the models/ directory
- **AND** the model file includes timestamp for versioning
- **AND** a success message confirms the save operation

#### Scenario: Model save validation
- **WHEN** a user attempts to save without a model name
- **THEN** an error message prompts for a model name
- **AND** when a model name is provided, the save proceeds

### Requirement: Data Validation for Training
The system SHALL validate training data before training begins.

#### Scenario: Target variable validation
- **WHEN** a user selects target variables
- **THEN** the system checks that selected columns exist in the data
- **AND** the system checks that target variables contain numeric data
- **AND** if validation fails, clear error messages are displayed

#### Scenario: Feature variable validation
- **WHEN** training data is prepared
- **THEN** the system checks that at least one feature variable exists
- **AND** the system checks that feature variables are numeric or can be converted
- **AND** missing values are handled (removed or filled with default strategy)

#### Scenario: Data size validation
- **WHEN** training data is prepared
- **THEN** the system checks that there are enough samples for training
- **AND** if data is too small, a warning is displayed

### Requirement: User-Friendly Training Interface
The system SHALL provide a clear, intuitive interface for non-technical users.

#### Scenario: Clear instructions
- **WHEN** a user visits the training page
- **THEN** clear step-by-step instructions are displayed
- **AND** the instructions use simple, non-technical language
- **AND** each step is clearly numbered or organized

#### Scenario: Visual feedback
- **WHEN** a user interacts with the training interface
- **THEN** visual feedback confirms selections (highlighting, checkmarks)
- **AND** the current step in the training process is clearly indicated
- **AND** disabled options are grayed out appropriately

#### Scenario: Error prevention
- **WHEN** a user makes an invalid selection
- **THEN** the system prevents proceeding to the next step
- **AND** helpful hints are displayed explaining what needs to be corrected

### Requirement: Repeated Random Split Evaluation Option
The system SHALL allow users to choose between single evaluation and repeated random split evaluation.

#### Scenario: Evaluation method selection
- **WHEN** a user is ready to train a model
- **THEN** the system displays radio buttons or dropdown for evaluation method selection:
  - Single evaluation (default)
  - Repeated random split evaluation
- **AND** each method has a brief description explaining its characteristics

#### Scenario: Repeated split configuration
- **WHEN** repeated random split evaluation is selected
- **THEN** the system displays a number input for number of repetitions
- **AND** the repetition range is 3 to 20 (default: 5)
- **AND** a description explains that more repetitions provide more stable results but take longer

#### Scenario: Evaluation method information display
- **WHEN** repeated random split evaluation is selected
- **THEN** the system displays information explaining:
  - What repeated random split evaluation does
  - How it differs from single evaluation
  - That it provides more stable and reliable results
  - That it takes longer than single evaluation

### Requirement: Repeated Random Split Evaluation Execution
The system SHALL perform repeated random split evaluation when selected.

#### Scenario: Execute repeated evaluations
- **WHEN** repeated random split evaluation is selected with N repetitions
- **THEN** the system performs N independent random splits and model training
- **AND** each repetition uses a different random seed to ensure independence
- **AND** evaluation metrics (R², MSE, MAE) are calculated for each repetition

#### Scenario: Calculate aggregate statistics
- **WHEN** all repetitions complete
- **THEN** the system calculates mean and standard deviation for each metric
- **AND** the statistics are calculated across all repetitions
- **AND** the results are displayed in a clear format

#### Scenario: Display evaluation results
- **WHEN** repeated random split evaluation completes
- **THEN** the system displays:
  - Mean and standard deviation for each metric (R², MSE, MAE)
  - Format: "Mean ± Std" (e.g., "0.85 ± 0.03")
  - Optionally: detailed results table showing each repetition's metrics
- **AND** the results are clearly labeled and easy to interpret

#### Scenario: Progress indication
- **WHEN** repeated random split evaluation is in progress
- **THEN** a progress indicator is displayed showing:
  - Current repetition number
  - Total number of repetitions
  - Estimated time remaining (if possible)
- **AND** the user understands that evaluation is in progress

#### Scenario: Single evaluation fallback
- **WHEN** single evaluation is selected
- **THEN** the system performs evaluation as before (single random split)
- **AND** the evaluation results are displayed in the same format as before
- **AND** no changes to existing behavior

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

