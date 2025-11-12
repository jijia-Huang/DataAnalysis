## ADDED Requirements

### Requirement: Model Abstract Framework
The system SHALL provide an abstract base class for all machine learning models to ensure consistent interface and easy model replacement.

#### Scenario: Base model interface
- **WHEN** a developer creates a new model class
- **THEN** the model MUST inherit from BaseModel
- **AND** the model MUST implement fit, predict, save, load, and get_info methods
- **AND** all models can be used interchangeably through the same interface

#### Scenario: Model replacement
- **WHEN** a user wants to switch from one model type to another
- **THEN** the system allows replacing the model without changing other code
- **AND** the new model works with the same training and prediction interfaces

### Requirement: Model Training Interface
The system SHALL provide a unified fit method for training models.

#### Scenario: Single target variable training
- **WHEN** a model is trained with a single target variable
- **THEN** the fit method accepts feature matrix X and target vector y
- **AND** the model trains successfully and stores the learned parameters

#### Scenario: Multiple target variables training
- **WHEN** a model is trained with multiple target variables
- **THEN** the fit method accepts feature matrix X and target matrix Y (multiple columns)
- **AND** the model trains successfully for all target variables
- **AND** the model can predict all target variables simultaneously

### Requirement: Model Prediction Interface
The system SHALL provide a unified predict method for making predictions.

#### Scenario: Single prediction
- **WHEN** a trained model receives feature data
- **THEN** the predict method returns predictions for the target variable(s)
- **AND** the predictions match the format of the training target (single or multiple columns)

### Requirement: Model Persistence
The system SHALL provide methods to save and load trained models.

#### Scenario: Model saving
- **WHEN** a user saves a trained model
- **THEN** the model object, training parameters, target variable names, and feature names are saved
- **AND** the model is saved in a format that can be loaded later
- **AND** the saved file includes metadata for model identification

#### Scenario: Model loading
- **WHEN** a user loads a saved model
- **THEN** the model is restored with all its parameters
- **AND** the model can immediately make predictions without retraining
- **AND** the model metadata (target variables, features) is available

### Requirement: Model Information
The system SHALL provide a method to retrieve model information.

#### Scenario: Model info retrieval
- **WHEN** a user requests model information
- **THEN** the system returns model type, training parameters, target variables, and feature names
- **AND** the information is displayed in a user-friendly format

