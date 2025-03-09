import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np

def identity_block(x, filters, kernel_size=3):
    """
    Implementation of the identity block for ResNet
    
    Arguments:
        x: input tensor
        filters: list of integers, the filters of 3 conv layer at main path
        kernel_size: default 3, the kernel size of middle conv layer at main path
    
    Returns:
        Output tensor for the block
    """
    filters1, filters2, filters3 = filters
    
    shortcut = x
    
    # First component
    x = layers.Conv2D(filters1, (1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second component
    x = layers.Conv2D(filters2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Third component
    x = layers.Conv2D(filters3, (1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut connection
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def conv_block(x, filters, kernel_size=3, stride=2):
    """
    Implementation of the convolutional block for ResNet
    
    Arguments:
        x: input tensor
        filters: list of integers, the filters of 3 conv layer at main path
        kernel_size: default 3, the kernel size of middle conv layer at main path
        stride: stride for the first convolution and shortcut
    
    Returns:
        Output tensor for the block
    """
    filters1, filters2, filters3 = filters
    
    # Shortcut connection with 1x1 conv to match dimensions
    shortcut = layers.Conv2D(filters3, (1, 1), strides=stride, padding='valid')(x)
    shortcut = layers.BatchNormalization()(shortcut)
    
    # First component with stride
    x = layers.Conv2D(filters1, (1, 1), strides=stride, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second component
    x = layers.Conv2D(filters2, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Third component
    x = layers.Conv2D(filters3, (1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut connection
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_custom_resnet(input_shape, num_classes):
    """
    Create a custom ResNet model
    
    Arguments:
        input_shape: shape of input images
        num_classes: number of classes
    
    Returns:
        ResNet model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # ResNet blocks
    # Block 1
    x = conv_block(x, [64, 64, 256], stride=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])
    
    # Block 2
    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    
    # Block 3
    x = conv_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    
    # Block 4
    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_pretrained_resnet(input_shape, num_classes, freeze_layers=True):
    """
    Create a ResNet model using pre-trained weights from ImageNet
    
    Arguments:
        input_shape: shape of input images
        num_classes: number of classes
        freeze_layers: whether to freeze the pre-trained layers
    
    Returns:
        ResNet model with pre-trained weights
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers if required
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model

def prepare_data(processed_images, image_labels, target_size=(256, 256, 3), test_split=0.2):
    """
    Prepare the processed data for the model
    
    Arguments:
        processed_images: Numpy array of processed images
        image_labels: Numpy array of image labels
        target_size: Target size of the images
        test_split: Proportion of data to use for testing
    
    Returns:
        train_images, test_images, train_labels, test_labels
    """
    # Convert labels to categorical
    num_classes = len(np.unique(image_labels))
    image_labels = tf.keras.utils.to_categorical(image_labels, num_classes)
    
    # Reshape flat images back to 3D if they were flattened
    if len(processed_images.shape) == 2:  # If images are flattened
        num_samples = processed_images.shape[0]
        processed_images = processed_images.reshape(num_samples, target_size[0], target_size[1], target_size[2])
    
    # Split data into training and testing sets
    total_samples = processed_images.shape[0]
    indices = np.random.permutation(total_samples)
    test_size = int(total_samples * test_split)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_images = processed_images[train_indices]
    test_images = processed_images[test_indices]
    train_labels = image_labels[train_indices]
    test_labels = image_labels[test_indices]
    
    return train_images, test_images, train_labels, test_labels

def train_model(model, train_images, train_labels, test_images, test_labels, 
               batch_size=32, epochs=50, learning_rate=0.001):
    """
    Train the ResNet model
    
    Arguments:
        model: The ResNet model
        train_images: Training images
        train_labels: Training labels
        test_images: Testing images
        test_labels: Testing labels
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Trained model and training history
    """
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_images, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

def evaluate_model(model, test_images, test_labels):
    """
    Evaluate the trained model
    
    Arguments:
        model: Trained model
        test_images: Test images
        test_labels: Test labels
    
    Returns:
        Test loss and accuracy
    """
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

if __name__ == "__main__":
    # Example usage (will need to be adjusted based on your actual data)
    from preprocessing import process_images
    
    # Parameters
    dataset_folder = "/Users/chris/Documents/UdeMHacks/Parasite Data Set"
    target_size = (256, 256)
    input_shape = (256, 256, 3)
    num_classes = 8  # Adjust based on your dataset
    
    # Process images
    processed_images, image_labels = process_images(dataset_folder, target_size)
    
    # Prepare data
    train_images, test_images, train_labels, test_labels = prepare_data(
        processed_images, image_labels, target_size=(256, 256, 3)
    )

    model = create_custom_resnet(input_shape, num_classes)
    
    # Train model
    trained_model, history = train_model(
        model, train_images, train_labels, test_images, test_labels,
        batch_size=32, epochs=50
    )
    
    # Evaluate model
    evaluate_model(trained_model, test_images, test_labels)
    
    # Save model
    trained_model.save("resnet_model.h5")