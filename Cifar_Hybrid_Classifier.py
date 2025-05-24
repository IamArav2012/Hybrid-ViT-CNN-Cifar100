import tensorflow as tf
import numpy as np
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from keras import regularizers, layers

image_size = 32
regularizers_rate = 0.001
patch_size = 4
projection_dim = 64
batch_size = 32
num_heads = 4
transformer_layers = 6
ffn_projection = [128, 64]
last_vit_dropout = 0.3
last_mlp_projection = [256, 128]
num_patches = (image_size // patch_size) ** 2


def build_cnn(): 

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x1 = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(regularizers_rate))(inputs)
    x1 = layers.BatchNormalization()(x1)
    feature_maps = layers.Activation('relu')(x1)

    for num_filters in [64, 128, 256]:

        x1 = layers.Conv2D(num_filters, 3, padding="same", kernel_regularizer=regularizers.l2(regularizers_rate))(feature_maps)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        if num_filters != 256:
            x1 = layers.MaxPooling2D((2,2), strides=2, padding='same')(x1)
        feature_maps = x1
    
    cnn_vector = layers.GlobalAveragePooling2D()(feature_maps) # G.A.P to reduce computational cost

    model = tf.keras.Model(inputs=inputs, outputs=cnn_vector)
    return model


class Patchify(tf.keras.layers.Layer):

    def __init__(self, patch_size): 
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            rates=[1, 1, 1, 1],
            padding='VALID',
            strides=[1, self.patch_size, self.patch_size, 1],
            sizes=[1, self.patch_size, self.patch_size, 1],
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
        
class PatchEncoder(tf.keras.layers.Layer): 
    def __init__(self, num_patches): 
        super().__init__()
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.pos_embedding = layers.Embedding(
            input_dim=num_patches + 1,  # +1 for CLS token
            output_dim=projection_dim
        )
        self.cls = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]

        # Project patches to higher dimensional space
        patch_embedding = self.projection(patches)

        # Positional indices 
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = self.pos_embedding(positions)

        # Match batch shape for broadcasting
        pos_embedding = tf.expand_dims(pos_embedding, axis=0)

        # Add positional info
        encoded = patch_embedding + pos_embedding

        # Broadcast cls token across batch
        cls_tokens = tf.broadcast_to(self.cls, [batch_size, 1, self.projection_dim])

        # Concatenate [CLS] token in front
        encoded_patches = tf.concat([cls_tokens, encoded], axis=1)

        return encoded_patches
    
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_vit(): 
    
    inputs = layers.Input(shape=(image_size, image_size, 3))
    patches = Patchify(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches)(patches)
    
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        self_att_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([self_att_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x4 = mlp(x3, hidden_units=ffn_projection, dropout_rate=0.1)
        encoded_patches = layers.Add()([x4, x3])
    
    vit_vector = encoded_patches[:,0,:]
    vit_vector = layers.Dropout(last_vit_dropout)(vit_vector)
    model = tf.keras.Model(inputs=inputs, outputs=vit_vector)
    return model

# Hybrid Model
def build_hybrid_classifier():
    # Input layer
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Submodels
    vit_vector = build_vit()(inputs)     
    cnn_features = build_cnn()(inputs)    

    # MLP on both
    vit_vector = mlp(vit_vector, hidden_units=last_mlp_projection, dropout_rate=0.5)
    cnn_vector = mlp(cnn_features, hidden_units=last_mlp_projection, dropout_rate=0.5)

    # Combine both
    combined = layers.Concatenate()([vit_vector, cnn_vector])

    # Final classifier
    outputs = layers.Dense(100, activation='softmax')(combined)

    # Full model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
        

def augment_fn(image, label):
    # Horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

    # Brightness adjustment
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)

    # Contrast adjustment
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, label

(x_train, y_train), (x_test_actual, y_test_actual) = cifar100.load_data()

x_train = x_train.astype('float32') / 255
x_test_actual = x_test_actual.astype('float32') / 255

x_train_actual, x_val, y_train_actual, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42
)

print('Shapes: x_train:{}, x_test:{}, x_val:{}'.format(x_train_actual.shape, x_test_actual.shape, x_val.shape))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_actual, y_train_actual)) \
    .shuffle(buffer_size=len(x_train_actual)) \
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_actual, y_test_actual)) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

if __name__ == '__main__':
    vit_model = build_vit()
    cnn_model = build_cnn()
    cnn_model.summary()
    vit_model.summary()

    hybrid_model = build_hybrid_classifier()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    hybrid_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=4,
        restore_best_weights=True,
        min_delta=0.001
    )
    hybrid_model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=early_stopping)

    y_pred = hybrid_model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test_actual, y_pred_classes))

    hybrid_model.save('cifar_hybrid.keras')