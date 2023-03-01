import tensorflow as tf
from keras import optimizers

tf.random.set_seed(42)

train_dir = "datax13"

trainDatagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = trainDatagen.flow_from_directory(directory=train_dir,
                                              batch_size=32,
                                              target_size=(224,224),
                                              class_mode="categorical",
                                              shuffle=True,
                                              seed=42)

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(filters = 32,
                           kernel_size = (3,3),
                           activation = 'relu',
                           input_shape = (224,224,3)),
    
    tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = 2,
                              padding = 'valid'),
    
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = 2,
                              padding = 'valid'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(26, activation = 'softmax')
])

model.compile(loss = "categorical_crossentropy",
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ["accuracy"])

history = model.fit(train_data, epochs=3)

model.save("sign_model14d")
