import tensorflow as tf
from tensorflow.keras import layers, models

class VGGM(models.Model):
    
    def __init__(self, n_classes=1251):
        super(VGGM, self).__init__()
        self.n_classes = n_classes
        self.features = models.Sequential([
            layers.Input(shape=(512, 300, 1)),
            layers.Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(momentum=0.5),
            layers.MaxPooling2D((3, 3), strides=(2, 2)),
            layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(momentum=0.5),
            layers.MaxPooling2D((3, 3), strides=(2, 2)),
            layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(momentum=0.5),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(momentum=0.5),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(momentum=0.5),
            layers.MaxPooling2D((5, 3), strides=(3, 2)),
            layers.Conv2D(4096, (9, 1)),
            layers.BatchNormalization(momentum=0.5),
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),
        ])
        
        self.classifier = models.Sequential([
            layers.Dense(1024, activation='relu', name = 'fc7'),
            layers.Dense(n_classes, name = 'fc8')
        ])
    
    def call(self, inputs):
        x = self.features(inputs)
        return self.classifier(x)

if __name__ == "__main__":
    model = VGGM(1251)
    model.build((None, 512, 300, 1))
    model.summary()
    