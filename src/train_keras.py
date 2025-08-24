from utils import load_images, validate_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential

def main():
    image_size = 64
    X, y = load_images("./images", image_size=(image_size, image_size))
    validate_dataset(X, y)

    X = X.reshape((-1, image_size, image_size, 1))
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2)

    model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(image_size, image_size, 1)
            ),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    model.save('cnn.keras')

if __name__ == "__main__":
    main()