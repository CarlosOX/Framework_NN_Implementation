import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from tensorflow.keras.regularizers import l2


# Definir una función para entrenar la red neuronal

# Cargar los datos
data_train = pd.read_csv('train_titanic.csv')

# Quitamos todos los valores nan ya que no son muchos datos los que perdemos.
data_train.dropna(inplace=True)

# Separar la columna Cabin por sus '/', para poder aplicar un dummy vector después 
data_train['Deck'] = data_train['Cabin'].str.split('/').str[0]

# Extraer el lado (Side)
data_train['Side'] = data_train['Cabin'].str.split('/').str[2]

# Cambiar los valores booleanos a enteros para su manipulación
data_train[['VIP', 'CryoSleep', 'Transported']] = data_train[['VIP', 'CryoSleep', 'Transported']].astype(int)

# Crear los dummy vector para los planetas, el destino, y los asientos de la cabina
data_train = pd.get_dummies(data_train, columns=["HomePlanet", "Destination", "Deck", "Side"], dtype=int)

# Crear nuestro dataframe para los labels de la columna transported
labels = data_train['Transported']

# Dropear columnas que ya no nos sirven para nuestro dataset final
data_encoded = data_train.drop(['Cabin', 'PassengerId', 'Name', 'Transported'], axis=1)


# Realiza el PCA
pca = PCA()
pca.fit(data_encoded)

# Calcula la varianza explicada acumulada
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

pca_breast = PCA(n_components=7)
principalComponents_breast = pca_breast.fit_transform(data_encoded)

PCA_DF = pd.DataFrame(data = principalComponents_breast
             , columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7'])



#Pasamos nuestros datos a arreglos de numpy
X_pca = PCA_DF.to_numpy()
Y_data = labels.to_numpy()


# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X_pca, Y_data, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Escalar los atributos para un mejor rendimiento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=l2(0.008), input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation ="relu",kernel_regularizer=l2(0.008)),
    tf.keras.layers.Dropout(0.5),    #  <- Dropouts para redicr overfitting
    tf.keras.layers.Dense(128, activation ="relu"),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation ="relu"),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation ="relu"),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')   # Funcion sigmoid ya que es una salida BOOL
])
# Compilar el modelo 

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)  # ajustamos la taza de aprendizaje
model_nn.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model_nn.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2, validation_data=(X_val, y_val))

# Ver el accuracy con los datos de test

y_pred_nn = (model_nn.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f"Precisión del modelo de red neuronal en el conjunto de prueba: {accuracy_nn:.2f}")
print(classification_report(y_test, y_pred_nn))


while True:
    print("Menu:")
    print("1. Ver gráficas")
    print("2. Salir")
   
    
    opcion = input("Seleccione una opción: ")

    if opcion == "1":

        # Codigo para realizar una prediccion
        # Crear la matriz de confusion
        conf_matrix = confusion_matrix(y_test, y_pred_nn)

        # Grafica la varianza explicada acumulada
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Varianza Explicada Acumulada vs. Número de Componentes Principales')
        plt.grid(True)


        # Grafico de la matriz de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión')

       
        # Gráficos de pérdida y precisión en entrenamiento y validación
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Loss vs. Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy vs. Epochs')

        plt.show()
    elif opcion == "2":
        print("Saliendo del programa.")
        break
        
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")