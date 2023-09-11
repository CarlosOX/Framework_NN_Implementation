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

# Pasar nuestros datos a arreglos de numpy
X_data = data_encoded.to_numpy()
Y_data = labels.to_numpy()

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Escalar los atributos para un mejor rendimiento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Crear un modelo de red neuronal 

model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation ="relu"),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),    #  <- Dropouts para redicr overfitting
    tf.keras.layers.Dense(128, activation ="relu"),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation ="relu"),
    tf.keras.layers.BatchNormalization(),  # <- Batch normalisation layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation ="relu"),
    tf.keras.layers.Dense(1, activation='sigmoid')   # Funcion sigmoid ya que es una salida BOOL
])

# Compilar el modelo
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model_nn.fit(X_train, y_train, epochs=100, batch_size=256, verbose=2, validation_data=(X_val, y_val))

# Ver el accuracy con los datos de test

y_pred_nn = (model_nn.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f"Precisión del modelo de red neuronal en el conjunto de prueba: {accuracy_nn:.2f}")
print(classification_report(y_test, y_pred_nn))


while True:
    print("Menu:")
    print("1. Ver gráficas")
    print("2. Realizar una predicción aleatoria")
    print("3. Salir")
    opcion = input("Seleccione una opción: ")

    if opcion == "1":

        # Codigo para realizar una prediccion
        # Crear la matriz de confusion
        conf_matrix = confusion_matrix(y_test, y_pred_nn)

        # Grafico de la matriz de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión')

       
        # Graficos de perdida y precision en entrenamiento y validacion
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

        # Mostrar todos los graficos al mismo tiempo
        plt.show()

    elif opcion == "2":
        

        def predecir_si_fue_transportado(X, modelo):  # Esta funcion predice un valor aleatorio dado de los datos de test

            X = np.array(X)
            X = scaler.transform(X)    
            prediccion = modelo.predict(X)
            
            # La salida de 'modelo.predict' sera una probabilidad, conviértela en una etiqueta (0 o 1)
            etiqueta_predicha = (prediccion > 0.5).astype(int)
            
            return etiqueta_predicha


        rdom = random.randint(1, 600)


        nueva_entrada = X_test[rdom].reshape(1,-1)  # Cambiar a la forma que acepta el modelo 
        resultado = predecir_si_fue_transportado(nueva_entrada, model_nn)

        if resultado[0] == 1:

            print("output del modelo : Fue transportado.")
        else:
            print("output del modelo: No fue transportado")
        
        if y_test[rdom] == 1:

            print("Valor real : Fue transportado.")
        else:
            print("Valor real: No fue transportado")



    elif opcion == "3":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")