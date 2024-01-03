# EL7008
Clasificación de Imágenes con Vision Transformer

Hecho por Luciano Vidal

En este repositorio se incluyen los distintos códigos utilizados para la implementación, entrenamiento y prueba del sistema de clasificación.

***
Este repositorio contiene los siguientes archivos:

* YogaDataset.py: código de python que permite leer la base de datos Yoga-84, permitiendo asignar la clase de las imagenes segun el nivel de jerarquía deseado.
* vit_model: código de python con todos los bloques que permiten la construcción del Vision Transformer utilizado para clasificar.
* train.py: código con las funciones utilizadas para realizar el entrenamiento del modelo y mostrar los resultados obtenidos.
* Notebooks/: directorio que contiene los seis notebooks en los cuales se realizaron los 6 entrenamientos solicitados, tanto los entrenamiento desde cero como utilizando *transfer learning*.
* demo/: directorio que contiene 6 imágenes de prueba y un notebook que permite probar los resultados del modelo entrenado al utilizar 6 clases. Importante notar que este notebook está diseñado para ejecutarse de forma **independiente y secuencial** en Google Colab.

***