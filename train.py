import numpy as np
import time
import torch
import matplotlib.pyplot as plt

# codigo que ejecuta el entrenamiento en una epoca, realizando
# la actualizacion de los pesos del modelo
def train_one_epoch(model, dataloader, optimizer, loss_fn):
  train_loss = [] # se almacena la loss de cada batch
  # se opera cada batch
  for i, data in enumerate(dataloader):
    # se leen los datos
    inputs, labels = data
    # se llevan los datos a la gpu
    inputs = inputs.to('cuda')
    labels = labels.to('cuda')
    # se fijan los gradientes en cero
    optimizer.zero_grad()
    # se generan las predicciones
    outputs = model(inputs)

    # se calcula la funcion de perdida y los gradientes
    loss = loss_fn(outputs, labels)
    loss.backward()
    # se ajustan los pesos del modelo
    optimizer.step()
    # se guarda la loss del batch
    train_loss.append(loss.detach().cpu().numpy())
  # se entrega la loss promedio
  return np.mean(train_loss)


# funcion para calcular la loss en validacion
def get_val_loss(model, dataloader, loss_fn):
  val_loss = []
  # no se calculan gradientes
  with torch.no_grad():
    # se analiza cada batch en validacion
    for i, data in enumerate(dataloader):
        # se leen los datos
        inputs, labels = data
        # se llevan a la GPU
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        # se evalua el modelo
        outputs = model(inputs)
        # se calcula la funcion de perdida
        loss = loss_fn(outputs, labels)
        # se guarda el valor de la función de perdida
        val_loss.append(loss.detach().cpu().numpy())
  return np.mean(val_loss) # se entrega la loss promedio


# funcion que permite ejecutar el entrenamiento de un modelo
# dados los hiperparametros deseados
def train_model(model,
                trainloader,
                valloader,
                optimizer,
                loss_fn,
                epochs,
                model_path # str con la direccion en que se desean guardar los parametros
                ):
  
  # listas para almacenar la evolución del loss en train y validacion
  train_loss_evol = []
  val_loss_evol = []

  # listas para almacenar la evolucion del accuracy en train y validacion
  train_acc_evol = []
  val_acc_evol = []

  # se inicializan variables del entrenamiento
  best_vloss = np.inf

  begin_time = time.time() # tiempo inicial

  # Loop de entrenamiento
  for epoch in range(epochs):
      print(f'EPOCH: {epoch + 1}')

      # se realiza el entrenamiento en test
      model.train()
      avg_loss = train_one_epoch(model, trainloader, optimizer, loss_fn)

      # se evalua el modelo en validacion
      model.eval()
      avg_vloss = get_val_loss(model, valloader, loss_fn)

      # se guardan los loss de test y validacion
      train_loss_evol.append(avg_loss)
      val_loss_evol.append(avg_vloss)
      

      # se guarda el mejor modelo considerando la loss en validacion
      if avg_vloss < best_vloss:
          best_vloss = avg_vloss
          # se guardan los parametros con menor loss en validacion
          torch.save(model.state_dict(), model_path)
          print(f'Modelo guardado en EPOCH {epoch+1}')
          
      # se obtiene el accuracy por epoca en train
      total = 0
      correct = 0
      model.eval()
      with torch.no_grad():
        for i, data in enumerate(trainloader):
          # se leen los datos
          inputs, labels = data
          # se llevan a la GPU
          inputs = inputs.to('cuda')
          labels = labels.to('cuda')
          # se obtienen las salidas del modelo
          outputs = model(inputs)
          # se va sumando el total de datos
          total += labels.size(0)
          # se obtienen las predicciones
          _, predicted = torch.max(outputs.data, 1)
          # se cuentan las predicciones que coinciden
          # con la etiqueta correcta
          correct += (predicted == labels).sum().item()
      # se calcula y guarda el accuracy
      accuracy_train = correct/total
      train_acc_evol.append(accuracy_train)

      # se obtiene el accuracy por epoca en validacion
      total = 0
      correct = 0
      model.eval()
      with torch.no_grad():
        for i, data in enumerate(valloader):
          # se leen los datos
          inputs, labels = data
          # se llevan a la GPU
          inputs = inputs.to('cuda')
          labels = labels.to('cuda')
          # se obtienen las salidad del modelo
          outputs = model(inputs)
          # se va sumando el total de datos
          total += labels.size(0)
          # se obtienen las predicciones
          _, predicted = torch.max(outputs.data, 1)
          # se cuentan las predicciones que coinciden
          # con la etiqueta correcta
          correct += (predicted == labels).sum().item()
      # se calcula y guarda el accuracy
      accuracy_val = correct/total
      val_acc_evol.append(accuracy_val)

      # se imprime la evolucion a lo largo de cada epoca
      print(f'LOSS train {avg_loss} valid {avg_vloss} | ACC train {accuracy_train} val {accuracy_val}')
      

  # se guarda el tiempo de entrenamiento
  end_time = time.time()
  execution_time = round(end_time - begin_time,1)

  # se guarda en un diccionario todas las metricas del entrenamiento
  results = {'train_loss' : train_loss_evol,
             'val_loss': val_loss_evol,
             'train_acc': train_acc_evol,
             'val_acc': val_acc_evol,
             'execution_time': execution_time}
  
  # se entregan las metricas resultantes
  return results

# funcion que recibe las metricas resultantes de un entrenamiento
# y genera graficos con la evolucion de la loss y el accuracy
def plot_results(results):

  # se leen las métricas
  train_loss_evol = results['train_loss']
  val_loss_evol = results['val_loss']
  train_acc_evol = results['train_acc']
  val_acc_evol = results['val_acc']
  execution_time = results['execution_time']

  # se crea el grafico con la evolucion de la loss
  fig, (ax) = plt.subplots(figsize=(6,4))
  fig.suptitle('Evolucion funcion de perdida')
  ax.set_title(f'Tiempo: {execution_time} s', color = 'gray', style='italic', fontsize=10)
  ax.plot(val_loss_evol, label = 'Validation')
  ax.plot(train_loss_evol, label = 'Train')
  ax.set_ylabel('Loss')
  ax.set_xlabel('Época')
  ax.grid(alpha=0.4)
  plt.legend()
  plt.show()

  # se crea el grafico con la evolucion del accuracy
  fig, (ax) = plt.subplots(figsize=(6,4))
  fig.suptitle('Evolucion en accuracy')
  ax.set_title(f'Tiempo: {execution_time} s', color = 'gray', style='italic', fontsize=10)
  ax.plot(val_acc_evol, label = 'Validation')
  ax.plot(train_acc_evol, label = 'Train')
  ax.set_ylabel('Accuracy')
  ax.set_xlabel('Época')
  ax.grid(alpha=0.4)
  plt.legend()
  plt.show()