## 00 Pytoch fundamentals

import torch
#2.15.0+cu132
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print(torch.__version__)


##Introduccion de Tensores
### Crear tensores - Se crean utilizando torch.Tensor()
#scalar
scalar = torch.tensor(7)
scalar 

scalar.ndim


#Volver a tener un tensor como Int 
scalar.item()


print("scalar:", scalar)
print("dimensiones del scalar:", scalar.ndim)
print("valor del scalar:", scalar.item())


#Vector
vector = torch.tensor([7,7])
vector.ndim #devuelve la cantidad de corchetes cerrados


vector.shape# devuelve el size del vector

print("vector:", vector)
print("dimensiones del vector:", vector.ndim)
print("forma del vector:", vector.shape)



# MATRIX
MATRIX = torch.tensor([[7, 7], [7, 8]])


MATRIX.ndim # 2 
MATRIX[0] # [7, 7]
MATRIX.shape # ([2, 2])

print("MATRIX:", MATRIX)
print("dimensiones de MATRIX:", MATRIX.ndim)
print("primera fila:", MATRIX[0])
print("forma de MATRIX:", MATRIX.shape)



# TENSOR 
TENSOR = torch.tensor([[[1, 2, 45], [3, 5, 2], [2, 3, 5]]])

print("TENSOR:", TENSOR)
print("dimensiones de TENSOR:", TENSOR.ndim)
print("forma de TENSOR:", TENSOR.shape)
print(TENSOR[0])

### RANDOM TENSORS 
## Porque random tensors : Son importantes porque en la forma en la que NN aprenden es con random numbers, 
## y los ajustan para representar mejor la data 

#Crear un random tensor of size (3, 4)

random_tensor = torch.rand(3, 4, 4)

print (random_tensor)
print(random_tensor.ndim, random_tensor.shape)


##ndim es el numero de dimensiones, es decir de corchetes que contienen otros corchetes, si tenemos dos conjuntos de corchetes = dim 2
## shape es para saber fila * columnas


### Crear un random tensor con un SHAPE similar a un tensor de una imagen
random_image_size_tensor = torch.rand(size= (224, 224, 3)) #altura, ancho y colores rojo, verde y azul


## Neural networks empiezan con numeros random -> miran la data -> los cambian -> vuelven a mirar -> y los actualizan otra vez