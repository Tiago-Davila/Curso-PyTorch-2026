## 00 Pytoch fundamentals

import torch
#2.15.0+cu132
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print(torch.__version__)


##Introduccion de Tensores ##################################################################################################################################
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

#Crear un random tensor of size (3, 4)##################################################################################################################################

random_tensor = torch.rand(3, 4, 4)

print (random_tensor)
print(random_tensor.ndim, random_tensor.shape)


##ndim es el numero de dimensiones, es decir de corchetes que contienen otros corchetes, si tenemos dos conjuntos de corchetes = dim 2
## shape es para saber fila * columnas


### Crear un random tensor con un SHAPE similar a un tensor de una imagen ##################################################################################################################################
random_image_size_tensor = torch.rand(size= (224, 224, 3)) #altura, ancho y colores rojo, verde y azul


## Neural networks empiezan con numeros random -> miran la data -> los cambian -> vuelven a mirar -> y los actualizan otra vez

### Ceros y Unos ##################################################################################################################################

#Todo ceros
zero = torch.zeros(3, 4)
print(zero)

ones = torch.ones(3, 4)
print(ones)


##dytype - default data type torch.float - float32
print(ones.dtype)


### Crear un rango de tensores y tipos de tensores
##torch.range

print((torch.arange(0, 10)))


one_to_ten = torch.arange(start=1, end=  10000, step=55) 
print (one_to_ten)

## tensors like

ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)


### Tensor Datatypes ########################################################################################
# los datatypes son de los tres problemas con los que se encuentra normalmente en Pythorch y DL
# 1. El tensor no tiene el DATATYPE correcto
# 2. El tensor no tiene el SHAPE correcto - para la multiplicacion de tensores
# 3. El tensor no esta en el DEVICE correcto - default "cuda"
float_32_tensor = torch.tensor([3.0, 3.2, 56.4], dtype=None, ### tipo de tensor ej float32, 16
                                                device=None, ### Que tipo de dispositivo corre el tensor
                                                requires_grad=False ## Wheter or not to track gradients with this tensors operations
)



### que significa en la documentacion de pythorch, los datatypes ej, 32-bit-floating-point
## tienen que ver con la capacidad de computo, y tiene que ver con la cantidad de numeros que sirven para la precision y detalle con la que el 
## modelo puede predecir


float_16_tensor = float_32_tensor.type(torch.float16)
print (float_16_tensor)
print (float_32_tensor)
print (float_16_tensor * float_32_tensor)


int_32_tensor = torch.tensor([3, 5, 6], dtype=torch.int32)
print(int_32_tensor*float_32_tensor)



### Obtuvimos informacion de los Tensores
# los datatypes son de los tres problemas con los que se encuentra normalmente en Pythorch y DL
# 1. El tensor no tiene el DATATYPE correcto - SOLUCION: para saber el datatype usar tensor.dtype y cambiar con tensor.type
# 2. El tensor no tiene el SHAPE correcto - para la multiplicacion de tensores - SOLUCION: para saber tensor.shape
# 3. El tensor no esta en el DEVICE correcto - default "cuda" - SOLUCION: para saber tensor.device



algun_tensor = torch.rand(3, 4)
print(algun_tensor)
print(f"El tipo del tensor es: ", algun_tensor.dtype)
print(f"El Shape del tensor es: ", algun_tensor.shape)
print(f"El device del tensor es: ", algun_tensor.device)


### Modificar Tensores (Operaciones de tensores) como suma, resta, multiplicacion (por elementos), division, multiplicacion de matrices
# para saber patrones de modelos hay una combinacion de las operaciones


## Suma de tensores 

tensor_uno = torch.tensor([2, 2, 3])
print(tensor_uno + 10) 
#multiplicacion
print(tensor_uno * 10)
#Resta 
print(tensor_uno - 10) 

# O 

print(torch.mul(tensor_uno, 10))
print(torch.add(tensor_uno, 10))
print(torch.sub(tensor_uno, 10))
print(torch.div(tensor_uno, 10))


#Multiplicacion de matrices

#hay dos formas de hacer multiplicacion en neural networks y DL
# Por elemento multiplicar, multiplicar cada elemento de la matriz por un numero
# Multiplicacion de matrices, multiplicar una matriz por otra matriz (dot product) 

print(torch.matmul(tensor_uno, tensor_uno)) ## 2 . 2 + 2. 2 + 3. 3 = 17

## o 

value = 0

for i in range (len(tensor_uno)): 
    value += tensor_uno[i] * tensor_uno[i]
print(value)

## Recorrer con for loop tarda mas tiempo que usar directamente una funcion propia de Pytorch



### Uno de los errores mas comunes de DL es SHAPE 
## para que la operacion de multiplicacion de matrices 
# 1. Las  dimensiones de cada matriz interna deben ser iguales  ej torch.matmul(tensor.rand([2,3]) * tensor.rand([2,3])) da error 
# 2. La matriz resultante tiene la shape de las dimensiones de afuera de la matriz ej ej torch.matmul(tensor.rand([2,3]) * tensor.rand([3,2]))
# va a dar un tensor.shape = ([2, 2])

tensor_prueba1 = torch.rand([3, 3])
tensor_prueba2 = torch.rand([3, 24])

print(torch.matmul(tensor_prueba1, tensor_prueba2))


 