
# Funcionamiento


## Instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python3 kneighbors.py
```

## Resultados

```
Ejemplo de partida con k=5 y distancia euclidiana
Number of incorrect predictions: 10
Accuracy: 72.22%
Precision: 81.82%
Recall: 90.0%
F1 Score: 85.71%
Comparacion usando diferentes valores de k, con datos estandarizados y diferentes tipos de distancia
#################################################################################
############################# Datos sin estandarizar ############################
#################################################################################
Distancia       k     Accuracy   Precision  Recall     F1 Score  
euclidean       1     69.44%     69.23%     90.0%      78.26%    
euclidean       2     61.11%     61.54%     80.0%      69.57%    
euclidean       3     75.0%      81.82%     90.0%      85.71%    
euclidean       4     75.0%      69.23%     90.0%      78.26%    
euclidean       5     72.22%     81.82%     90.0%      85.71%    
euclidean       6     72.22%     81.82%     90.0%      85.71%    
euclidean       7     69.44%     80.0%      80.0%      80.0%     
euclidean       8     69.44%     100.0%     90.0%      94.74%    
euclidean       9     66.67%     100.0%     70.0%      82.35%    
euclidean       10    63.89%     100.0%     70.0%      82.35%    
manhattan       1     83.33%     81.82%     90.0%      85.71%    
manhattan       2     75.0%      80.0%      80.0%      80.0%     
manhattan       3     77.78%     81.82%     90.0%      85.71%    
manhattan       4     75.0%      69.23%     90.0%      78.26%    
manhattan       5     77.78%     81.82%     90.0%      85.71%    
manhattan       6     83.33%     81.82%     90.0%      85.71%    
manhattan       7     72.22%     80.0%      80.0%      80.0%     
manhattan       8     77.78%     100.0%     100.0%     100.0%    
manhattan       9     75.0%      90.0%      90.0%      90.0%     
manhattan       10    77.78%     90.0%      90.0%      90.0%     
chebyshev       1     69.44%     75.0%      90.0%      81.82%    
chebyshev       2     63.89%     61.54%     80.0%      69.57%    
chebyshev       3     75.0%      80.0%      80.0%      80.0%     
chebyshev       4     72.22%     61.54%     80.0%      69.57%    
chebyshev       5     69.44%     80.0%      80.0%      80.0%     
chebyshev       6     69.44%     81.82%     90.0%      85.71%    
chebyshev       7     66.67%     72.73%     80.0%      76.19%    
chebyshev       8     66.67%     100.0%     80.0%      88.89%    
chebyshev       9     66.67%     87.5%      70.0%      77.78%    
chebyshev       10    66.67%     100.0%     70.0%      82.35%    
La mejor distancia/k es ('euclidean', 8) para maximizar el f1 score, con un f1 score de 94.74%
La mejor distancia/k es ('manhattan', 1) para maximizar la accuracy, con una accuracy de 83.33%
La mejor distancia/k es ('manhattan', 9) para maximizar la precision, con una precision de 90.0%

#################################################################################
############################# Datos estandarizados ##############################
#################################################################################
Distancia       k     Accuracy   Precision  Recall     F1 Score  
euclidean       1     91.67%     100.0%     100.0%     100.0%    
euclidean       2     91.67%     100.0%     100.0%     100.0%    
euclidean       3     91.67%     100.0%     100.0%     100.0%    
euclidean       4     91.67%     100.0%     100.0%     100.0%    
euclidean       5     91.67%     100.0%     100.0%     100.0%    
euclidean       6     91.67%     100.0%     100.0%     100.0%    
euclidean       7     86.11%     83.33%     100.0%     90.91%    
euclidean       8     86.11%     83.33%     100.0%     90.91%    
euclidean       9     83.33%     76.92%     100.0%     86.96%    
euclidean       10    88.89%     90.91%     100.0%     95.24%    
manhattan       1     91.67%     100.0%     100.0%     100.0%    
manhattan       2     91.67%     100.0%     100.0%     100.0%    
manhattan       3     91.67%     100.0%     100.0%     100.0%    
manhattan       4     91.67%     100.0%     100.0%     100.0%    
manhattan       5     91.67%     90.91%     100.0%     95.24%    
manhattan       6     91.67%     100.0%     100.0%     100.0%    
manhattan       7     88.89%     83.33%     100.0%     90.91%    
manhattan       8     88.89%     83.33%     100.0%     90.91%    
manhattan       9     91.67%     90.91%     100.0%     95.24%    
manhattan       10    88.89%     83.33%     100.0%     90.91%    
chebyshev       1     83.33%     76.92%     100.0%     86.96%    
chebyshev       2     88.89%     90.91%     100.0%     95.24%    
chebyshev       3     80.56%     76.92%     100.0%     86.96%    
chebyshev       4     88.89%     83.33%     100.0%     90.91%    
chebyshev       5     88.89%     83.33%     100.0%     90.91%    
chebyshev       6     88.89%     83.33%     100.0%     90.91%    
chebyshev       7     88.89%     83.33%     100.0%     90.91%    
chebyshev       8     88.89%     83.33%     100.0%     90.91%    
chebyshev       9     91.67%     83.33%     100.0%     90.91%    
chebyshev       10    91.67%     83.33%     100.0%     90.91%    
La mejor distancia/k es ('euclidean', 10) para maximizar el f1 score, con un f1 score de 95.24%
La mejor distancia/k es ('euclidean', 1) para maximizar la accuracy, con una accuracy de 91.67%
La mejor distancia/k es ('euclidean', 10) para maximizar la precision, con una precision de 90.91%
```