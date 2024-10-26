# AItea Building Lab

Aitea Building Lab es una herramienta cuya finalidad es la de crear modelos y algoritmos de machine learning, donde se pueden incluir redes neuronales u otras técnicas para exportarlas al estandar ONNX. Tambien es una plataforma para
testear estos modelos.
La herramienta tambien proporciona la posibilidad de convertir los modelos ONNX a TensorFlow Litle con la finalidad de que estos puedan correr en un sistema embebido tipo ESP32. 


# Entrada de los datos





# Compilación usando Nuitka
nuitka --module execution/testing_tool.py  --include-package=models_warehouse --include-package=metaclass --include-package=utils   --show-modules


