ACP = "Análisis de componentes principales". Para reducir la dimensión de lso datos, a fin de reducir el costo computacional en la ejecución.
Técnicamente lo que hace la ACP es buscar la proyección, segúin la cual los datos queden mejor representados en términos de los Mínimos Cuadrados. Los valores que no presentan una correlación lineal (Componentes principales).

ACP -> Se usa mucho en la fase de análisis exploratorio.
La idea fundamental, parte del "problema de la dimensión": una muestra aleatoria de n individuos con m rasgos observables (variables).
La matriz de correlaciones tendrá :
(m)
(2)  coeficientes. ("número factorial m tomados de a 2"). Notar como puede enlentecer mucho meter m variables...(var. redundantes, datos en exceso, etc).

Componentes principales: conjunto dado por el conjunto original de variables transformado en otro, en el cual ya no tengan info. relacionada entre si (info. redundante). De tal manera, que las nuevas variables, son combinaciones lineales de las anteriores y se van construyendo según el orden de importancia en cuanto a la variabilidad total que recogen de la muestra.

"Cuanto mayor sea la variabilidad de los datos, se considera que existe más información entre las variables".

Soluciones --> Pearson,Hoteling.


----------------------------
El objetivo es identificar variables que estén correlacionadas, para ir juntandolas o eliminandolas si son redundants, para ir quitando dimensión (si tiene sentido, ej. en presencia de cuellos de botella) para reducir el costo computacional.