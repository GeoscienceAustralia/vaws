* connection name and zone location

|zone| A | B | C |
|:-:|:----:|:---:|:---:|
|6| 6 | 12 | 18 |
|5| 5 | 11 | 17 |
|4| 4 | 10 | 16 |
|3| 3 | 9 | 15 |
|2| 2 | 8 | 14 |
|1| 1 | 7 | 13 |

* 

|conn| type | strength | Cpe |
|:-:|:----:|:---:|:---:|
|1| corner | 2.31| -0.1|
|2| gable| 1.54| -0.1|
|3| gable| 1.54| -0.1|
|4| gable| 1.54| -0.1|
|5| gable| 1.54| -1|
|6| corner| 2.31| -7|
|7| eave| 4.62| -0.1|
|8| sheeting| 2.695| -0.1|
|9| sheeting| 2.695| -1|
|10| sheeting| 2.695| -3.5|
|11| sheeting| 2.695| -1|
|12| eave| 4.62| -0.1|
|13| eave| 4.62| -7|
|14| sheeting| 2.695| -1|
|15| sheeting| 2.695| -0.1|
|16| sheeting| 2.695| -0.1|
|17|sheeting|2.695|-0.1|
|18| eave| 4.62| -0.1|

* connection - zone influence
	- 1:1 relation

* zone area

|zone| A(0) | B(1) | C(2) |
|:-:|:----:|:---:|:---:|
|6(5)| 0.2025 | 0.405 | 0.405 |
|5(4)| 0.405 | 0.81 | 0.81 |
|4(3)| 0.405 | 0.81 | 0.81 |
|3(2)| 0.405 | 0.81 | 0.81 |
|2(1)| 0.405 | 0.81 | 0.81 |
|1(0)| 0.2025 | 0.405 | 0.405 |


* costing area

- sheetinggable, 0.405 x 4 = 1.62 
- sheetingeave,0.405 x 4 = 1.62
- sheetingcorner,0.225 x 2 = 0.45
- sheeting,0.81 x 8 = 6.48


* logic 
 1. calculate qz (each wind speed)
 2. calculate zone pressures (