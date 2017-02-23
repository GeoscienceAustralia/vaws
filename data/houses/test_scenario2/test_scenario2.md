* connection name - zone location

|zone| A | B | C |
|:-:|:----:|:---:|:---:|
|6| 6 | 12 | 18 |
|5| 5 | 11 | 17 |
|4| 4 | 10 | 16 |
|3| 3 | 9 | 15 |
|2| 2 | 8 | 14 |
|1| 1 | 7 | 13 |

* connection type 
	- sheeting gable (1): 2, 3, 4, 5
	- sheeting eave (2): 7, 12, 13, 18
	- sheeting corner (3): 1, 6
	- sheeting (4): 8-11, 14-17

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