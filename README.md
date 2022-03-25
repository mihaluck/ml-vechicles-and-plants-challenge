# ml-vechicles-and-plants-challenge

Выполнение тестового задания по машинному обучению на классификацию изображений.

## Формулировка задачи

Есть набор из 3 000 изображений, а так же `data.json` файл, содержащий разметку - принадлежность каждого изображения к типу, категории и подкатегории, и набор тегов для каждого изображения. 

Используя python и [этот датасет](https://drive.google.com/drive/folders/1wHOf6eGv2esYtqFbBuGW9eigoZhRDmMZ?usp=sharing) постройте и обучите модель, которая среди предложенных изображений выявляет машины и растения, и выводит на экран список файлов в каждой категории с указанием вероятности совпадения. 

## Установка зависимостей

Предполагается, что на вашей машине установлен [python v3](https://www.python.org/downloads/) и [git](https://git-scm.com/downloads). 

Требуется установить дополнительные пакеты python:

```
pip install jupyterlab tensorflow pandas matplotlib sklearn numpy
```

## Клонирование репозитория

Клонировать репозиторий командой:

```
git clone https://github.com/mihaluck/ml-vechicles-and-plants-challenge.git
```

## Запуск проекта

В папке проеката выполнить команду: 

```
jupyter lab
```

В открывшенся окне jupyterlab внутри браузера выбрать файл `task.ipynb` и запустить его.

Результат работы программы появится в папке `Vehicles_vs_Plants`.
