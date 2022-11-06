{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/viking316/kollege/blob/main/demo.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "# import matplotlib.pyplot as plt\n",
        "\n"
      ],
      "metadata": {
        "id": "1LAP-Yv7zUEY"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dataset = pd.read_csv('Data.csv')\n",
        "X = dataset.iloc[:, :-1].values\n",
        "Y = dataset.iloc[:, -1].values\n",
        "print(X)\n",
        "print(Y)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EqkHmmQpzfsX",
        "outputId": "f4ffe348-ece2-4389-a418-815079cc27a8"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 27.0 48000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 61000.0]\n",
            " ['Germany' 40.0 nan]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' nan 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 50.0 83000.0]\n",
            " ['France' 37.0 67000.0]]\n",
            "['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(X)\n",
        "from sklearn.impute import SimpleImputer\n",
        "imp = SimpleImputer(missing_values=np.nan,strategy='mean')\n",
        "print(X)\n",
        "imp.fit(X[:, 2:3])\n",
        "X[:, 2:3] = imp.transform(X[:, 2:3])\n",
        "print(X)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hq5J1H6W06bp",
        "outputId": "ec263b88-8ef4-4342-ce72-83c0ef7c67f8"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 27.0 48000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 61000.0]\n",
            " ['Germany' 40.0 63777.77777777778]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' 38.77777777777778 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 50.0 83000.0]\n",
            " ['France' 37.0 67000.0]]\n",
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 27.0 48000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 61000.0]\n",
            " ['Germany' 40.0 63777.77777777778]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' 38.77777777777778 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 50.0 83000.0]\n",
            " ['France' 37.0 67000.0]]\n",
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 27.0 48000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 61000.0]\n",
            " ['Germany' 40.0 63777.77777777778]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' 38.77777777777778 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 50.0 83000.0]\n",
            " ['France' 37.0 67000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')\n",
        "X = np.array(ct.fit_transform(X))\n",
        "print(X)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nYQr3g2B2yOe",
        "outputId": "40f610d2-e4b0-458e-9273-8c3472575399"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1.0 0.0 0.0 44.0 72000.0]\n",
            " [0.0 0.0 1.0 27.0 48000.0]\n",
            " [0.0 1.0 0.0 30.0 54000.0]\n",
            " [0.0 0.0 1.0 38.0 61000.0]\n",
            " [0.0 1.0 0.0 40.0 63777.77777777778]\n",
            " [1.0 0.0 0.0 35.0 58000.0]\n",
            " [0.0 0.0 1.0 38.77777777777778 52000.0]\n",
            " [1.0 0.0 0.0 48.0 79000.0]\n",
            " [0.0 1.0 0.0 50.0 83000.0]\n",
            " [1.0 0.0 0.0 37.0 67000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "le = LabelEncoder()\n",
        "Y = le.fit_transform(Y)\n",
        "print(Y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8LPNn0hK0p3y",
        "outputId": "8b40db13-d758-4b6e-ee5d-927ce0b8adbb"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 1 0 0 1 1 0 1 0 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "75wi-cEf08Px"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}