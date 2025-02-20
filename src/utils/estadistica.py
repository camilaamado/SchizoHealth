import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest, levene, ttest_ind, mannwhitneyu
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# ================================================================================
# FUNCION PARA COMPARAR DISTRIBUCIONES ENTRE CLUSTERS
# ================================================================================
def plot_distribution(data, title):
    """
    Genera gráficos de distribución en formato 2x2 utilizando Seaborn:
    - Histograma con KDE
    - Boxplot
    - Violin plot
    - Q-Q Plot

    Parámetros:
    - data: Serie de Pandas o lista de valores.
    - title: Título del gráfico.
    """
    data = np.array(data)  # Asegura que los datos sean un array de numpy

    plt.figure(figsize=(12, 10))

    # Histograma con KDE
    plt.subplot(2, 2, 1)
    sns.histplot(data, kde=True)
    plt.title(f'Histograma y KDE - {title}')

    # Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=data)
    plt.title(f'Boxplot - {title}')

    # Violin plot
    plt.subplot(2, 2, 3)
    sns.violinplot(x=data)
    plt.title(f'Violin Plot - {title}')

    # Q-Q Plot (Cálculo manual)
    plt.subplot(2, 2, 4)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot - {title}')

    # Ajuste automático de los espacios entre los gráficos
    plt.tight_layout()
    #plt.show()

# ================================================================================
# FUNCIONES PARA TESTS ESTADÍSTICOS
# ================================================================================
import numpy as np
from math import sqrt
from scipy.stats import kstest, levene, ttest_ind, mannwhitneyu, pearsonr

def kolmogorov_smirnov_test(data):
    """
    Prueba de Kolmogorov-Smirnov para normalidad.
    
    Parámetros:
    - data: Serie de Pandas o lista de valores.
    
    Retorna:
    - Estadístico KS y p-valor.
    """
    stat, p = kstest(data, 'norm')
    print(f"Kolmogorov-Smirnov Test: Estadístico={stat:.4f}, p-valor={p:.4f}")
    return stat, p

def levene_test(data_cluster1, data_cluster2):
    """
    Prueba de Levene para igualdad de varianzas.

    Parámetros:
    - data_cluster1: Datos del primer cluster.
    - data_cluster2: Datos del segundo cluster.

    Retorna:
    - Estadístico de Levene y p-valor.
    """
    stat, p = levene(data_cluster1, data_cluster2)
    print(f"Test de Levene: Estadístico={stat:.4f}, p-valor={p:.4f}")
    return stat, p

def t_test(data_cluster1, data_cluster2, equal_var=True):
    """
    Prueba T de Student para igualdad de medias.

    Parámetros:
    - data_cluster1: Datos del primer cluster.
    - data_cluster2: Datos del segundo cluster.
    - equal_var: Booleano que indica si se asumen varianzas iguales.

    Retorna:
    - Estadístico t y p-valor, y d de Cohen.
    """
    stat, p = ttest_ind(data_cluster1, data_cluster2, equal_var=equal_var)
    print(f"Test T de Student: Estadístico={stat:.4f}, p-valor={p:.4f}")
    
    # Calcular la d de Cohen
    mean1, mean2 = np.mean(data_cluster1), np.mean(data_cluster2)
    std1, std2 = np.std(data_cluster1, ddof=1), np.std(data_cluster2, ddof=1)
    cohen_d = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
    print(f"d de Cohen: {cohen_d:.4f}")
    
    # Calcular el r de Pearson
    r_pearson, _ = pearsonr(data_cluster1, data_cluster2)
    print(f"r de Pearson: {r_pearson:.4f}")

    return stat, p, cohen_d, r_pearson

def mann_whitney_test(data_cluster1, data_cluster2):
    """
    Prueba de Mann-Whitney U para comparar medianas cuando los datos NO son normales.

    Parámetros:
    - data_cluster1: Datos del primer cluster.
    - data_cluster2: Datos del segundo cluster.

    Retorna:
    - Estadístico U y p-valor.
    """
    stat, p = mannwhitneyu(data_cluster1, data_cluster2)
    print(f"Test de Mann-Whitney U: Estadístico={stat:.4f}, p-valor={p:.4f}")
    return stat, p

def glass_delta(data_cluster1, data_cluster2):
    """
    Calcular el tamaño del efecto Glass' delta.

    Parámetros:
    - data_cluster1: Datos del primer cluster.
    - data_cluster2: Datos del segundo cluster.

    Retorna:
    - Glass' delta.
    """
    mean1, mean2 = np.mean(data_cluster1), np.mean(data_cluster2)
    std2 = np.std(data_cluster2, ddof=1)  # Usamos la desviación estándar del segundo grupo

    glass_d = (mean1 - mean2) / std2
    print(f"Glass' delta: {glass_d:.4f}")
    
    return glass_d
